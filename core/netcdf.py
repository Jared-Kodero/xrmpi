"""MPI-parallel NetCDF4 output for distributed xarray data.

Rank 0 creates the file through xarray's NetCDF4 store (``netcdf_backend``). Ranks then
reopen it with MPI-IO and each writes the hyperslabs it owns. The writer only sees
rank-local data plus the global partition metadata, so root-owned input is partitioned
by the caller (see :func:`xrmpi.core.io.to_netcdf`) before reaching it.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import netCDF4
import numpy as np
import xarray as xr

from ..mpi.diagnostics import MPIError
from ..mpi.mpi_init import MPI
from ..mpp.mpp import mpp_sync
from .chunks import validate_explicit_chunk_bytes
from .meta import mpp_get_meta, strip_mpi_meta
from .netcdf_backend import (
    agree_time_encoding,
    create_schema,
    default_encoding,
    encode_variable,
    time_kinds,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping
    from os import PathLike

    from ..mpi.context import MPIContext


class NetCDFWriteError(MPIError):
    """Parallel NetCDF write failure."""


def _open(path: str, comm: MPI.Comm, hints: str | None) -> netCDF4.Dataset:
    """Open ``path`` for writing, with MPI-IO when ``comm`` has several ranks."""
    if comm.size == 1:
        return netCDF4.Dataset(path, mode="r+")
    info = MPI.Info.Create()
    try:
        for item in filter(None, (part.strip() for part in (hints or "").split(";"))):
            key, separator, value = item.partition("=")
            if not separator or not key.strip():
                raise ValueError(f"Invalid MPI-IO hint {item!r}; expected key=value.")
            info.Set(key.strip(), value.strip())
        return netCDF4.Dataset(path, mode="r+", parallel=True, comm=comm, info=info)
    finally:
        info.Free()


def _owners(
    local: xr.Dataset, dims: tuple[str, ...], bounds: list[dict[str, tuple[int, int]]]
) -> dict[tuple[int, ...], list[Hashable]]:
    """Group distributed variables by the ranks that write them.

    A variable lacking some partition dimension is replicated along it; only the lowest
    rank holding each distinct non-empty hyperslab writes it. Variables with the same
    writers share one communicator.
    """
    groups: dict[tuple[int, ...], list[Hashable]] = {}
    for name, variable in local.variables.items():
        owned = [dim for dim in dims if dim in variable.dims]
        if not owned:
            continue
        writers: dict[tuple[tuple[int, int], ...], int] = {}
        for rank, rank_bounds in enumerate(bounds):
            region = tuple(rank_bounds[dim] for dim in owned)
            if all(stop > start for start, stop in region):
                writers.setdefault(region, rank)
        if writers:
            groups.setdefault(tuple(sorted(writers.values())), []).append(name)
    return groups


def mpp_to_netcdf_parallel(
    mpi_context: MPIContext,
    data: xr.Dataset | xr.DataArray,
    path: str | PathLike[str],
    *,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | str | None = None,
    zlib: bool = True,
    complevel: int = 4,
    shuffle: bool = True,
    chunks: Mapping[Hashable, Iterable[int]] | None = None,
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> str:
    """Write distributed ``data`` with MPI collective NetCDF4 I/O.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context; every rank must call this function.
    data : xarray.Dataset or xarray.DataArray
        Rank-local data carrying distribution metadata.
    path : str or os.PathLike
        Output file.
    encoding, unlimited_dims
        As in :meth:`xarray.Dataset.to_netcdf`.
    zlib, complevel, shuffle
        Default compression of numeric data variables; ``encoding`` takes precedence.
    chunks : mapping, optional
        Variable name to HDF5 chunk shape. Defaults to chunks aligned with the
        partition.
    hints : str, optional
        Semicolon-separated MPI-IO ``key=value`` hints.
    nofill : bool, default True
        Disable NetCDF pre-filling.
    allow_serial : bool, default False
        Allow a communicator with one rank.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    comm = mpi_context.comm
    if comm.size == 1 and not allow_serial:
        raise NetCDFWriteError(
            "Parallel NetCDF requires multiple MPI ranks; set allow_serial=True."
        )
    if comm.size > 1 and not netCDF4.__has_parallel4_support__:
        raise NetCDFWriteError("netCDF4 lacks parallel HDF5 support.")
    path = str(Path(path).expanduser().resolve(strict=False))

    error: BaseException | None = None
    try:
        dataset = data.to_dataset() if isinstance(data, xr.DataArray) else data
        meta = mpp_get_meta(dataset)
        if meta is None:
            raise ValueError("Parallel NetCDF output requires distributed data.")
        if chunks is not None:
            validate_explicit_chunk_bytes(
                dataset, {k: tuple(v) for k, v in chunks.items()}
            )
    except BaseException as exc:
        error = exc
    mpi_context.raise_if_error(error, "parallel NetCDF input validation")

    if chunks is None and "save_chunks" not in meta:
        from .io import mpp_attach_save_chunks

        mpp_attach_save_chunks(mpi_context, dataset)
        meta = mpp_get_meta(dataset)
    local = strip_mpi_meta(dataset)
    dims = tuple(str(dim) for dim in meta["dims"])
    sizes = {str(dim): int(size) for dim, size in meta["global_sizes"].items()}
    bounds = comm.allgather(
        {dim: (int(meta["starts"][dim]), int(meta["stops"][dim])) for dim in dims}
    )
    groups = _owners(local, dims, bounds)
    deferred = {name for names in groups.values() for name in names}
    if isinstance(unlimited_dims, str):
        unlimited_dims = (unlimited_dims,)
    unlimited = tuple(
        local.encoding.get("unlimited_dims", ())
        if unlimited_dims is None
        else unlimited_dims
    )

    options = default_encoding(
        local,
        encoding,
        zlib=zlib,
        complevel=complevel,
        shuffle=shuffle,
        chunks=chunks or meta.get("save_chunks"),
    )

    # Ranks writing slabs of one variable must share its time encoding, which xarray
    # would otherwise infer from each rank's own values. Replicated variables are
    # encoded by xarray itself.
    kinds: dict[Hashable, tuple[str, Any]] = {}
    for rank_kinds in comm.allgather(time_kinds(local.variables)):
        for name, (kind, sample) in rank_kinds.items():
            if kinds.get(name, (kind, None))[1] is None:
                kinds[name] = (kind, sample)
    for name, (kind, _) in kinds.items():
        if name not in deferred:
            continue
        options[name] = agree_time_encoding(
            comm,
            kind,
            local.variables[name],
            {**local[name].encoding, **options.get(name, {})},
        )
    full = {
        name: {**variable.encoding, **options.get(name, {})}
        for name, variable in local.variables.items()
    }

    error = None
    if comm.rank == 0:
        try:
            variables: dict[Hashable, xr.Variable] = {}
            for name, variable in local.variables.items():
                if name in deferred:
                    if variable.dtype.kind in "USO" and name not in kinds:
                        raise NetCDFWriteError(
                            f"Parallel NetCDF cannot write distributed variable {name!r}"
                            + f" of dtype {variable.dtype}."
                        )
                    shape = tuple(
                        sizes.get(str(dim), length)
                        for dim, length in zip(
                            variable.dims, variable.shape, strict=True
                        )
                    )
                    sample = kinds.get(name, (None, None))[1]
                    placeholder = (
                        da.full(shape, sample, dtype=object, chunks=-1)
                        if variable.dtype.kind == "O"
                        else da.zeros(shape, dtype=variable.dtype, chunks=-1)
                    )
                    variable = xr.Variable(variable.dims, placeholder, variable.attrs)
                else:
                    variable = variable.copy(deep=False)
                    variable.load()
                variable.encoding = full[name]
                variables[name] = variable
            coords = xr.Coordinates(
                {name: variables.pop(name) for name in local.coords}, indexes={}
            )
            schema = xr.Dataset(variables, coords, local.attrs)
            create_schema(path, schema, unlimited_dims=unlimited, nofill=nofill)
        except BaseException as exc:
            error = exc
    mpi_context.raise_if_error(error, "NetCDF file creation")
    mpp_sync(comm)

    try:
        for owners, names in sorted(groups.items()):
            member = comm.rank in owners
            writers = comm.Split(0 if member else MPI.UNDEFINED, comm.rank)
            handle = _open(path, writers, hints) if member else None
            try:
                for name in names:
                    error = values = None
                    if member:
                        try:
                            values = encode_variable(
                                name, local.variables[name], full[name]
                            ).values
                            stored = handle.variables[name].dtype
                            if (
                                values.dtype.kind == "f"
                                and stored.kind in "iu"
                                and not np.array_equal(values, values.astype(stored))
                            ):
                                raise NetCDFWriteError(
                                    f"{name!r}: encoded values are not representable as "
                                    + f"{stored}; set encoding[{name!r}]['dtype']."
                                )
                        except BaseException as exc:
                            error = exc
                    mpi_context.raise_if_error(error, f"NetCDF encoding of {name!r}")
                    if member:
                        variable = handle.variables[name]
                        variable.set_auto_maskandscale(False)
                        if writers.size > 1:
                            variable.set_collective(True)
                        index = tuple(
                            slice(*bounds[comm.rank][str(dim)])
                            if str(dim) in dims
                            else slice(None)
                            for dim in local.variables[name].dims
                        )
                        variable[index] = values
                        values = None
            finally:
                if handle is not None:
                    handle.close()
                if writers != MPI.COMM_NULL:
                    writers.Free()
            mpp_sync(comm)
    except MPIError:
        raise
    except BaseException:
        # One rank failing inside a collective write would leave the others blocked.
        traceback.print_exc()
        sys.stderr.flush()
        if comm.size > 1:
            comm.Abort(1)
        raise
    return path


__all__ = ["NetCDFWriteError", "mpp_to_netcdf_parallel"]
