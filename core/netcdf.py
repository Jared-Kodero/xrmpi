"""Provide MPI-aware NetCDF4 output for distributed xarray data."""

from __future__ import annotations

import contextlib
import math
import sys
import traceback
import warnings
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cftime
import dask
import netCDF4
import numpy as np
import xarray as xr
from xarray.coding.times import encode_cf_datetime, encode_cf_timedelta

from ...xgeo.core.progress import SerialProgressBar
from ..mpi.diagnostics import MPIError
from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v
from ..mpp.mpp import mpp_broadcast, mpp_sync
from .chunks import get_chunk_bounds, get_chunks, get_partition_chunk_size
from .halo import mpp_global_field_xr as mpp_global_field
from .meta import mpp_get_meta, strip_export_attrs
from .planning import mpp_resolve_comm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike
    from typing import Any, Literal

    from ..mpi.context import MPIContext


class NetCDFWriteError(MPIError):
    """Parallel NetCDF write failure."""


# Keep chunks below HDF5's 4 GiB hard limit; target half the limit for filter overhead.


def is_cftime(da: xr.DataArray) -> bool:
    """Report whether an object-dtype variable holds cftime datetimes."""
    if da.dtype != object:
        return False
    values = np.asarray(da.values).reshape(-1)
    return values.size > 0 and isinstance(values[0], cftime.datetime)


def is_time_like(da: xr.DataArray) -> bool:
    """Report whether a variable carries datetime, cftime or timedelta values.

    Returns
    -------
    bool
        ``True`` when the variable requires CF numeric encoding before it can be written
        through the ``netCDF4`` interface.

    """
    return (
        np.issubdtype(da.dtype, np.datetime64)
        or np.issubdtype(da.dtype, np.timedelta64)
        or is_cftime(da)
    )


def encode_time(
    da: xr.DataArray,
    units: str | None = None,
    calendar: str | None = None,
) -> xr.DataArray:
    """Encode datetime64, cftime or timedelta64 values to CF numeric values.

    Returns
    -------
    xarray.DataArray
        Numeric variable carrying ``units`` and, where applicable, ``calendar`` in both
        ``attrs`` and ``encoding``.

    """
    if np.issubdtype(da.dtype, np.datetime64) and not is_cftime(da):
        target_units = units or "seconds since 1970-01-01 00:00:00"
        target_calendar = calendar or "proleptic_gregorian"
        num, out_units, out_calendar = encode_cf_datetime(
            da,
            units=target_units,
            calendar=target_calendar,
            dtype=np.dtype("int64"),
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units, "calendar": out_calendar})
        encoded.encoding.update({"units": out_units, "calendar": out_calendar})
        return encoded

    if is_cftime(da):
        num, out_units, out_calendar = encode_cf_datetime(
            da,
            units=units or da.encoding.get("units"),
            calendar=calendar or da.encoding.get("calendar"),
            dtype=da.encoding.get("dtype"),
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units, "calendar": out_calendar})
        encoded.encoding.update({"units": out_units, "calendar": out_calendar})
        return encoded

    if np.issubdtype(da.dtype, np.timedelta64):
        num, out_units = encode_cf_timedelta(
            da,
            units=units or da.encoding.get("units"),
            dtype=da.encoding.get("dtype"),
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units})
        encoded.encoding.update({"units": out_units})
        return encoded

    return da


def encode_dataset_time(ds: xr.Dataset) -> xr.Dataset:
    """Encode every time-like variable of a dataset without touching the input."""
    out = ds.copy()
    replacements: dict[Any, xr.DataArray] = {}
    for name in list(out.variables):
        variable = out[name]
        if is_time_like(variable):
            replacements[name] = encode_time(variable)
    for name, variable in replacements.items():
        out[name] = variable
    return out


__all__ = [
    "encode_dataset_time",
    "encode_time",
    "is_cftime",
    "is_time_like",
]


def set_attrs(target: Any, attrs: Mapping[str, Any]) -> None:
    """Set serializable NetCDF attributes."""
    for key, value in strip_export_attrs(attrs).items():
        if key != "_FillValue" and value is not None:
            target.setncattr(str(key), value)


def _normalise_variable(
    source: xr.DataArray,
) -> tuple[np.ndarray[Any, Any], str | np.dtype[Any]]:
    """Normalize an xarray variable for NetCDF output."""
    variable = encode_time(source) if is_time_like(source) else source
    values = np.asarray(variable.values)

    if values.dtype.kind in ("U", "S", "O"):
        values = np.asarray(
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values.ravel()
            ],
            dtype=object,
        ).reshape(values.shape)
        return values, "str"

    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    if values.dtype.byteorder not in ("=", "|"):
        values = values.astype(values.dtype.newbyteorder("="))
    return np.ascontiguousarray(values), values.dtype


@contextlib.contextmanager
def quiet_netcdf4_writes():
    """Suppress the netCDF4 NumPy shape-assignment warning.

    Yields
    ------
    None
        Context with the warning filtered.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
        )
        yield


def preextend_unlimited(
    nc: netCDF4.Dataset,
    schema: Mapping[str, Any],
    name: str,
    ncvar: netCDF4.Variable,
) -> None:
    """Pre-extend unlimited dimensions before parallel writes."""
    dims = ncvar.dimensions
    unlimited = set(schema["unlimited_dim"])
    if not any(dim in unlimited for dim in dims):
        return

    index = tuple(
        int(schema["sizes"][dim]) - 1 if dim in unlimited else 0 for dim in dims
    )
    if any(position < 0 for position in index):
        return

    fill = ncvar.getncattr("_FillValue") if "_FillValue" in ncvar.ncattrs() else 0
    with quiet_netcdf4_writes():
        ncvar[index] = fill


def create_file(
    path: str,
    schema: Mapping[str, Any],
    root_data: Mapping[str, Mapping[str, Any]],
) -> None:
    """Create NetCDF metadata and write nonpartitioned data."""
    partition_dim = schema["partition_dim"]
    partition_dims = (
        ()
        if partition_dim is None
        else (partition_dim,)
        if isinstance(partition_dim, str)
        else tuple(partition_dim)
    )
    unlimited = set(schema["unlimited_dim"])
    chunks = schema["chunks"]
    prewritten = set(schema.get("prewritten", ()))

    with netCDF4.Dataset(path, mode="w", format="NETCDF4") as nc:
        if schema["nofill"]:
            nc.set_fill_off()

        for dim, length in schema["sizes"].items():
            nc.createDimension(dim, None if dim in unlimited else length)

        for name, variable in root_data.items():
            dims = variable["dims"]
            dtype = variable["dtype"]
            attrs = variable["attrs"]
            kwargs: dict[str, Any] = {}

            if "_FillValue" in attrs:
                kwargs["fill_value"] = attrs["_FillValue"]
            if name in chunks and dims and dtype != "str":
                kwargs["chunksizes"] = chunks[name]
            if (
                schema["deflate"] is not None
                and dims
                and dtype != "str"
                and not variable["coord"]
            ):
                kwargs["zlib"] = True
                kwargs["complevel"] = schema["deflate"]
                kwargs["shuffle"] = schema["shuffle"]

            ncvar = nc.createVariable(name, dtype, dims, **kwargs)
            set_attrs(ncvar, attrs)

            partitioned = (
                bool(partition_dims)
                and any(d in dims for d in partition_dims)
                and name not in prewritten
            )
            if not partitioned:
                with quiet_netcdf4_writes():
                    ncvar[...] = variable["data"]
            elif dtype != "str":
                preextend_unlimited(nc, schema, name, ncvar)

        set_attrs(nc, schema["attrs"])


def mpp_writer_comm(mpi_context: MPIContext, has_data: bool) -> MPI.Comm:
    """Create the communicator used for collective writes."""
    if mpi_context.comm.size == 1:
        return mpi_context.comm if has_data else MPI.COMM_NULL
    return mpi_context.comm.Split(
        1 if has_data else MPI.UNDEFINED, mpi_context.comm.rank
    )


def mpp_free_writer_comm(mpi_context: MPIContext, comm: MPI.Comm) -> None:
    """Free a writer communicator when required."""
    if comm != MPI.COMM_NULL and comm != mpi_context.comm:
        comm.Free()


def open_in_parallel(
    path: str,
    schema: Mapping[str, Any],
    comm: MPI.Comm,
) -> netCDF4.Dataset:
    """Open a NetCDF file for MPI I/O."""
    info: MPI.Info | None = None
    if comm.size > 1:
        info = MPI.Info.Create()
        try:
            for item in (schema["hints"] or "").split(";"):
                if not item.strip():
                    continue
                key, separator, value = item.partition("=")
                if not separator or not key.strip():
                    raise ValueError(
                        f"Invalid MPI-IO hint {item!r}; expected key=value."
                    )
                info.Set(key.strip(), value.strip())
            return netCDF4.Dataset(
                path,
                mode="r+",
                parallel=True,
                comm=comm,
                info=info,
            )
        finally:
            info.Free()
    return netCDF4.Dataset(path, mode="r+")


def mpp_close_writer(
    mpi_context: MPIContext, nc: netCDF4.Dataset | None, comm: MPI.Comm
) -> None:
    """Close a parallel NetCDF writer and synchronize ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    nc : netCDF4.Dataset or None
        Open writer on participating ranks.
    comm : mpi4py.MPI.Comm
        Writer communicator to free.
    """
    if nc is not None:
        nc.close()
    mpp_free_writer_comm(mpi_context, comm)
    mpp_sync(mpi_context.comm)


def mpp_write_distributed(
    mpi_context: MPIContext,
    path: str,
    schema: Mapping[str, Any],
    ds: xr.Dataset,
    meta: Mapping[str, Any],
) -> None:
    """Write rank-local slabs from distributed data."""

    partition_dims = set(meta["dims"])
    starts = meta["starts"]
    stops = meta["stops"]
    has_data = all(stops[dim] > starts[dim] for dim in partition_dims)
    prewritten = set(schema.get("prewritten", ()))

    comm = mpp_writer_comm(mpi_context, has_data)
    nc: netCDF4.Dataset | None = None
    try:
        if comm == MPI.COMM_NULL:
            return
        nc = open_in_parallel(path, schema, comm)
        for name, spec in schema["variables"].items():
            dims = tuple(spec["dims"])
            written_dims = partition_dims & set(dims)
            if not written_dims or name in prewritten:
                continue
            if spec["dtype"] == "str":
                raise NetCDFWriteError(
                    f"Parallel NetCDF cannot write partitioned string variable {name!r}."
                )

            values, _ = _normalise_variable(ds[name])
            ncvar = nc.variables[name]
            if comm.size > 1:
                ncvar.set_collective(True)
            index = tuple(
                slice(starts[dim], stops[dim]) if dim in written_dims else slice(None)
                for dim in dims
            )
            with quiet_netcdf4_writes():
                ncvar[index] = values
    finally:
        mpp_close_writer(mpi_context, nc, comm)


def mpp_write_partitioned(
    mpi_context: MPIContext,
    path: str,
    schema: Mapping[str, Any],
    source_ds: xr.Dataset | None,
) -> None:
    """Scatter root-owned data and write rank-local slabs."""
    partition_dim = schema["partition_dim"]
    if partition_dim is None:
        return
    prewritten = set(schema.get("prewritten", ()))

    length = int(schema["sizes"][partition_dim])
    # Align rank boundaries to the NetCDF chunk grid so writes do not straddle
    # partition-axis chunks.
    chunk_size = int(
        schema.get("partition_chunk_size")
        or max(1, math.ceil(length / mpi_context.comm.size))
    )
    bounds = [
        get_chunk_bounds(length, chunk_size, rank, mpi_context.comm.size)
        for rank in range(mpi_context.comm.size)
    ]
    counts = np.array([stop - start for start, stop in bounds], dtype=np.int64)
    start, stop = bounds[mpi_context.comm.rank]

    comm = mpp_writer_comm(mpi_context, stop > start)
    nc: netCDF4.Dataset | None = None
    try:
        if comm != MPI.COMM_NULL:
            nc = open_in_parallel(path, schema, comm)

        for name, spec in schema["variables"].items():
            dims = tuple(spec["dims"])
            if partition_dim not in dims or name in prewritten:
                continue
            if spec["dtype"] == "str":
                raise NetCDFWriteError(
                    f"Parallel NetCDF cannot write partitioned string variable {name!r}."
                )

            axis = dims.index(partition_dim)
            shape = tuple(int(value) for value in spec["shape"])
            moved_shape = (shape[axis], *shape[:axis], *shape[axis + 1 :])
            local_shape = (int(counts[mpi_context.comm.rank]), *moved_shape[1:])
            dtype = np.dtype(spec["dtype"])

            send = None
            if mpi_context.comm.rank == 0:
                if source_ds is None:
                    raise AssertionError("Rank 0 source Dataset is missing.")
                variable = source_ds[name]
                variable = encode_time(variable) if is_time_like(variable) else variable
                send = np.ascontiguousarray(
                    np.moveaxis(np.asarray(variable.values), axis, 0),
                    dtype=dtype,
                )

            local = mpi_context.scatterv(send, counts, local_shape, dtype)
            # Release each temporary scatter buffer promptly; ``source_ds`` still owns
            # eager backing arrays.
            send = None
            if nc is None:
                continue

            local = np.moveaxis(local, 0, axis)
            ncvar = nc.variables[name]
            if comm.size > 1:
                ncvar.set_collective(True)
            index = tuple(
                slice(start, stop) if dim == partition_dim else slice(None)
                for dim in dims
            )
            with quiet_netcdf4_writes():
                ncvar[index] = local
    finally:
        mpp_close_writer(mpi_context, nc, comm)


def _normalise_unlimited(
    unlimited_dim: str | Iterable[str] | None, sizes: Mapping[str, int]
) -> tuple[str, ...]:
    """Normalise an unlimited-dimension request and check it against ``sizes``."""
    if unlimited_dim is None:
        unlimited: tuple[str, ...] = ()
    elif isinstance(unlimited_dim, str):
        unlimited = (unlimited_dim,)
    else:
        unlimited = tuple(unlimited_dim)
    missing = set(unlimited) - set(sizes)
    if missing:
        raise ValueError(f"Unknown unlimited dimensions: {sorted(missing)}.")
    return unlimited


def _check_deflate(deflate: int | None) -> int | None:
    """Validate a deflate level."""
    if deflate is not None and not 0 <= int(deflate) <= 9:
        raise ValueError("deflate must be None or an integer in [0, 9].")
    return None if deflate is None else int(deflate)


def _variable_record(
    name: str,
    dims: tuple[Hashable, ...],
    dtype: Any,
    shape: tuple[int, ...],
    values: Any,
    attrs: Mapping[str, Any],
    is_coord: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the rank-0 buffer entry and the broadcast schema entry for a variable.

    Returns
    -------
    tuple[dict, dict]
        ``(root_entry, schema_entry)``. ``values`` is None for data that each
        rank writes into its own hyperslab later.
    """
    return (
        {
            "attrs": dict(attrs),
            "data": values,
            "dims": dims,
            "dtype": dtype,
            "coord": is_coord,
        },
        {
            "coord": is_coord,
            "dims": dims,
            "dtype": "str" if dtype == "str" else np.dtype(dtype).str,
            "shape": tuple(int(length) for length in shape),
        },
    )


def _gather_partitioned_coords(
    mpi_context: MPIContext,
    local_ds: xr.Dataset,
    local_meta: Mapping[str, Any],
    partition_dims: tuple[str, ...],
) -> dict[str, xr.DataArray]:
    """Reassemble every coordinate that is split across ranks.

    A coordinate lying along a partition axis exists only in pieces, so the
    schema cannot be written until the pieces are joined. Each is gathered on
    the root of the communicator that varies along its own axis.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    local_ds : xarray.Dataset
        This rank's data.
    local_meta : mapping
        Distribution metadata for ``local_ds``.
    partition_dims : tuple of str
        Dimensions the dataset is partitioned over.

    Returns
    -------
    dict[str, xarray.DataArray]
        Reassembled coordinates, populated on gathering roots only.

    Raises
    ------
    NetCDFWriteError
        If a coordinate spans more than one partition dimension.
    """
    error: BaseException | None = None
    coord_names: list[tuple[str, str]] = []
    try:
        for name, coord in local_ds.coords.items():
            touched = [d for d in partition_dims if d in coord.dims]
            if not touched:
                continue
            if len(touched) > 1:
                raise NetCDFWriteError(
                    f"Coordinate {name!r} spans multiple partition dims: "
                    + f"{tuple(touched)!r}."
                )
            coord_names.append((str(name), touched[0]))
    except BaseException as exc:
        coord_names = []
        error = exc
    mpi_context.raise_if_error(
        error, "parallel NetCDF coordinate discovery", signature=tuple(coord_names)
    )

    gathered: dict[str, xr.DataArray] = {}
    for coord_name, coord_dim in coord_names:
        dim_comm = (
            mpi_context.comm
            if len(partition_dims) == 1
            else mpp_resolve_comm(mpi_context, local_meta, (coord_dim,))
        )
        try:
            rebuilt = mpp_global_field(
                mpi_context,
                local_ds[coord_name],
                coord_dim,
                dim_comm,
                start=int(local_meta["starts"][coord_dim]),
                stop=int(local_meta["stops"][coord_dim]),
                global_size=int(local_meta["global_sizes"][coord_dim]),
            )
            if rebuilt is not None:
                gathered[coord_name] = rebuilt
        except BaseException as exc:
            error = exc
        mpi_context.raise_if_error(
            error, f"parallel NetCDF coordinate gather ({coord_name})"
        )
    return gathered


def mpp_to_netcdf_parallel(
    mpi_context: MPIContext,
    data: xr.Dataset | xr.DataArray | None,
    path: str | PathLike[str],
    partition_dim: str | None = None,
    deflate: int | None = None,
    shuffle: bool = True,
    chunks: Mapping[str, Iterable[int]] | None = None,
    unlimited_dim: str | Iterable[str] | None = (),
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> str:
    """Write an xarray object with MPI collective NetCDF I/O."""
    if mpi_context.comm.size == 1 and not allow_serial:
        raise NetCDFWriteError(
            "Parallel NetCDF requires multiple MPI ranks; set allow_serial=True."
        )
    if mpi_context.comm.size > 1 and not getattr(
        netCDF4, "__has_parallel4_support__", False
    ):
        raise NetCDFWriteError("netCDF4 lacks parallel HDF5 support.")

    # Rank-local validation is collected rather than raised. Only rank 0 holds
    # real data on the scatter path, so raising here would abort one rank while
    # every other rank waited in the following collectives.
    local_ds: xr.Dataset | None = None
    error: BaseException | None = None
    try:
        if isinstance(data, xr.DataArray):
            if data.name is None:
                raise ValueError("DataArray must have a name for NetCDF output.")
            # ``DataArray.to_dataset`` moves attrs to the variable; ``mpp_get_meta``
            # checks there.
            local_ds = data.to_dataset()
        elif isinstance(data, xr.Dataset):
            local_ds = data
        elif data is not None:
            raise TypeError("data must be an xarray Dataset, DataArray, or None.")
    except BaseException as exc:
        error = exc
    mpi_context.raise_if_error(error, "parallel NetCDF input validation")

    local_meta = mpp_get_meta(local_ds) if local_ds is not None else None
    distributed = local_meta is not None

    # The distributed and scatter paths post different collectives, so every
    # rank must take the same one. Disagreement is reported instead of hanging.
    agreed = gather_v(distributed, mpi_context.comm)
    if any(agreed) and not all(agreed):
        disagreeing = [rank for rank, state in enumerate(agreed) if state != agreed[0]]
        raise NetCDFWriteError(f"MPI ranks disagree on mpi_meta state: {disagreeing}.")

    root_data: dict[str, dict[str, Any]] | None = None
    schema: dict[str, Any] | None = None
    output_path: str | None = None
    error = None

    if not distributed:
        # Use lazy distribution only for Dask-backed root data; eager arrays stay on
        # ``Scatterv``.
        is_dask_backed = False
        try:
            if mpi_context.comm.rank == 0 and local_ds is not None:
                is_dask_backed = any(
                    dask.is_dask_collection(variable.data)
                    for variable in local_ds.variables.values()
                )
        except BaseException as exc:
            error = exc
        mpi_context.raise_if_error(error, "parallel NetCDF dask-backed detection")
        is_dask_backed = mpp_broadcast(is_dask_backed, mpi_context.comm, root=0)

        if is_dask_backed:
            # Import locally to avoid the ``.io``/``.netcdf`` cycle.
            from .io import distribute_data

            local_ds = distribute_data(
                local_ds if mpi_context.comm.rank == 0 else None,
                mpi_context,
                dim=partition_dim if partition_dim is not None else "auto",
                root=0,
            )
            local_meta = mpp_get_meta(local_ds)
            distributed = local_meta is not None

    if distributed:
        if local_ds is None or local_meta is None:
            raise AssertionError("Distributed data and metadata are missing.")

        distributed_dims = tuple(str(d) for d in local_meta["dims"])
        if len(distributed_dims) == 1:
            distributed_dim = distributed_dims[0]
            if partition_dim is not None and partition_dim != distributed_dim:
                error = ValueError(
                    f"partition_dim {partition_dim!r} does not match "
                    + f"distributed dimension {distributed_dim!r}"
                )
            partition_dim = distributed_dim
        else:
            if partition_dim is not None and partition_dim not in distributed_dims:
                error = ValueError(
                    f"partition_dim {partition_dim!r} does not match any of "
                    + f"the distributed dimensions {distributed_dims!r}"
                )
            partition_dim = distributed_dims
        mpi_context.raise_if_error(error, "parallel NetCDF partition dimension")

        # Plan save chunks collectively from global metadata before rank-0 schema
        # construction.
        if chunks is None:
            # Import locally to avoid the ``.io``/``.netcdf`` cycle.
            from .io import mpp_attach_save_chunks

            mpp_attach_save_chunks(mpi_context, local_ds)
            local_meta = mpp_get_meta(local_ds)
            if local_meta is None:
                raise AssertionError("attach_save_chunks cleared mpi_meta.")

        partition_dims_tuple = (
            (partition_dim,) if isinstance(partition_dim, str) else tuple(partition_dim)
        )
        global_sizes_map = local_meta["global_sizes"]
        prewritten_coords = _gather_partitioned_coords(
            mpi_context, local_ds, local_meta, partition_dims_tuple
        )

        # No data gather/scatter. Rank 0 only constructs the schema from its
        # local metadata and mpi_meta's global partition length.
        if mpi_context.comm.rank == 0:
            try:
                ds = local_ds
                _check_deflate(deflate)
                sizes = dict(ds.sizes)
                for d in partition_dims_tuple:
                    sizes[d] = int(global_sizes_map[d])
                unlimited = _normalise_unlimited(unlimited_dim, sizes)

                chunk_map = (
                    get_chunks(
                        ds,
                        chunks,
                        partition_dim if len(partition_dims_tuple) == 1 else None,
                        sizes[partition_dim]
                        if len(partition_dims_tuple) == 1
                        else None,
                    )
                    if chunks is not None
                    else local_meta["save_chunks"]
                )
                root_data = {}
                variables: dict[str, dict[str, Any]] = {}
                for name, source in ds.variables.items():
                    if name in prewritten_coords:
                        source = prewritten_coords[name]
                    variable = encode_time(source) if is_time_like(source) else source
                    values, dtype = _normalise_variable(source)
                    attrs = strip_export_attrs(variable.attrs)
                    dims = tuple(variable.dims)
                    shape = list(values.shape)
                    for d in partition_dims_tuple:
                        if d in dims:
                            shape[dims.index(d)] = sizes[d]

                    root_data[name], variables[name] = _variable_record(
                        name,
                        dims,
                        dtype,
                        tuple(shape),
                        values,
                        attrs,
                        name in ds.coords,
                    )

                output_path = str(Path(path).expanduser().resolve(strict=False))
                schema = {
                    "attrs": strip_export_attrs(ds.attrs),
                    "chunks": chunk_map,
                    "deflate": None if deflate is None else int(deflate),
                    "hints": hints,
                    "nofill": bool(nofill),
                    "partition_dim": partition_dim,
                    "prewritten": tuple(sorted(prewritten_coords)),
                    "shuffle": bool(shuffle),
                    "sizes": sizes,
                    "unlimited_dim": unlimited,
                    "variables": variables,
                }
            except BaseException as exc:
                error = exc
    elif mpi_context.comm.rank == 0:
        try:
            if local_ds is None:
                raise TypeError("Rank 0 must provide an xarray Dataset or DataArray.")
            ds = local_ds
            if partition_dim is None:
                partition_dim = next(
                    (da.dims[0] for da in ds.data_vars.values() if da.dims),
                    None,
                )
            elif partition_dim not in ds.sizes:
                raise ValueError(f"Unknown partition dimension {partition_dim!r}.")

            _check_deflate(deflate)
            unlimited = _normalise_unlimited(unlimited_dim, ds.sizes)

            partition_chunk_size = get_partition_chunk_size(
                ds, partition_dim, mpi_context.comm.size
            )
            chunk_map = get_chunks(ds, chunks, partition_dim, partition_chunk_size)
            # Root-owned coordinates are already global and can be written directly.
            prewritten_names = frozenset(
                name for name in ds.coords if partition_dim in ds[name].dims
            )
            root_data = {}
            variables = {}
            for name, source in ds.variables.items():
                dims = tuple(source.dims)
                partitioned = partition_dim in dims and name not in prewritten_names

                if partitioned and not is_time_like(source):
                    # Defer partitioned data; keep time-like values eager for safe encoding.
                    if source.dtype.kind in ("U", "S", "O"):
                        dtype: str | np.dtype[Any] = "str"
                    elif source.dtype.kind == "b":
                        dtype = np.dtype(np.int8)
                    elif source.dtype.byteorder not in ("=", "|"):
                        dtype = source.dtype.newbyteorder("=")
                    else:
                        dtype = source.dtype
                    root_data[name], variables[name] = _variable_record(
                        name,
                        dims,
                        dtype,
                        tuple(source.shape),
                        None,
                        strip_export_attrs(source.attrs),
                        name in ds.coords,
                    )
                    continue

                variable = encode_time(source) if is_time_like(source) else source
                values, dtype = _normalise_variable(source)
                root_data[name], variables[name] = _variable_record(
                    name,
                    tuple(variable.dims),
                    dtype,
                    values.shape,
                    values,
                    strip_export_attrs(variable.attrs),
                    name in ds.coords,
                )

            output_path = str(Path(path).expanduser().resolve(strict=False))
            schema = {
                "attrs": strip_export_attrs(ds.attrs),
                "chunks": chunk_map,
                "deflate": None if deflate is None else int(deflate),
                "hints": hints,
                "nofill": bool(nofill),
                "partition_chunk_size": partition_chunk_size,
                "partition_dim": partition_dim,
                "prewritten": tuple(sorted(prewritten_names)),
                "shuffle": bool(shuffle),
                "sizes": dict(ds.sizes),
                "unlimited_dim": unlimited,
                "variables": variables,
            }

        except BaseException as exc:
            error = exc

    mpi_context.raise_if_error(error, "parallel NetCDF preparation")
    output_path, schema = mpp_broadcast((output_path, schema), mpi_context.comm, root=0)
    if output_path is None or schema is None:
        raise AssertionError("Rank 0 did not broadcast the NetCDF schema.")

    error = None
    if mpi_context.comm.rank == 0:
        try:
            if root_data is None:
                raise AssertionError("Rank 0 data buffers are missing.")
            create_file(output_path, schema, root_data)
        except BaseException as exc:
            error = exc
    mpi_context.raise_if_error(error, "serial NetCDF schema creation")
    mpp_sync(mpi_context.comm)

    try:
        if distributed:
            if local_ds is None or local_meta is None:
                raise AssertionError("Distributed rank-local data are missing.")
            mpp_write_distributed(
                mpi_context, output_path, schema, local_ds, local_meta
            )
        else:
            mpp_write_partitioned(mpi_context, output_path, schema, local_ds)
    except BaseException:
        # Aborting without a diagnostic leaves the job log with nothing but
        # "MPI_ABORT was invoked", so the failure is reported first.
        traceback.print_exc()
        sys.stderr.flush()
        if mpi_context.comm.size > 1:
            mpi_context.comm.Abort(1)
        raise

    mpp_sync(mpi_context.comm)
    return output_path


def resolve_unlimited_dim(
    unlimited_dim: str | Iterable[str] | None, sizes: Iterable[str]
) -> str | None:
    """Reduce an unlimited-dimension specification to a single name.

    Raises
    ------
    TypeError
        If the specification is neither a string nor an iterable of strings.
    ValueError
        If the requested dimension is absent from the data.

    """
    if unlimited_dim is None:
        return None

    if isinstance(unlimited_dim, str):
        name = unlimited_dim
    else:
        try:
            names = [item for item in unlimited_dim]
        except TypeError as exc:
            raise TypeError(
                "unlimited_dim must be str or iterable[str], got "
                + f"{type(unlimited_dim).__name__}."
            ) from exc
        if not names:
            return None
        if any(not isinstance(item, str) for item in names):
            raise TypeError("Every unlimited_dim entry must be a string.")
        name = names[0]

    known = list(sizes)
    if name not in known:
        raise ValueError(f"{name!r} is not a dimension in data; available: {known}.")
    return name


def to_netcdf_serial(
    data: xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    unlimited_dim: str | Iterable[str] | None = None,
    *,
    batch_size: int = 24,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
) -> None:
    """Write a Dataset or DataArray serially to NetCDF.

    Returns
    -------
    None

    """
    if isinstance(data, xr.DataArray) and not Path(file).exists():
        if data.name is None:
            raise ValueError("DataArray must have a name to create a new file.")
        data = data.to_dataset()

    if isinstance(data, xr.Dataset):
        dataset_to_netcdf(
            file=file,
            data=data,
            unlimited_dim=unlimited_dim,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
        )
        return

    dataarray_to_netcdf(
        file=file,
        da=data,
        format=format,
        shuffle=shuffle,
        zlib=zlib,
        complevel=complevel,
    )


def dataset_to_netcdf(
    file: str | PathLike[str],
    data: xr.Dataset,
    unlimited_dim: str | Iterable[str] | None = None,
    batch_size: int = 1,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
) -> None:
    """Write a Dataset, defining the file once and appending in batches.

    Returns
    -------
    None

    """
    file = Path(file)
    file.unlink(missing_ok=True)

    dim0 = resolve_unlimited_dim(unlimited_dim, data.sizes)
    if dim0 is None:
        dim0 = next(iter(data.sizes))

    n_items = data.sizes[dim0]

    if n_items < 1:
        raise ValueError(f"Cannot write an empty dimension: {dim0!r}.")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    # Encode on a copy. Assigning encoded variables back into the argument
    # would leave the caller holding an integer time axis after this returns.
    data = encode_dataset_time(data)

    # First write defines the file, dimensions, variables, attrs, and encodings.
    # Keep this as a single record.

    data0 = data.isel({dim0: slice(0, 1)})

    enc = {
        v: {"zlib": zlib, "complevel": complevel, "shuffle": shuffle}
        for v in data0.data_vars
    }

    data0.to_netcdf(file, encoding=enc, format=format, unlimited_dims=[dim0])

    # Append the remaining records in batches.
    starts = range(1, n_items, batch_size)

    if show_progress:
        data_slices = SerialProgressBar(
            starts, description="Writing NetCDF file", file=stdout
        )
    else:
        data_slices = starts

    for start in data_slices:
        stop = min(start + batch_size, n_items)

        nc_append(
            data.isel({dim0: slice(start, stop)}),
            file,
            dim=dim0,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            encoded_dataset=True,
        )


def dataarray_to_netcdf(
    file: str | PathLike[str],
    da: xr.DataArray,
    format: str = "NETCDF4",
    shuffle: bool | None = None,
    zlib: bool | None = None,
    complevel: int | None = None,
) -> None:
    """Write / append a DataArray to a NetCDF file"""

    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray.DataArray")

    if not Path(file).exists():
        raise FileNotFoundError(f"File {file!r} does not exist!")

    with netCDF4.Dataset(file, mode="r+", format=format) as ncf:
        varname = da.name
        if varname is None:
            raise ValueError("DataArray must have a name.")

        if is_time_like(da):
            stored = ncf.variables.get(varname)
            units = getattr(stored, "units", None) if stored is not None else None
            calendar = getattr(stored, "calendar", None) if stored is not None else None
            da = encode_time(da, units=units, calendar=calendar)

        # Overwrite values if the variable was created on a previous run.
        if varname in ncf.variables:
            ncf.variables[varname][:] = da.values

        else:
            ncvar = createVariable(
                ncf,
                da,
                varname,
                zlib=zlib,
                complevel=complevel,
                shuffle=shuffle,
                write_values=False,
            )

            with quiet_netcdf4_writes():
                ncvar[:] = da.values


def nc_append(
    data: xr.Dataset,
    file: str | PathLike[str],
    dim: str = "time",
    mode: Literal["a", "r+"] = "r+",
    format: str = "NETCDF4",
    shuffle: bool | None = None,
    zlib: bool | None = None,
    complevel: int | None = None,
    encoded_dataset: bool = False,
) -> None:
    """Append a Dataset along an unlimited dimension."""

    if isinstance(data, xr.DataArray):
        ds = data.to_dataset()
    else:
        ds = data

    if dim not in ds.sizes:
        raise ValueError(f"Append dimension {dim!r} not present in the data")

    n_new = ds.sizes[dim]

    with netCDF4.Dataset(file, mode=mode, format=format) as ncf:
        if dim not in ncf.dimensions:
            raise ValueError(f"Append dimension {dim!r} not found in {file}")

        # Confirm dim is the unlimited axis, and report the actual one on mismatch.
        unlimited = [d for d, o in ncf.dimensions.items() if o.isunlimited()]
        if dim not in unlimited:
            raise ValueError(
                f"{dim!r} is not unlimited in {file}; available: "
                + f"{unlimited or 'none'}."
            )

        offset = ncf.dimensions[dim].size

        encoded_arrays: dict[Any, xr.DataArray] = {}
        if encoded_dataset:
            encoded_arrays = {**ds.coords, **ds.data_vars}
        else:
            for varname, da in {**ds.coords, **ds.data_vars}.items():
                if not is_time_like(da):
                    encoded_arrays[varname] = da
                    continue

                stored = ncf.variables.get(varname)
                units = getattr(stored, "units", None)
                calendar = (
                    getattr(stored, "calendar", None) if stored is not None else None
                )
                encoded_arrays[varname] = encode_time(
                    da, units=units if stored is not None else None, calendar=calendar
                )

        for varname, da in encoded_arrays.items():
            exists = varname in ncf.variables
            # Static variables: write once on creation, then leave untouched.
            if dim not in da.dims:
                if not exists:
                    _ = createVariable(
                        ncf,
                        da,
                        varname,
                        zlib=zlib,
                        complevel=complevel,
                        shuffle=shuffle,
                        write_values=True,
                    )
                continue

            # First append for this variable creates it with size 0 along dim.
            if not exists:
                _ = createVariable(
                    ncf,
                    da,
                    varname,
                    zlib=zlib,
                    complevel=complevel,
                    shuffle=shuffle,
                    write_values=False,
                )

            ncvar = ncf.variables[varname]
            arr = da.transpose(*ncvar.dimensions).values

            if arr.dtype.kind in "mMO":
                raise TypeError(
                    f"Variable {varname!r} has unencoded dtype {arr.dtype}."
                )

            if ncvar.dtype != arr.dtype:
                arr = arr.astype(ncvar.dtype)

            index = tuple(
                slice(offset, offset + n_new) if d == dim else slice(None)
                for d in ncvar.dimensions
            )
            with quiet_netcdf4_writes():
                ncvar[index] = arr


def createVariable(
    ncf: netCDF4.Dataset,
    da: xr.DataArray,
    varname: str,
    zlib: bool | None = None,
    complevel: int | None = None,
    shuffle: bool | None = None,
    write_values: bool = False,
) -> netCDF4.Variable:
    """Execute createVariable."""
    missing = [d for d in da.dims if d not in ncf.dimensions]
    if missing:
        raise ValueError(f"Cannot create {varname}: missing dimensions {missing}.")

    kwargs = {}
    if zlib is not None:
        kwargs["zlib"] = zlib
    if complevel is not None:
        kwargs["complevel"] = complevel
    if shuffle is not None:
        kwargs["shuffle"] = shuffle

    ncvar = ncf.createVariable(
        varname=varname, datatype=da.dtype, dimensions=da.dims, **kwargs
    )
    for attr_name, attr_val in strip_export_attrs(da.attrs).items():
        ncvar.setncattr(attr_name, attr_val)

    if write_values:
        with quiet_netcdf4_writes():
            ncvar[:] = da.values

    return ncvar


__all__ = [
    "NetCDFWriteError",
    "createVariable",
    "dataarray_to_netcdf",
    "dataset_to_netcdf",
    "mpp_to_netcdf_parallel",
    "mpp_writer_comm",
    "nc_append",
    "resolve_unlimited_dim",
    "to_netcdf_serial",
]
