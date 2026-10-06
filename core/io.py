"""Provide xarray I/O and redistribution across MPI ranks."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from ..mpi.context import MPIContext
from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v

if TYPE_CHECKING:
    from .core import MPIXarray

from ..mpp.mpp import mpp_sync
from ..mpp.mpp_domains_define import mpp_define_domains
from .chunks import (
    compute_save_chunks,
    get_chunk_bounds,
    get_chunk_info,
    get_effective_chunk_size,
)
from .distribute import (
    _as_partition_dims,
    mpp_create_dataarray,
    mpp_create_dataset,
    mpp_partition,
)
from .meta import (
    choose_partition_dim,
    mpp_get_meta,
    mpp_log_partition_report,
    mpp_should_log_partitions,
    mpp_update_meta,
    set_save_chunks,
)
from .netcdf import mpp_to_netcdf_parallel, nc_append, to_netcdf_serial

__all__ = [
    "empty_distributed_dataset",
    "is_distributed_empty",
    "nc_append",
    "to_netcdf",
]

_NO_DATA_ATTR = "_climtools_no_data"


def _open_partitioned(
    mpi_context: MPIContext,
    filename_or_obj: Any,
    dims: tuple[Hashable, ...],
    open_fn: Callable[..., xr.Dataset],
    chunks: Any,
    log_partitions: bool,
    automatic: bool,
    kwargs: dict[str, Any],
) -> xr.Dataset:
    """Open a Dataset lazily and keep only this rank's slice.

    Rank 0 reads the header alone and broadcasts the global shape, so the
    other ranks never touch the file until they open their own slice. As in
    FMS, one axis and several differ only in how the per-axis bounds are
    chosen: a single axis follows the file's chunk boundaries, a process grid
    divides each axis evenly.
    """
    comm = mpi_context.comm

    plan: dict[str, Any] | None = None
    error: BaseException | None = None
    if mpi_context.is_root():
        try:
            with open_fn(filename_or_obj, chunks=None, **kwargs) as metadata:
                resolved = dims
                if automatic:
                    resolved = (
                        choose_partition_dim(metadata.sizes, comm.size, rank=comm.rank),
                    )
                for d in resolved:
                    if d not in metadata.dims:
                        raise ValueError(f"Unknown partition dimension {d!r}.")
                plan = {
                    "dims": resolved,
                    "global_sizes": {d: int(metadata.sizes[d]) for d in resolved},
                    "chunk_info": get_chunk_info(metadata, comm.size),
                }
        except BaseException as exc:
            error = exc

    mpi_context.raise_if_error(error, "open_dataset planning")
    plan = mpi_context.broadcast(plan, root=0)
    dims = plan["dims"]
    global_sizes = plan["global_sizes"]
    chunk_info = plan["chunk_info"]

    cart = None
    if len(dims) == 1:
        dim = dims[0]
        bounds = {
            dim: get_chunk_bounds(
                global_sizes[dim], chunk_info[str(dim)], comm.rank, comm.size
            )
        }
    else:
        domain = mpp_define_domains(mpi_context, global_sizes, dims)
        bounds = {d: (domain.starts[d], domain.stops[d]) for d in dims}
        cart = domain.cart

    # Hold every rank here so none starts reading before the plan is settled.
    mpp_sync(comm)

    data: xr.Dataset = open_fn(filename_or_obj, chunks=chunks, **kwargs)
    data = data.isel({d: slice(*bounds[d]) for d in dims})

    if cart is not None:
        chunk_info = {
            str(name): get_effective_chunk_size(int(length), None, comm.size)
            for name, length in data.sizes.items()
        }

    starts = {d: bounds[d][0] for d in dims}
    stops = {d: bounds[d][1] for d in dims}
    single = len(dims) == 1 and cart is None
    mpp_update_meta(
        data,
        dim=dims[0] if single else dims,
        global_size=global_sizes[dims[0]] if single else global_sizes,
        start=starts[dims[0]] if single else starts,
        stop=stops[dims[0]] if single else stops,
        chunk_info=chunk_info,
        cart=cart,
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        mpp_log_partition_report(
            mpi_context,
            data,
            dims[0] if single else dims,
            origin="open_dataset",
            global_size=global_sizes[dims[0]] if single else global_sizes,
            start=starts[dims[0]] if single else starts,
            stop=stops[dims[0]] if single else stops,
            grid_shape=None if cart is None else cart["grid_shape"],
            coords=None if cart is None else cart["coords"],
            automatic=automatic,
        )
    return data


# mpi4py point-to-point tag for mpp_partition(); arbitrary but fixed so a
# stray message from unrelated code can never be mistaken for a piece
# this call is expecting.


def mpp_attach_save_chunks(
    mpi_context: MPIContext, value: xr.Dataset | xr.DataArray
) -> xr.Dataset | xr.DataArray:
    """Attach write-time chunk metadata to a distributed object.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed rank-local object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        ``value`` with ``mpi_meta["save_chunks"]`` attached.

    Raises
    ------
    ValueError
        If required partition chunk metadata are missing.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        return value

    save_chunks: dict[str, tuple[int, ...]] | None = None
    error: BaseException | None = None
    if mpi_context.is_root():
        try:
            save_chunks = compute_save_chunks(value, meta, mpi_context.comm.size)
        except BaseException as exc:
            error = exc
    mpi_context.raise_if_error(error, "attach_save_chunks planning")

    save_chunks = mpi_context.broadcast(save_chunks, root=0)
    set_save_chunks(value, cast("dict[str, tuple[int, ...]]", save_chunks))
    return value


def open_distributed_dataset(
    filename: Path | str | PathLike,
    mpi_context: MPIContext | MPI.Intracomm,
    *,
    partition_dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    chunks: Any = None,
    log_partitions: bool = True,
    **kwargs: Any,
) -> MPIXarray:
    """Open a Dataset lazily and partition it across MPI ranks.

    Parameters
    ----------
    filename : str, path-like, file-like, or list of these
        Input accepted by ``xarray.open_dataset``/``xarray.open_mfdataset``.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime whose communicator the result is bound to.
    partition_dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Dimension(s) to partition.
    chunks : int, dict, "auto" or None, optional
        Passed unchanged to xarray.
    log_partitions : bool, optional
        Print one aligned table showing which global interval each rank received.
    **kwargs : Any
        Additional arguments passed unchanged to ``xarray.open_dataset``/ ``xarray.open_mfdataset`` (e.g.

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.

    """

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    xr.set_options(keep_attrs=True)

    use_mfdataset = (isinstance(filename, str) and "*" in filename) or isinstance(
        filename, (list, tuple)
    )
    open_fn: Callable[..., xr.Dataset] = (
        xr.open_mfdataset if use_mfdataset else xr.open_dataset
    )

    requested_dims = _as_partition_dims(partition_dim)
    automatic = requested_dims == "auto"
    # "auto" leaves the axis for rank 0 to choose from the file header.
    dims: tuple[Hashable, ...] = (
        requested_dims
        if isinstance(requested_dims, tuple)
        else ()
        if automatic
        else (requested_dims,)
    )
    data = _open_partitioned(
        mpi_context,
        filename,
        dims,
        open_fn,
        chunks,
        log_partitions,
        automatic,
        kwargs,
    )

    from .core import MPIXarray

    return MPIXarray(data, mpi_context)


def create_distributed_dataarray(
    mpi_context: MPIContext | MPI.Intracomm,
    fill: Callable[..., Any],
    dims: Sequence[Hashable],
    *,
    shape: Sequence[int] | Mapping[Hashable, int] | None = None,
    dim: Hashable | int | Sequence[Hashable] = 0,
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    name: Hashable | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = False,
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> MPIXarray:
    """Create an :class:`MPIXarray` DataArray from a fill function.

    Parameters
    ----------
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        MPI context or communicator.
    fill : callable
        Function producing rank-local values.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes.
    dim : Hashable, int, or sequence of Hashable
        Partition dimension or dimensions.
    dtype : Any, optional
        Fill-function output dtype.
    coords : mapping, optional
        DataArray coordinates.
    name : Hashable, optional
        DataArray name.
    attrs : mapping, optional
        DataArray attributes.
    log_partitions : bool, optional
        Log the rank layout.
    min_partition_size : int or mapping, optional
        Minimum non-empty local extent per partition dimension.

    Returns
    -------
    MPIXarray
        Distributed DataArray wrapper.
    """
    from .core import MPIXarray

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_create_dataarray(
        mpi_context,
        fill,
        dims,
        shape=shape,
        dim=dim,
        dtype=dtype,
        coords=coords,
        name=name,
        attrs=attrs,
        log_partitions=log_partitions,
        min_partition_size=min_partition_size,
    )
    return MPIXarray(data, mpi_context)


def create_distributed_dataset(
    mpi_context: MPIContext | MPI.Intracomm,
    data_vars: Mapping[
        Hashable, xr.DataArray | tuple[Sequence[Hashable], Callable[..., Any]]
    ],
    sizes: Mapping[Hashable, int] | None = None,
    *,
    dim: Hashable | Sequence[Hashable],
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = True,
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> MPIXarray:
    """Create an :class:`MPIXarray` Dataset from rank-local variables.

    Parameters
    ----------
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        MPI context or communicator.
    data_vars : mapping
        DataArrays or ``(dims, fill)`` variable specifications.
    sizes : mapping, optional
        Global dimension sizes.
    dim : Hashable or sequence of Hashable
        Partition dimension or dimensions.
    dtype : Any or mapping, optional
        Default or per-variable fill dtype.
    coords, attrs : mapping, optional
        Dataset coordinates and attributes.
    log_partitions : bool, optional
        Log the rank layout.
    min_partition_size : int or mapping, optional
        Minimum non-empty local extent per partition dimension.

    Returns
    -------
    MPIXarray
        Distributed Dataset wrapper.
    """
    from .core import MPIXarray

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_create_dataset(
        mpi_context,
        data_vars,
        sizes,
        dim=dim,
        dtype=dtype,
        coords=coords,
        attrs=attrs,
        log_partitions=log_partitions,
        min_partition_size=min_partition_size,
    )
    return MPIXarray(data, mpi_context)


def distribute_data(
    value: MPIXarray | xr.Dataset | xr.DataArray | None,
    mpi_context: MPIContext | MPI.Intracomm,
    dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    *,
    root: int = 0,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> MPIXarray:
    """Partition a root-owned xarray object across MPI ranks.

    Parameters
    ----------
    value : MPIXarray, xarray.Dataset, xarray.DataArray, or None
        Complete object on ``root``; non-root ranks must pass None.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime or communicator the result is bound to.
    dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Partition dimension(s).
    root : int, optional
        Rank that owns ``value``.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Rank-local slice with ``.meta`` set.

    """
    from .core import MPIXarray, unwrap

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_partition(
        mpi_context,
        unwrap(value),
        dim,
        root=root,
        chunk_info=chunk_info,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, mpi_context)


def empty_distributed_dataset() -> xr.Dataset:
    """Return a placeholder Dataset for a non-root MPI rank.

    Returns
    -------
    xarray.Dataset
        Dataset marked as containing no rank-local data.

    """
    return xr.Dataset(attrs={_NO_DATA_ATTR: True})


def is_distributed_empty(data: xr.Dataset | xr.DataArray) -> bool:
    """Return whether an object is a non-root MPI placeholder.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to inspect.

    Returns
    -------
    bool
        True when ``data`` is an MPI placeholder Dataset.

    """
    return isinstance(data, xr.Dataset) and data.attrs.get(_NO_DATA_ATTR) is True


def to_netcdf(
    data: xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    mpi_context: MPIContext | MPI.Intracomm | None = None,
    unlimited_dim: str | Iterable[str] | None = None,
    partition_dim: str | None = None,
    *,
    parallel: bool = False,
    batch_size: int = 24,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
    chunks: Mapping[str, Iterable[int]] | None = None,
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> None:
    """Write an xarray object to NetCDF.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to write.
    file : str or os.PathLike
        Output path.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm, optional
        MPI context or communicator.
    unlimited_dim : str or iterable of str, optional
        Unlimited dimension names.
    partition_dim : str, optional
        MPI partition dimension.
    parallel : bool, default False
        Use MPI-parallel NetCDF-4 output.
    batch_size : int, default 24
        Slices written per serial append.
    format : str, default "NETCDF4"
        NetCDF format for serial output.
    shuffle, zlib : bool, default True
        HDF5 filters.
    complevel : int, default 4
        Compression level.
    show_progress : bool, default True
        Display serial write progress.
    stdout : Any, optional
        Progress output stream.
    chunks : mapping, optional
        Explicit NetCDF chunk shapes.
    hints : str, optional
        Semicolon-separated MPI-IO ``key=value`` hints.
    nofill : bool, default True
        Disable NetCDF pre-filling in parallel mode.
    allow_serial : bool, default False
        Permit the parallel writer with one MPI rank.
    """

    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError("data must be an xarray.Dataset or xarray.DataArray")

    target_path = Path(file)

    if parallel:
        if not mpi_context:
            from ..mpi.context import get_mpi_ctx

            mpi_context = get_mpi_ctx()
        if not isinstance(mpi_context, MPIContext):
            mpi_context = MPIContext(mpi_context)

        mpi_meta = mpp_get_meta(data)
        distributed = mpi_meta is not None

        # Ranks must agree on the write path. If one rank saw valid mpi_meta
        # and another did not, the two paths post different collectives and
        # the writer would block instead of reporting the inconsistency.
        agreed = gather_v(distributed, mpi_context.comm)
        if any(agreed) and not all(agreed):
            disagreeing = [
                rank for rank, state in enumerate(agreed) if state != agreed[0]
            ]
            raise mpi_context.MPIError(
                f"MPI ranks disagree on distribution state: {disagreeing}."
            )

        if distributed:
            distributed_dim = str(mpi_meta["dim"])
            if partition_dim is not None and partition_dim != distributed_dim:
                raise ValueError(
                    f"partition_dim={partition_dim!r} differs from distributed "
                    + f"dim={distributed_dim!r}."
                )
            partition_dim = distributed_dim
        elif mpi_context.comm.rank != 0:
            data = empty_distributed_dataset()

        mpp_to_netcdf_parallel(
            mpi_context,
            data,
            target_path,
            partition_dim=partition_dim,
            deflate=complevel if zlib else None,
            shuffle=shuffle,
            chunks=chunks,
            unlimited_dim=unlimited_dim if unlimited_dim is not None else (),
            hints=hints,
            nofill=nofill,
            allow_serial=allow_serial,
        )
        return

    to_netcdf_serial(
        data=data,
        file=target_path,
        unlimited_dim=unlimited_dim,
        batch_size=batch_size,
        format=format,
        shuffle=shuffle,
        zlib=zlib,
        complevel=complevel,
        show_progress=show_progress,
        stdout=stdout,
    )
