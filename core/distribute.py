"""Build distributed objects and place existing data onto a domain.

Defining how an array is divided is :mod:`~xgeo.mpp.mpp_domains`'s job;
this module applies that division to xarray objects, either by creating one
whose blocks are filled per rank or by cutting an existing object on the root
rank and sending each piece to its owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.mpp_domains_define import mpp_define_domains
from .chunks import (
    get_chunk_bounds,
    get_effective_chunk_size,
    prune_chunk_info,
)
from .meta import (
    choose_partition_dim,
    delayed_local,
    localize_coord,
    mpp_get_meta,
    mpp_log_partition_report,
    mpp_should_log_partitions,
    mpp_update_meta,
    resolve_sizes,
    strip_mpi_meta,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    from ..mpi.context import MPIContext


_DISTRIBUTE_TAG = 0x6469_7374  # b"dist" as an int, easy to spot in a trace


def _resolve_single_dim(
    requested: Any, stripped: xr.Dataset | xr.DataArray, comm: MPI.Comm
) -> tuple[Hashable, ...]:
    """Resolve a one-axis partition request, honouring ``"auto"``."""
    dim = requested[0] if isinstance(requested, tuple) else requested
    if dim == "auto":
        dim = choose_partition_dim(stripped.sizes, comm.size, rank=comm.rank)
    if dim not in stripped.dims:
        raise ValueError(f"Unknown partition dimension {dim!r}.")
    return (dim,)


def mpp_partition(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray | None,
    dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    *,
    root: int = 0,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """partition a root-owned xarray object across MPI ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset, xarray.DataArray, or None
        Complete object on ``root``; non-root ranks must pass None.
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
    xarray.Dataset or xarray.DataArray
        Rank-local slice carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ownership, metadata, or ``dim`` is invalid.

    """
    comm = mpi_context.comm
    is_root = mpi_context.is_root(root)
    requested_dims = _as_partition_dims(dim)
    multi_dim = isinstance(requested_dims, tuple) and len(requested_dims) > 1

    # Prepare every slice before communication so a root-side failure is
    # synchronized before any rank can block in send/receive.
    error: BaseException | None = None
    pieces: list[Any] | None = None
    replicated_value: xr.Dataset | xr.DataArray | None = None
    try:
        if is_root:
            if value is None:
                raise ValueError(f"Rank {root} (root) must provide a value, not None.")
            if mpp_get_meta(value) is not None:
                raise ValueError("Object is already distributed.")
            stripped = strip_mpi_meta(value)

            if not stripped.dims:
                # Nothing to partition: send the (necessarily small)
                # whole object to every rank as replicated data,
                # mirroring repartition's handling of the same case.
                replicated_value = stripped
            else:
                dims = (
                    cast("tuple[Hashable, ...]", requested_dims)
                    if multi_dim
                    else _resolve_single_dim(requested_dims, stripped, comm)
                )
                pieces = _partition_pieces(
                    mpi_context, stripped, dims, comm.size, chunk_info
                )
        elif value is not None:
            raise ValueError(
                f"Only root rank {root} may provide value; got rank {comm.rank}."
            )
    except BaseException as exc:
        error = exc
    mpi_context.raise_if_error(error, "partition")

    # Broadcast which transfer path root prepared.
    dimensionless = mpi_context.broadcast(
        replicated_value is not None if is_root else None, root=root
    )

    # Transfer the validated pieces.
    if dimensionless:
        # Nothing to partition: same small object broadcast to every
        # rank, no per-rank slicing or point-to-point send needed.
        output = mpi_context.broadcast(replicated_value if is_root else None, root=root)
        return cast("xr.Dataset | xr.DataArray", output)

    if is_root:
        assert pieces is not None
        output = pieces[root]
        # Post all sends before waiting so scatter latency does not serialize with rank
        # count.
        mpi_context.send_all(
            {rank: piece for rank, piece in enumerate(pieces) if rank != root},
            tag=_DISTRIBUTE_TAG,
        )
    else:
        output = mpi_context.receive(source=root, tag=_DISTRIBUTE_TAG)

    if mpp_should_log_partitions(mpi_context, log_partitions):
        meta = mpp_get_meta(output)
        if meta is not None and "cart" in meta:
            mpp_log_partition_report(
                mpi_context,
                output,
                meta["dims"],
                origin="partition",
                global_size=meta["global_sizes"],
                start=meta["starts"],
                stop=meta["stops"],
                grid_shape=meta["cart"]["grid_shape"],
                coords=meta["cart"]["coords"],
            )
        elif meta is not None:
            mpp_log_partition_report(
                mpi_context,
                output,
                meta["dim"],
                origin="partition",
                global_size=meta["global_size"],
                start=meta["start"],
                stop=meta["stop"],
                automatic=(dim == "auto"),
            )
    return output


def _as_partition_dims(
    dim: Hashable | Sequence[Hashable] | Literal["auto"],
) -> Literal["auto"] | tuple[Hashable, ...]:
    """Normalize ``mpp_partition()``'s ``dim`` argument."""
    if dim == "auto":
        return "auto"
    if isinstance(dim, (list, tuple)):
        dims = tuple(dim)
        if not dims:
            raise ValueError("partition_dim sequence must not be empty.")
        return dims
    return (dim,)


def _detach_attrs(piece: xr.Dataset | xr.DataArray) -> None:
    """Break shallow-copy attribute sharing before per-rank metadata is added."""
    piece.attrs = dict(piece.attrs)
    if isinstance(piece, xr.Dataset):
        for variable in piece.variables.values():
            variable.attrs = dict(variable.attrs)


def _partition_pieces(
    mpi_context: MPIContext,
    stripped: xr.Dataset | xr.DataArray,
    dims: tuple[Hashable, ...],
    comm_size: int,
    chunk_info: Mapping[str, int] | None = None,
) -> list[Any]:
    """Slice ``stripped`` into one piece per rank.

    A single axis follows the source's chunk boundaries so each rank's slice
    lines up with whole chunks; a process grid divides every axis evenly, as
    FMS does per axis in ``mpp_define_domains``.
    """
    for d in dims:
        if d not in stripped.dims:
            raise ValueError(f"Unknown partition dimension {d!r}.")
    sizes = {d: int(stripped.sizes[d]) for d in dims}
    single = len(dims) == 1

    info = dict(chunk_info or {})
    if single:
        dim = dims[0]
        info[str(dim)] = get_effective_chunk_size(
            sizes[dim],
            int(
                info.get(
                    str(dim), get_effective_chunk_size(sizes[dim], None, comm_size)
                )
            ),
            comm_size,
        )

    pieces = []
    for rank in range(comm_size):
        cart = None
        if single:
            dim = dims[0]
            bounds = {
                dim: get_chunk_bounds(sizes[dim], info[str(dim)], rank, comm_size)
            }
        else:
            domain = mpp_define_domains(mpi_context, sizes, dims, rank=rank)
            bounds = {d: (domain.starts[d], domain.stops[d]) for d in dims}
            cart = domain.cart

        piece = stripped.isel({d: slice(*bounds[d]) for d in dims})
        _detach_attrs(piece)

        piece_info = prune_chunk_info(info, piece) if single else {}
        for other_dim, other_length in piece.sizes.items():
            piece_info.setdefault(
                str(other_dim),
                get_effective_chunk_size(int(other_length), None, comm_size),
            )
        starts = {d: bounds[d][0] for d in dims}
        stops = {d: bounds[d][1] for d in dims}
        mpp_update_meta(
            piece,
            dim=dims[0] if single else dims,
            global_size=sizes[dims[0]] if single else sizes,
            start=starts[dims[0]] if single else starts,
            stop=stops[dims[0]] if single else stops,
            chunk_info=piece_info,
            cart=cart,
        )
        pieces.append(piece)
    return pieces


def _normalize_create_dim(
    dim: Hashable | int | Sequence[Hashable], dims: Sequence[Hashable]
) -> tuple[Hashable, ...]:
    """Normalize ``create_dataarray``/``create_dataset``'s ``dim`` to a tuple."""
    if isinstance(dim, (list, tuple)):
        if not dim:
            raise ValueError("dim sequence must not be empty.")
        for d in dim:
            if d not in dims:
                raise ValueError(f"dim {d!r} is not in dims {tuple(dims)!r}.")
        if len(set(dim)) != len(dim):
            raise ValueError(f"dim entries must be unique; got {tuple(dim)!r}.")
        return tuple(dim)
    axis_or_name = dims.index(dim) if not isinstance(dim, Integral) else int(dim)
    if not 0 <= axis_or_name < len(dims):
        raise ValueError(f"dim {dim!r} is not in dims {tuple(dims)!r}.")
    return (dims[axis_or_name],)


def _min_chunk_map(
    partition_dims: tuple[Hashable, ...],
    min_partition_size: int | Mapping[Hashable, int] | None,
) -> dict[Hashable, int]:
    """Expand a scalar or mapping minimum extent to one entry per axis."""
    if min_partition_size is None:
        return {}
    if isinstance(min_partition_size, Mapping):
        return dict(min_partition_size)
    return dict.fromkeys(partition_dims, min_partition_size)


def _localize_coords(
    coords: Mapping[Hashable, Any] | None,
    partition_dims: tuple[Hashable, ...],
    bounds: Mapping[Hashable, tuple[int, int]],
    resolved_sizes: Mapping[Hashable, int],
) -> dict[Hashable, Any]:
    """Trim each partitioned coordinate to this rank's compute domain."""
    local = dict(coords) if coords else {}
    for d in partition_dims:
        if d in local:
            start, stop = bounds[d]
            local[d] = localize_coord(local[d], int(resolved_sizes[d]), start, stop)
    return local


def _attach_created_meta(
    mpi_context: MPIContext,
    obj: xr.Dataset | xr.DataArray,
    partition_dims: tuple[Hashable, ...],
    bounds: Mapping[Hashable, tuple[int, int]],
    resolved_sizes: Mapping[Hashable, int],
    cart: dict[str, Any] | None,
    origin: str,
    log_partitions: bool,
) -> None:
    """Record the domain on a freshly created object and report it."""
    single = len(partition_dims) == 1
    first = partition_dims[0]
    global_sizes = {d: int(resolved_sizes[d]) for d in partition_dims}
    starts = {d: bounds[d][0] for d in partition_dims}
    stops = {d: bounds[d][1] for d in partition_dims}

    mpp_update_meta(
        obj,
        dim=first if single else partition_dims,
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info={str(d): stops[d] - starts[d] for d in partition_dims},
        cart=cart,
    )
    if not mpp_should_log_partitions(mpi_context, log_partitions):
        return
    mpp_log_partition_report(
        mpi_context,
        obj,
        first if single else partition_dims,
        origin=origin,
        global_size=global_sizes[first] if single else global_sizes,
        start=starts[first] if single else starts,
        stop=stops[first] if single else stops,
        grid_shape=None if cart is None else cart["grid_shape"],
        coords=None if cart is None else cart["coords"],
    )


def mpp_create_dataarray(
    mpi_context: MPIContext,
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
) -> xr.DataArray:
    """Create a distributed DataArray from a rank-local fill function.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    fill : callable
        Function producing this rank's local values.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes.
    dim : Hashable, int, or sequence of Hashable
        Dimension or dimensions to partition.
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
    xarray.DataArray
        Rank-local DataArray carrying MPI metadata.

    Raises
    ------
    ValueError
        If partition dimensions or global sizes are invalid.
    """
    partition_dims = _normalize_create_dim(dim, dims)
    min_chunk_map = _min_chunk_map(partition_dims, min_partition_size)

    if shape is None or isinstance(shape, Mapping):
        explicit_sizes = dict(shape) if shape else None
    else:
        if len(shape) != len(dims):
            raise ValueError(f"shape has {len(shape)} entries; dims has {len(dims)}.")
        explicit_sizes = dict(zip(dims, shape, strict=True))
    resolved_sizes = resolve_sizes(dims, explicit_sizes, coords)

    extents = tuple(int(resolved_sizes[d]) for d in partition_dims)
    sizes = dict(zip(partition_dims, extents, strict=True))

    domain = mpp_define_domains(
        mpi_context, sizes, partition_dims, min_partition_size=min_chunk_map
    )
    bounds = {d: (domain.starts[d], domain.stops[d]) for d in partition_dims}
    cart = domain.cart

    local_shape = tuple(
        (bounds[name][1] - bounds[name][0])
        if name in bounds
        else int(resolved_sizes[name])
        for name in dims
    )

    fill_args = tuple(v for d in partition_dims for v in bounds[d])
    local_data = delayed_local(fill, fill_args, local_shape, dtype)

    local_coords = _localize_coords(coords, partition_dims, bounds, resolved_sizes)

    da = xr.DataArray(
        local_data, dims=tuple(dims), coords=local_coords, name=name, attrs=attrs
    )
    _attach_created_meta(
        mpi_context,
        da,
        partition_dims,
        bounds,
        resolved_sizes,
        cart,
        "create_dataarray",
        log_partitions,
    )
    return da


def mpp_create_dataset(
    mpi_context: MPIContext,
    data_vars: Mapping[
        Hashable,
        xr.DataArray | tuple[Sequence[Hashable], Callable[[int, int], Any]],
    ],
    sizes: Mapping[Hashable, int] | None = None,
    *,
    dim: Hashable | Sequence[Hashable],
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = True,
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> xr.Dataset:
    """Create a distributed Dataset from rank-local variables.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    data_vars : mapping
        DataArrays or ``(dims, fill)`` variable specifications.
    sizes : mapping, optional
        Global dimension sizes.
    dim : Hashable or sequence of Hashable
        Dimension or dimensions to partition.
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
    xarray.Dataset
        Rank-local Dataset carrying MPI metadata.
    """
    if isinstance(dim, (list, tuple)):
        if not dim:
            raise ValueError("dim sequence must not be empty.")
        if len(set(dim)) != len(dim):
            raise ValueError(f"dim entries must be unique; got {tuple(dim)!r}.")
        partition_dims = tuple(dim)
    else:
        partition_dims = (dim,)
    min_chunk_map = _min_chunk_map(partition_dims, min_partition_size)

    required_dims: set[Hashable] = set(partition_dims)
    for spec in data_vars.values():
        if not isinstance(spec, xr.DataArray):
            var_dims, _ = spec
            required_dims.update(var_dims)
    resolved_sizes = resolve_sizes(required_dims, sizes, coords)

    extents = tuple(int(resolved_sizes[d]) for d in partition_dims)
    sizes = dict(zip(partition_dims, extents, strict=True))

    domain = mpp_define_domains(
        mpi_context, sizes, partition_dims, min_partition_size=min_chunk_map
    )
    bounds = {d: (domain.starts[d], domain.stops[d]) for d in partition_dims}
    cart = domain.cart

    dtype_map = dtype if isinstance(dtype, Mapping) else None

    built_vars: dict[Hashable, Any] = {}
    for var_name, spec in data_vars.items():
        if isinstance(spec, xr.DataArray):
            for d in partition_dims:
                if d in spec.dims:
                    d_start, d_stop = bounds[d]
                    expected_len = d_stop - d_start
                    if int(spec.sizes[d]) != expected_len:
                        raise ValueError(
                            f"data_vars[{var_name!r}] has local {d!r} length "
                            + f"{spec.sizes[d]}; expected {expected_len}."
                        )
            built_vars[var_name] = spec
            continue

        var_dims, var_fill = spec
        var_dtype = (
            dtype_map.get(var_name, np.float64) if dtype_map is not None else dtype
        )
        local_dims_here = [d for d in partition_dims if d in var_dims]
        local_shape = tuple(
            (bounds[name][1] - bounds[name][0])
            if name in local_dims_here
            else int(resolved_sizes[name])
            for name in var_dims
        )
        if local_dims_here:
            fill_args = tuple(v for d in local_dims_here for v in bounds[d])
            local_data = delayed_local(var_fill, fill_args, local_shape, var_dtype)
        elif callable(var_fill):
            # Not partitioned: identical on every rank, so there is no
            # (start, stop) to give -- fill() takes no arguments and
            # closes over whatever sizes it needs itself.
            local_data = delayed_local(var_fill, (), local_shape, var_dtype)
        else:
            local_data = var_fill
        built_vars[var_name] = (tuple(var_dims), local_data)

    local_coords = _localize_coords(coords, partition_dims, bounds, resolved_sizes)

    ds = xr.Dataset(built_vars, coords=local_coords, attrs=attrs)
    _attach_created_meta(
        mpi_context,
        ds,
        partition_dims,
        bounds,
        resolved_sizes,
        cart,
        "create_dataset",
        log_partitions,
    )
    return ds


def mpp_repartition(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable | Literal["auto"] = "auto",
    *,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Partition a replicated xarray object across MPI ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Complete object present on every rank.
    dim : Hashable or {"auto"}, optional
        New partition dimension.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Rank-local slice carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ``value`` is already distributed or ``dim`` is invalid.

    """
    if mpp_get_meta(value) is not None:
        raise ValueError("Object is already distributed.")

    stripped_value = strip_mpi_meta(value)
    automatic = dim == "auto"
    if automatic:
        if not value.dims:
            return stripped_value
        dim = choose_partition_dim(
            value.sizes, mpi_context.comm.size, rank=mpi_context.comm.rank
        )

    if dim not in value.dims:
        raise ValueError(f"Repartition dimension {dim!r} does not exist.")

    info = dict(chunk_info or {})
    length = int(value.sizes[dim])
    chunk_size = int(
        info.get(
            str(dim),
            get_effective_chunk_size(length, None, mpi_context.comm.size),
        )
    )
    chunk_size = get_effective_chunk_size(length, chunk_size, mpi_context.comm.size)
    info[str(dim)] = chunk_size

    start, stop = get_chunk_bounds(
        length, chunk_size, mpi_context.comm.rank, mpi_context.comm.size
    )
    output = stripped_value.isel({dim: slice(start, stop)})
    info = prune_chunk_info(info, output)
    for other_dim, other_length in output.sizes.items():
        info.setdefault(
            str(other_dim),
            get_effective_chunk_size(int(other_length), None, mpi_context.comm.size),
        )

    mpp_update_meta(
        output, dim=dim, global_size=length, start=start, stop=stop, chunk_info=info
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        mpp_log_partition_report(
            mpi_context,
            output,
            dim,
            origin="repartition",
            global_size=length,
            start=start,
            stop=stop,
            automatic=automatic,
        )
    return output
