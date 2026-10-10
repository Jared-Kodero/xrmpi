"""Manage MPI distribution metadata for xarray objects."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..mpi.context import MPIContext

MPI_META = "mpi_meta"
#: Lightweight partition flag stripped before NetCDF export.
PARTITIONED_ATTR = "mpi_partitioned"
#: Centralize internal attrs that must be removed before export or user-level
#: comparisons.
_INTERNAL_ATTRS = frozenset({MPI_META, PARTITIONED_ATTR})
# Partition identity is defined by dimensions, global sizes, and local ownership bounds.
# Plural keys support multi-D partitions; singular keys mirror the first axis for
# compatibility.


def _canonicalize_meta(meta: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return ``meta`` with both plural and singular partition keys present."""
    if "dims" in meta:
        required = {"dims", "global_sizes", "starts", "stops", "chunk_info"}
        if not required <= meta.keys():
            return None
        dims = tuple(meta["dims"])
        global_sizes, starts, stops = (
            meta["global_sizes"],
            meta["starts"],
            meta["stops"],
        )
        if not dims or not all(
            d in global_sizes and d in starts and d in stops for d in dims
        ):
            return None
        out = dict(meta)
        out["dims"] = dims
        out.setdefault("dim", dims[0])
        out.setdefault("global_size", global_sizes[dims[0]])
        out.setdefault("start", starts[dims[0]])
        out.setdefault("stop", stops[dims[0]])
        return out

    required = {"dim", "global_size", "start", "stop", "chunk_info"}
    if not required <= meta.keys():
        return None
    dim = meta["dim"]
    out = dict(meta)
    out["dims"] = (dim,)
    out["global_sizes"] = {dim: meta["global_size"]}
    out["starts"] = {dim: meta["start"]}
    out["stops"] = {dim: meta["stop"]}
    return out


def _partitions_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two partition metadata mappings own the same slice."""
    left_c = _canonicalize_meta(left)
    right_c = _canonicalize_meta(right)
    if left_c is None or right_c is None:
        return False
    if set(left_c["dims"]) != set(right_c["dims"]):
        return False
    return all(
        left_c["starts"].get(dim) == right_c["starts"].get(dim)
        and left_c["stops"].get(dim) == right_c["stops"].get(dim)
        and left_c["global_sizes"].get(dim) == right_c["global_sizes"].get(dim)
        for dim in left_c["dims"]
    )


def _validate_mpi_meta(
    value: xr.Dataset | xr.DataArray, meta: Any
) -> dict[str, Any] | None:
    """Return ``meta`` when it describes a valid partition of ``value``."""
    if not isinstance(meta, dict):
        return None

    canonical = _canonicalize_meta(meta)
    if canonical is None:
        return None

    dims = canonical["dims"]
    present = [dim for dim in dims if dim in value.dims]
    if not present:
        return None

    for dim in present:
        start = int(canonical["starts"][dim])
        stop = int(canonical["stops"][dim])
        global_size = int(canonical["global_sizes"][dim])
        if start < 0 or stop < start or stop > global_size:
            return None
        if int(value.sizes[dim]) != stop - start:
            return None

    if not isinstance(canonical["chunk_info"], dict):
        return None

    return cast("dict[str, Any]", canonical)


def mpp_get_meta(value: xr.Dataset | xr.DataArray) -> dict[str, Any] | None:
    """Return validated MPI distribution metadata."""
    meta = _validate_mpi_meta(value, value.attrs.get(MPI_META))
    if meta is not None:
        return meta

    if not isinstance(value, xr.Dataset):
        return None

    candidates: list[dict[str, Any]] = []
    for variable in value.variables.values():
        candidate = variable.attrs.get(MPI_META)
        if isinstance(candidate, dict):
            candidates.append(candidate)

    if not candidates:
        return None

    reference = candidates[0]
    for candidate in candidates[1:]:
        if not _partitions_match(candidate, reference):
            return None

    return _validate_mpi_meta(value, reference)


def mpp_set_meta(value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]) -> None:
    """Attach an already-built ``meta`` dict to ``value`` and its variables."""
    dims = meta["dims"]
    value.attrs[MPI_META] = dict(meta)
    if isinstance(value, xr.Dataset):
        for variable in value.variables.values():
            variable.attrs.pop(MPI_META, None)
            if any(dim in variable.dims for dim in dims):
                variable.attrs[MPI_META] = dict(meta)


def _as_dims(dim: Hashable | Iterable[Hashable]) -> tuple[str, ...]:
    """Normalize a ``dim`` argument (one dim, or a sequence of dims) to a tuple."""
    dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
    if not dims:
        raise ValueError("At least one partition dimension is required.")
    if len(set(dims)) != len(dims):
        raise ValueError(f"Partition dimensions must be unique; got {dims!r}.")
    return tuple(str(d) for d in dims)


def _as_dim_map(
    dims: tuple[str, ...], value: int | Mapping[Hashable, int], name: str
) -> dict[str, int]:
    """Normalize a per-dimension argument to a ``{dim: value}`` mapping."""
    if isinstance(value, Mapping):
        resolved = {str(k): int(v) for k, v in value.items()}
        missing = [dim for dim in dims if dim not in resolved]
        if missing:
            raise ValueError(f"{name} is missing an entry for {missing!r}.")
        return resolved
    if len(dims) != 1:
        raise ValueError(
            f"{name} must map partition dimensions to values for multi-D partitioning."
        )
    return {dims[0]: int(cast("int", value))}


def mpp_update_meta(
    value: xr.Dataset | xr.DataArray,
    *,
    dim: Hashable | Sequence[Hashable],
    global_size: int | Mapping[Hashable, int],
    start: int | Mapping[Hashable, int],
    stop: int | Mapping[Hashable, int],
    chunk_info: Mapping[Hashable, int],
    cart: Mapping[str, Any] | None = None,
) -> None:
    """Attach MPI distribution metadata for one or more partition dimensions.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Rank-local xarray object.
    dim : hashable or sequence of hashable
        Distributed dimension(s).
    global_size : int or mapping
        Global length of ``dim``.
    start, stop : int or mapping
        Global half-open interval owned by this rank, per dimension.
    chunk_info : mapping
        Effective climtools chunk size for every retained dimension.
    cart : mapping, optional
        Cartesian topology descriptor (``grid_shape``, ``coords``, ``periods``),
        attached only for a multi-dimensional partition.

    """
    dims = _as_dims(dim)
    global_sizes = _as_dim_map(dims, global_size, "global_size")
    starts = _as_dim_map(dims, start, "start")
    stops = _as_dim_map(dims, stop, "stop")

    meta: dict[str, Any] = {
        "dims": dims,
        "global_sizes": {d: global_sizes[d] for d in dims},
        "starts": {d: starts[d] for d in dims},
        "stops": {d: stops[d] for d in dims},
        # Backward-compatible singular aliases; see the module-level note
        # above _canonicalize_meta.
        "dim": dims[0],
        "global_size": global_sizes[dims[0]],
        "start": starts[dims[0]],
        "stop": stops[dims[0]],
        "chunk_info": {
            str(name): int(size)
            for name, size in chunk_info.items()
            if name in value.dims and int(size) > 0
        },
    }
    if cart is not None:
        meta["cart"] = dict(cart)
    mpp_set_meta(value, meta)


def set_save_chunks(
    value: xr.Dataset | xr.DataArray, save_chunks: Mapping[str, tuple[int, ...]]
) -> None:
    """Attach save_chunks to ``value``'s existing MPI distribution metadata.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Rank-local xarray object that already carries valid MPI distribution metadata
        (see :func:`mpp_get_meta`).
    save_chunks : mapping
        Mapping from variable name to save_chunk shape.

    Raises
    ------
    ValueError
        If ``value`` carries no valid MPI distribution metadata to attach
        ``save_chunks`` to.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        raise ValueError("value carries no MPI distribution metadata")
    updated = dict(meta)
    updated["save_chunks"] = {
        str(name): tuple(int(length) for length in shape)
        for name, shape in save_chunks.items()
    }
    mpp_set_meta(value, updated)


def reattach_meta_after_collapse(
    result: xr.Dataset | xr.DataArray, meta: Mapping[str, Any], dim: str
) -> xr.Dataset | xr.DataArray:
    """Carry forward metadata for the dims that survive ``dim`` collapsing away.

    Shared by ops that resolve one global label along ``dim`` and
    replicate the answer to every rank (`mpp_sel_scalar`, `mpp_isel_scalar`,
    `isel`'s singleton-repartition case). ``dim`` itself gets no
    replacement value; other active partition dimensions carry over
    unchanged.

    Returns
    -------
    xr.Dataset | xr.DataArray
        ``result`` with metadata reattached for any surviving partition
        dimension, or unchanged if none survive.

    """
    from .chunks import prune_chunk_info

    remaining_dims = tuple(
        d for d in meta["dims"] if d != dim and d in getattr(result, "dims", ())
    )
    if not remaining_dims:
        return result
    mpp_update_meta(
        result,
        dim=remaining_dims,
        global_size={d: int(meta["global_sizes"][d]) for d in remaining_dims},
        start={d: int(meta["starts"][d]) for d in remaining_dims},
        stop={d: int(meta["stops"][d]) for d in remaining_dims},
        chunk_info=prune_chunk_info(meta["chunk_info"], result),
        # Dropping a Cartesian axis invalidates the cached topology; rebuild it lazily
        # when needed.
        cart=None,
    )
    return result


def strip_mpi_meta(value: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Return a shallow copy without MPI distribution metadata."""
    output = value.copy(deep=False)
    for key in _INTERNAL_ATTRS:
        output.attrs.pop(key, None)
    if isinstance(output, xr.Dataset):
        for variable in output.variables.values():
            for key in _INTERNAL_ATTRS:
                variable.attrs.pop(key, None)
    return output


def strip_export_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``attrs`` without internal MPI bookkeeping keys."""
    return {key: value for key, value in attrs.items() if key not in _INTERNAL_ATTRS}


def mpp_should_log_partitions(mpi_context: MPIContext, log_partitions: bool) -> bool:
    """Resolve partition logging collectively.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    log_partitions : bool
        Local logging request.

    Returns
    -------
    bool
        True on every rank if any rank requested logging.
    """
    return bool(mpi_context.comm.allreduce(bool(log_partitions), op=MPI.LOR))


def _aligned_rows(header: list[str], rows: list[list[str]]) -> list[str]:
    """Right-align string cells into fixed-width columns."""
    widths = [max(len(h), *(len(r[n]) for r in rows)) for n, h in enumerate(header)]
    return [
        "   " + "  ".join(f"{c:>{w}}" for c, w in zip(cells, widths, strict=True))
        for cells in (header, *rows)
    ]


def mpp_set_domain_bounds(
    result: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    dim: Hashable,
    *,
    global_size: int,
    start: int,
    stop: int,
    chunk_info: Mapping[str, int],
) -> None:
    """Rewrite one axis's compute domain, leaving the other axes intact.

    Parameters
    ----------
    result : xarray.Dataset or xarray.DataArray
        Object whose metadata is updated in place.
    meta : mapping
        Metadata the object carried before the operation.
    dim : Hashable
        Axis whose bounds changed.
    global_size, start, stop : int
        New global length and this rank's half-open bounds along ``dim``.
    chunk_info : mapping
        Chunk sizes for the new object.
    """
    bounds = {key: dict(meta[key]) for key in ("global_sizes", "starts", "stops")}
    bounds["global_sizes"][dim] = global_size
    bounds["starts"][dim] = start
    bounds["stops"][dim] = stop
    mpp_update_meta(
        result,
        dim=meta["dims"],
        global_size=bounds["global_sizes"],
        start=bounds["starts"],
        stop=bounds["stops"],
        chunk_info=chunk_info,
        cart=meta.get("cart"),
    )


def mpp_redefine_domain(
    mpi_context: MPIContext,
    result: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    dim: Hashable,
) -> xr.Dataset | xr.DataArray:
    """Re-derive one axis's compute domain from this rank's new local extent.

    An operation that changes how many elements a rank holds along ``dim``
    invalidates every rank's offsets. As in FMS, the domain is redefined
    rather than patched: the new global length and each rank's bounds follow
    from an exclusive scan of the local extents.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    result : xarray.Dataset or xarray.DataArray
        Object produced by the operation, updated in place.
    meta : mapping
        Metadata the input carried.
    dim : Hashable
        Axis whose local extent changed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        ``result``, carrying the redefined domain.
    """
    from ..mpp.ext_collectives import partition_offsets
    from ..mpp.ext_domains import dim_comm
    from .chunks import prune_chunk_info

    comm = dim_comm(meta, dim, mpi_context)
    global_size, start, stop = partition_offsets(comm, int(result.sizes[dim]))
    mpp_set_domain_bounds(
        result,
        meta,
        dim,
        global_size=global_size,
        start=start,
        stop=stop,
        chunk_info=prune_chunk_info(meta["chunk_info"], result),
    )
    return result


def mpp_log_partition_report(
    mpi_context: MPIContext,
    data: xr.Dataset | xr.DataArray,
    dim: Hashable | tuple[Hashable, ...],
    *,
    origin: str,
    global_size: int | Mapping[Hashable, int],
    start: int | Mapping[Hashable, int],
    stop: int | Mapping[Hashable, int],
    grid_shape: tuple[int, ...] | None = None,
    coords: tuple[int, ...] | None = None,
    automatic: bool = False,
    detail: bool = True,
) -> None:
    """Print how a global array is divided across ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    data : xarray.Dataset or xarray.DataArray
        Partitioned object, read for its shape and chunking.
    dim : Hashable or tuple of Hashable
        Partitioned dimension, or dimensions for a Cartesian layout.
    origin : str
        Label naming what produced this partition.
    global_size, start, stop : int or mapping
        Global length and this rank's half-open bounds, per dimension.
    grid_shape : tuple of int, optional
        Process-grid shape for a Cartesian layout.
    coords : tuple of int, optional
        This rank's position in the process grid.
    automatic : bool, default False
        Whether the partition dimension was chosen rather than requested.
    detail : bool, default True
        Whether to include the per-rank table.
    """
    comm = mpi_context.comm
    dims = tuple(dim) if isinstance(dim, tuple | list) else (dim,)
    cartesian = grid_shape is not None or len(dims) > 1

    def per_dim(value: Any) -> Mapping[Hashable, int]:
        """Accept either a scalar for one dimension or a full mapping."""
        return value if isinstance(value, Mapping) else {dims[0]: value}

    starts, stops, sizes = per_dim(start), per_dim(stop), per_dim(global_size)
    local = (
        int(comm.rank),
        tuple(int(c) for c in (coords or ())),
        tuple(int(starts[d]) for d in dims),
        tuple(int(stops[d]) for d in dims),
    )

    rows = gather_v(local, comm, root=0)
    if comm.rank != 0 or rows is None:
        return

    border, separator = "=" * 80, "-" * 80
    dims_str = ", ".join(f"{str(d)!r}{' (auto)' if automatic else ''}" for d in dims)
    title = "MPI CARTESIAN PARTITION REPORT" if cartesian else "MPI PARTITION REPORT"
    lines = [border, f" {title}: {origin}", border]

    if cartesian:
        grid = "x".join(str(n) for n in (grid_shape or (comm.size,)))
        lines += [
            f" 🔹 Dimensions   : {dims_str}",
            f" 🔹 Process grid : {grid} ({comm.size} ranks)",
            " 🔹 Global sizes : " + ", ".join(f"{d!s}={int(sizes[d])}" for d in dims),
        ]
    else:
        counts = [row[3][0] - row[2][0] for row in rows]
        idle = sum(1 for count in counts if count == 0)
        split = (
            f"{min(counts)}/rank"
            if min(counts) == max(counts)
            else f"{min(counts)}-{max(counts)}/rank"
        )
        if idle:
            split += f" (IDLE={idle})"
        shape = " ".join(
            f"{name!s}={int(length)}"
            for name, length in data.sizes.items()
            if name != dims[0]
        )
        # Dataset.chunks maps dimension to chunk sizes; DataArray.chunks is a
        # bare tuple in dimension order.
        chunking = data.chunks or {}
        if not isinstance(chunking, Mapping):
            chunking = dict(zip(data.dims, chunking, strict=True))
        chunks = "  ".join(
            f"{name!s}={max(int(size) for size in sizes_)}"
            for name, sizes_ in chunking.items()
        )
        lines += [
            f" 🔹 Dimension    : {dims_str}",
            f" 🔹 Global Size  : {sizes[dims[0]]} (Ranks: {comm.size})",
            f" 🔹 Split        : {split}",
            f" 🔹 Shape        : {shape or 'scalar'}",
            f" 🔹 Chunks/Rank  : {chunks or 'unchunked'}",
        ]

    if detail:
        show_coords = any(row[1] for row in rows)
        header = ["rank"] + (["coords"] if show_coords else [])
        for d in dims:
            header += [f"{d} slice", f"{d} n"] if cartesian else ["slice", "n"]
        table = []
        for rank_id, rank_coords, rank_starts, rank_stops in rows:
            cells = [str(rank_id)] + ([str(rank_coords)] if show_coords else [])
            for n in range(len(dims)):
                cells += [
                    f"{rank_starts[n]}:{rank_stops[n]}",
                    str(rank_stops[n] - rank_starts[n]),
                ]
            table.append(cells)
        lines += [separator, *_aligned_rows(header, table)]

    lines.append(border)
    mpi_context.log("")
    mpi_context.log("\n".join(lines), flush=True, prefix=False)
    mpi_context.log("", prefix=False)


def indexer_is_scalar(indexer: Any) -> bool:
    """Return whether an isel/sel indexer selects a single position.

    Returns
    -------
    bool
        True when ``indexer`` selects exactly one position and therefore drops its
        dimension, rather than keeping it with length one.

    """
    return not isinstance(indexer, (slice, list, tuple, np.ndarray, xr.DataArray))


def _coord_length(spec: Any) -> int | None:
    """Return a coordinate spec's own length, or None if it has none."""
    array = spec[1] if isinstance(spec, tuple) else spec
    array = np.asarray(array)
    return int(array.shape[0]) if array.ndim > 0 else None


def resolve_sizes(
    required_dims: Iterable[Hashable],
    sizes: Mapping[Hashable, int] | None,
    coords: Mapping[Hashable, Any] | None,
) -> dict[Hashable, int]:
    """Fill in any dimension length missing from ``sizes`` using ``coords``."""
    resolved = dict(sizes) if sizes else {}
    coords = coords or {}
    missing = []
    for dim_name in required_dims:
        if dim_name in resolved:
            continue
        length = _coord_length(coords[dim_name]) if dim_name in coords else None
        if length is None:
            missing.append(dim_name)
        else:
            resolved[dim_name] = length
    if missing:
        raise ValueError(
            f"Missing sizes for dimensions: {sorted(str(d) for d in missing)}."
        )
    return resolved


def localize_coord(spec: Any, global_size: int, start: int, stop: int) -> Any:
    """Slice a coordinate spec to ``[start:stop)`` if it is full-length."""
    if isinstance(spec, tuple):
        coord_dims, coord_array, *rest = spec
    else:
        coord_dims, coord_array, rest = None, spec, []
    coord_array = np.asarray(coord_array)
    if coord_array.shape and coord_array.shape[0] == global_size:
        coord_array = coord_array[start:stop]
    if coord_dims is None:
        return coord_array
    return (coord_dims, coord_array, *rest)


def _fill_local(
    fn: Callable[..., Any], shape: tuple[int, ...], dtype: Any, *args: Any
) -> np.ndarray[Any, Any]:
    """Evaluate ``fn(*args)`` and enforce the declared local shape and dtype.

    Dask takes the declared dtype on trust, so a fill returning another dtype
    would leave array metadata and computed chunks disagreeing.
    """
    value = np.asarray(fn(*args), dtype=dtype)
    if value.shape != shape:
        raise ValueError(
            f"fill returned shape {value.shape}; expected the local shape {shape}."
        )
    return value


def delayed_local(
    fn: Callable[..., Any], args: tuple[Any, ...], shape: tuple[int, ...], dtype: Any
) -> Any:
    """Wrap ``fn(*args)`` as one rank's own slice, not yet computed.

    The result is cast to ``dtype`` and must have exactly ``shape``.
    """
    import dask
    import dask.array as dask_array

    return dask_array.from_delayed(
        dask.delayed(_fill_local)(fn, shape, dtype, *args), shape=shape, dtype=dtype
    )


_SHORT_PARTITION_WARNED: set[tuple[str, int, int]] = set()
"""Distinct (dim, length, mpi_size) triples already warned about this process.

Populated by :func:`choose_partition_dim`. Not meant to be read or mutated
directly; exists at module scope only so the warning survives across many
independent calls within one process without needing to thread state through
every caller.
"""


def choose_partition_dim(
    sizes: Mapping[Hashable, int],
    mpi_size: int,
    *,
    exclude: Iterable[Hashable] = (),
    rank: int | None = None,
) -> Hashable:
    """Select a partition dimension automatically.

    Parameters
    ----------
    sizes : mapping
        Dimension name to global length.
    mpi_size : int
        Number of ranks the data will be spread over.
    exclude : iterable of hashable, optional
        Dimensions that must not be chosen, for example a dimension the caller intends
        to reduce over.
    rank : int, optional
        Calling rank, used only to gate the short-partition warning below to rank 0.

    Returns
    -------
    hashable
        Chosen dimension.

    Raises
    ------
    ValueError
        If no dimension is available.

    """
    blocked = set(exclude)
    candidates = [
        (dim, int(length)) for dim, length in sizes.items() if dim not in blocked
    ]
    if not candidates:
        raise ValueError("No dimension is available for automatic partitioning.")

    usable = [item for item in candidates if item[1] > 1] or candidates
    order = {dim: position for position, (dim, _) in enumerate(usable)}
    dim, length = max(usable, key=lambda item: (item[1], -order[item[0]]))

    return dim


def mpp_operand_meta(operand: Any) -> dict[str, Any] | None:
    """Return ``operand``'s MPI distribution metadata, if any."""
    if isinstance(operand, (xr.Dataset, xr.DataArray)):
        return mpp_get_meta(operand)
    return None


def reattach_meta(result: Any, meta: Mapping[str, Any]) -> Any:
    """Put an unchanged distribution back on an operation's result.

    For an operation that leaves the partition alone, the result carries the
    same bounds as the input. Chunk info is pruned to the dimensions the
    result actually has, since an operation may drop one.

    Parameters
    ----------
    result : Any
        Operation result. Non-xarray values are returned untouched.
    meta : mapping
        Distribution metadata the input carried.

    Returns
    -------
    Any
        ``result``, tagged when it is a Dataset or DataArray.
    """
    from .chunks import prune_chunk_info

    if isinstance(result, xr.Dataset | xr.DataArray):
        mpp_update_meta(
            result,
            dim=meta["dims"],
            global_size=meta["global_sizes"],
            start=meta["starts"],
            stop=meta["stops"],
            chunk_info=prune_chunk_info(meta["chunk_info"], result),
            cart=meta.get("cart"),
        )
    return result


def mpp_concat_along(
    pieces: list[Any], template: xr.Dataset | xr.DataArray, dim: Hashable
) -> xr.Dataset | xr.DataArray:
    """Join gathered slices back into one object along ``dim``.

    A Dataset needs ``data_vars="minimal"`` so variables that do not span
    ``dim`` are not needlessly broadcast along it; a DataArray has no such
    distinction.
    """
    if isinstance(template, xr.Dataset):
        return xr.concat(pieces, dim=dim, data_vars="minimal")
    return xr.concat(pieces, dim=dim)


def mpp_partition_meta(
    value: xr.Dataset | xr.DataArray, dim: Hashable
) -> dict[str, Any] | None:
    """Return the distribution metadata only if ``dim`` is partitioned.

    Almost every distributed operation opens by asking the same question: is
    the dimension I am about to touch actually split across ranks? If it is
    not, the operation is rank-local and plain xarray handles it. Returning
    the metadata on the distributed path and None otherwise puts that test in
    one place.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to inspect.
    dim : Hashable
        Dimension the caller is about to operate on.

    Returns
    -------
    dict or None
        The metadata when ``dim`` is partitioned, otherwise None.
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return None
    return meta
