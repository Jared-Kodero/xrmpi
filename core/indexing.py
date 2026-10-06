"""Provide global-coordinate indexing for distributed xarray objects."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v, partition_offsets, scatter_v
from ..mpp.ext_domains import dim_comm as _dim_comm
from ..mpp.ext_domains import slice_compute_domain

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

import pandas as pd

from ..mpp.mpp import mpp_broadcast
from .chunks import get_chunk_bounds, get_effective_chunk_size, prune_chunk_info
from .halo import mpp_redistribute
from .meta import (
    choose_partition_dim,
    indexer_is_scalar,
    mpp_get_meta,
    mpp_set_domain_bounds,
    mpp_update_meta,
    reattach_meta,
    reattach_meta_after_collapse,
    strip_mpi_meta,
)
from .planning import _agree


def _select_partition_dim(
    meta: Mapping[str, Any], supplied: Mapping[Any, Any], *, caller: str
) -> Hashable | None:
    """Return the sole active partition dimension present in ``supplied``."""
    hit = tuple(dim for dim in meta["dims"] if dim in supplied)
    if not hit:
        return None
    if len(hit) > 1:
        raise NotImplementedError(
            f"{caller} supports one partition dimension per call; got {hit!r}."
        )
    return hit[0]


def mpp_isel(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    indexers: Mapping[Any, Any] | None = None,
    *,
    partition_dim: Hashable | Literal["auto"] | None = None,
    **indexers_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Index a distributed object with global integer coordinates.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to index.
    indexers : mapping, optional
        Integer indexers using global coordinates on the partition dimension.
    partition_dim : Hashable or {"auto"} or None, optional
        Only consulted when a *slice* on the partition dimension leaves a single global
        element behind (a scalar indexer already collapses the dimension entirely and
        broadcasts, so this does not apply there).
    **indexers_kwargs : Any
        Additional indexers passed by dimension name.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Indexed object with updated distribution metadata.

    """
    supplied = dict(indexers or {})
    supplied.update(indexers_kwargs)
    meta = mpp_get_meta(value)
    if meta is None:
        return value.isel(supplied)

    dim = _select_partition_dim(meta, supplied, caller="isel")
    if dim is None:
        return value.isel(supplied)

    distributed_indexer = supplied.pop(dim)
    if indexer_is_scalar(distributed_indexer):
        return mpp_isel_scalar(
            mpi_context, value, dim, int(distributed_indexer), supplied
        )

    if not isinstance(distributed_indexer, slice):
        raise NotImplementedError(
            "Distributed isel supports only slices or scalar indices."
        )
    if distributed_indexer.step not in (None, 1):
        raise NotImplementedError("Distributed isel currently requires slice step 1.")

    global_size = int(meta["global_sizes"][dim])
    requested_start, requested_stop, _ = distributed_indexer.indices(global_size)
    requested_stop = max(requested_start, requested_stop)

    # Compute slice offsets from local contiguous bounds; no cross-rank metadata
    # exchange is needed.
    local_start, local_stop, new_start = slice_compute_domain(
        int(meta["starts"][dim]),
        int(meta["stops"][dim]),
        requested_start,
        requested_stop,
    )
    local_indexers = dict(supplied)
    local_indexers[dim] = slice(local_start, local_stop)
    output = value.isel(local_indexers)

    new_global_size = requested_stop - requested_start
    if new_global_size == 1 and partition_dim is not None:
        # Rare enough not to be worth deriving every rank's share locally;
        # the branch itself is taken identically on every rank, so the
        # collective below stays consistent.
        dim_comm = _dim_comm(meta, dim, mpi_context)
        counts = gather_v(int(output.sizes[dim]), dim_comm)
        if len(meta["dims"]) > 1:
            raise NotImplementedError(
                f"Cannot redistribute collapsed partition dimension {dim!r}."
            )
        return _repartition_singleton(mpi_context, output, dim, counts, partition_dim)

    new_stop = new_start + (local_stop - local_start)
    chunk_info = prune_chunk_info(meta["chunk_info"], output)
    mpp_set_domain_bounds(
        output,
        meta,
        dim,
        global_size=new_global_size,
        start=new_start,
        stop=new_stop,
        chunk_info=chunk_info,
    )
    return output


def mpp_isel_scalar(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    index: int,
    other_indexers: Mapping[Any, Any],
) -> xr.Dataset | xr.DataArray:
    """Select one global integer index from the partition dimension.

    Raises
    ------
    IndexError
        If ``index`` is outside the global dimension.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        return value.isel({dim: index, **other_indexers})

    global_size = int(meta["global_sizes"][dim])
    normalized = index + global_size if index < 0 else index
    if normalized < 0 or normalized >= global_size:
        raise IndexError(
            f"Index {index} is out of bounds for {dim!r} (size {global_size})."
        )

    dim_comm = _dim_comm(meta, dim, mpi_context)
    # Find a scalar index owner with a fixed-size reduction instead of gathering rank
    # bounds.
    claim = np.array(
        [
            dim_comm.rank
            if int(meta["starts"][dim]) <= normalized < int(meta["stops"][dim])
            else -1
        ],
        dtype=np.int64,
    )
    elected = np.empty_like(claim)
    dim_comm.Allreduce(claim, elected, op=MPI.MAX)
    owner = int(elected[0])
    if owner < 0:
        raise RuntimeError("Distributed partitions do not own the requested index.")

    result = None
    if dim_comm.rank == owner:
        local_index = normalized - int(meta["starts"][dim])
        result = strip_mpi_meta(value).isel({dim: local_index, **other_indexers})
    return _broadcast_from_owner(mpi_context, dim_comm, owner, result, meta, dim)


def mpp_sel(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    indexers: Mapping[Any, Any] | None = None,
    method: str | None = None,
    tolerance: Any = None,
    drop: bool = False,
    *,
    partition_dim: Hashable | Literal["auto"] | None = None,
    **indexers_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Index a distributed object with global coordinate labels.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to index.
    indexers : mapping, optional
        Label indexers using global semantics on the partition dimension.
    method : str, optional
        Inexact matching method passed to xarray.
    tolerance : Any, optional
        Maximum distance for inexact matches.
    drop : bool, optional
        Drop selected coordinate variables.
    partition_dim : Hashable or {"auto"} or None, optional
        Only consulted when a label *slice* on the partition dimension leaves a single
        global element behind (a scalar label already collapses the dimension entirely
        and broadcasts, so this does not apply there).
    **indexers_kwargs : Any
        Additional indexers passed by dimension name.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Indexed object with updated distribution metadata.

    """
    supplied = dict(indexers or {})
    supplied.update(indexers_kwargs)
    meta = mpp_get_meta(value)
    if meta is None:
        return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

    dim = _select_partition_dim(meta, supplied, caller="sel")
    if dim is None:
        return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

    distributed_indexer = supplied.pop(dim)
    if indexer_is_scalar(distributed_indexer):
        return mpp_sel_scalar(
            mpi_context,
            value,
            dim,
            distributed_indexer,
            supplied,
            method=method,
            tolerance=tolerance,
            drop=drop,
        )

    if not isinstance(distributed_indexer, slice):
        raise NotImplementedError(
            "Distributed sel supports only slices or scalar labels."
        )

    local_indexers = dict(supplied)
    local_indexers[dim] = distributed_indexer
    output = value.sel(local_indexers, method=method, tolerance=tolerance, drop=drop)
    dim_comm = _dim_comm(meta, dim, mpi_context)

    local_length = int(output.sizes[dim])
    new_global_size, new_start, new_stop = partition_offsets(dim_comm, local_length)
    if new_global_size == 1 and partition_dim is not None:
        counts = gather_v(local_length, dim_comm)
        if len(meta["dims"]) > 1:
            raise NotImplementedError(
                f"Cannot redistribute collapsed partition dimension {dim!r}."
            )
        return _repartition_singleton(mpi_context, output, dim, counts, partition_dim)

    chunk_info = prune_chunk_info(meta["chunk_info"], output)
    mpp_set_domain_bounds(
        output,
        meta,
        dim,
        global_size=new_global_size,
        start=new_start,
        stop=new_stop,
        chunk_info=chunk_info,
    )
    return output


#: How to break a tie when several ranks offer an inexact match: ``pad``
#: wants the largest candidate label, the others the smallest.
_SEL_TIEBREAK = {
    "nearest": min,
    "pad": max,
    "ffill": max,
    "backfill": min,
    "bfill": min,
}


def _broadcast_from_owner(
    mpi_context: MPIContext,
    comm: MPI.Comm,
    owner: int,
    result: Any,
    meta: Mapping[str, Any] | None,
    dim: Hashable,
) -> xr.Dataset | xr.DataArray:
    """Share the owning rank's selection and restore partition metadata.

    The payload is materialised first because ``bcast`` pickles it, and a
    lazy array would otherwise be rebuilt on every receiving rank.
    """
    payload = result.load() if comm.rank == owner and result is not None else None
    shared = mpp_broadcast(payload, comm, root=owner)
    if meta is not None:
        shared = reattach_meta_after_collapse(shared, meta, dim)
    return cast("xr.Dataset | xr.DataArray", shared)


def mpp_sel_scalar(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    label: Any,
    other_indexers: Mapping[Any, Any],
    *,
    method: str | None,
    tolerance: Any,
    drop: bool,
) -> xr.Dataset | xr.DataArray:
    """Select one global label from the partition dimension."""
    meta = mpp_get_meta(value)
    dim_comm = mpi_context.comm if meta is None else _dim_comm(meta, dim, mpi_context)

    if method is not None:
        if meta is None:
            return value.sel(
                {dim: label, **other_indexers},
                method=method,
                tolerance=tolerance,
                drop=drop,
            )

        if dim in value.coords:
            local_coord = np.asarray(value[dim].values)
        else:
            local_coord = np.arange(int(meta["starts"][dim]), int(meta["stops"][dim]))
        local_start = int(meta["starts"][dim])

        # Resolve each rank's best inexact match locally; reduce only the candidate
        # metadata globally.
        pick = _SEL_TIEBREAK.get(method)
        if pick is None:
            raise NotImplementedError(
                f"Distributed sel does not support method={method!r}."
            )

        candidate: tuple[int, Any] | None = None
        if local_coord.size:
            locator = xr.DataArray(
                np.arange(local_coord.size, dtype=np.int64),
                dims=(dim,),
                coords={dim: local_coord},
            )
            try:
                selected = locator.sel({dim: label}, method=method, tolerance=tolerance)
            except (KeyError, IndexError):
                selected = None
            if selected is not None:
                if selected.ndim != 0:
                    raise NotImplementedError(
                        "Inexact sel requires a unique 1-D index."
                    )
                local_index = int(selected.item())
                matched_coord = local_coord[local_index]
                key = (
                    abs(matched_coord - label) if method == "nearest" else matched_coord
                )
                candidate = (local_start + local_index, key)

        # Exchange only one candidate tuple per rank to choose the global match.
        candidates = [c for c in gather_v(candidate, dim_comm) if c is not None]
        if not candidates:
            raise KeyError(f"No match for label {label!r} on {dim!r}.")
        global_index = pick(candidates, key=lambda pair: pair[1])[0]

        bounds = gather_v((int(meta["starts"][dim]), int(meta["stops"][dim])), dim_comm)
        owner = next(
            rank
            for rank, (start, stop) in enumerate(bounds)
            if start <= global_index < stop
        )

        result = None
        error: BaseException | None = None
        if dim_comm.rank == owner:
            try:
                local_index = global_index - int(meta["starts"][dim])
                result = strip_mpi_meta(value).isel({dim: local_index}, drop=drop)
                if other_indexers:
                    result = result.sel(
                        other_indexers,
                        method=method,
                        tolerance=tolerance,
                        drop=drop,
                    )
            except BaseException as exc:
                error = exc
        mpi_context.raise_if_error(error, "distributed scalar selection", comm=dim_comm)
        return _broadcast_from_owner(mpi_context, dim_comm, owner, result, meta, dim)

    result = None
    found = False
    try:
        result = strip_mpi_meta(value).sel(
            {dim: label, **other_indexers},
            method=method,
            tolerance=tolerance,
            drop=drop,
        )
        found = True
    except (KeyError, IndexError):
        pass

    # Use one fixed-size sum to detect and identify a unique matching rank.
    claim = np.array([int(found), dim_comm.rank if found else 0], dtype=np.int64)
    tally = np.empty_like(claim)
    dim_comm.Allreduce(claim, tally, op=MPI.SUM)
    owner_count = int(tally[0])
    if owner_count == 0:
        raise KeyError(f"No rank contains label {label!r} on {dim!r}.")
    if owner_count > 1:
        raise NotImplementedError("Scalar sel requires a unique owning rank.")
    return _broadcast_from_owner(
        mpi_context, dim_comm, int(tally[1]), result, meta, dim
    )


def _repartition_singleton(
    mpi_context: MPIContext,
    output: xr.Dataset | xr.DataArray,
    old_dim: Hashable,
    counts: list[int],
    partition_dim: Hashable | Literal["auto"],
) -> xr.Dataset | xr.DataArray:
    """Scatter a slice-``isel``/``sel`` result stranded on one rank."""
    owner = counts.index(1)
    stripped = strip_mpi_meta(output)
    comm = mpi_context.comm

    def _keep_single_owner() -> xr.Dataset | xr.DataArray:
        """Keep the singleton result on one owning rank."""
        new_start = sum(counts[: comm.rank])
        new_stop = new_start + counts[comm.rank]
        chunk_info = prune_chunk_info({str(old_dim): 1}, output)
        mpp_update_meta(
            output,
            dim=old_dim,
            global_size=1,
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
        return output

    candidates = {
        name: int(length) for name, length in stripped.sizes.items() if name != old_dim
    }
    target = partition_dim
    if target == "auto":
        if not candidates or not any(n > 1 for n in candidates.values()):
            return _keep_single_owner()
        target = choose_partition_dim(candidates, comm.size, rank=comm.rank)
    elif target not in candidates:
        raise ValueError(f"partition_dim={target!r} is absent after selection.")

    target_length = candidates[target]
    chunk_size = get_effective_chunk_size(target_length, None, comm.size)

    # Guard owner-side slicing before scatter so root failures cannot strand other
    # ranks.
    error: BaseException | None = None
    parts: list[xr.Dataset | xr.DataArray] | None = None
    if comm.rank == owner:
        try:
            parts = [
                stripped.isel(
                    {
                        target: slice(
                            *get_chunk_bounds(target_length, chunk_size, r, comm.size)
                        )
                    }
                )
                for r in range(comm.size)
            ]
        except BaseException as exc:
            error = exc
    mpi_context.raise_if_error(error, "isel/sel partition_dim scatter")

    local = scatter_v(parts if comm.rank == owner else None, comm, root=owner)

    start, stop = get_chunk_bounds(target_length, chunk_size, comm.rank, comm.size)
    info = {str(target): chunk_size}
    info = prune_chunk_info(info, local)
    for other_dim, other_length in local.sizes.items():
        info.setdefault(
            str(other_dim),
            get_effective_chunk_size(int(other_length), None, comm.size),
        )
    mpp_update_meta(
        local,
        dim=target,
        global_size=target_length,
        start=start,
        stop=stop,
        chunk_info=info,
    )
    return cast("xr.Dataset | xr.DataArray", local)


def mpp_reindex(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    indexers: Mapping[Hashable, Any] | None = None,
    *,
    method: str | None = None,
    tolerance: float | Iterable[float] | None = None,
    fill_value: Any = np.nan,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
    **indexers_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Reindex ``value`` onto new coordinate labels, redistributing if needed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reindex; distributed or replicated.
    indexers : mapping, optional
        New coordinate labels per dimension, exactly as
        ``xarray.Dataset.reindex``/``DataArray.reindex`` accepts.
    method : str, optional
        Forwarded to ``pandas.Index.get_indexer`` when the partition dimension is
        reindexed (``None``, ``"nearest"``, ``"ffill"``/ ``"pad"``,
        ``"bfill"``/``"backfill"``); forwarded to xarray's own ``reindex`` otherwise.
    tolerance : float or iterable of float, optional
        Forwarded to ``pandas.Index.get_indexer``/xarray's ``reindex``.
    fill_value : Any, optional
        Value used for labels with no match in ``value``.
    chunk_info : mapping, optional
        Reserved for parity with ``repartition``'s signature; not consulted by the
        redistributing path, which always balances.
    log_partitions : bool, optional
        Currently unused by the redistributing path.
    **indexers_kwargs : Any
        Additional indexers given as keywords, merged with ``indexers``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The reindexed object: rank-local (metadata preserved) if no partitioned
        dimension was touched; freshly, memory-scalably redistributed (new bounds,
        possibly a new global length) otherwise -- see ``mpp_redistribute``.

    Raises
    ------
    ValueError
        If no indexers are given.
    NotImplementedError
        If more than one active partition dimension is reindexed at once, or a reindexed
        partition dimension's new coordinate is not one-dimensional.

    """
    indexers = {**(indexers or {}), **indexers_kwargs}
    if not indexers:
        raise ValueError("requires at least one indexer")

    meta = mpp_get_meta(value)
    if meta is None:
        return value.reindex(
            indexers, method=method, tolerance=tolerance, fill_value=fill_value
        )

    partition_dims = meta["dims"]
    touched = tuple(str(d) for d in partition_dims if d in indexers)

    if not touched:
        result = strip_mpi_meta(value).reindex(
            indexers, method=method, tolerance=tolerance, fill_value=fill_value
        )
        reattach_meta(result, meta)
        return result

    if len(touched) > 1:
        raise NotImplementedError(
            f"Cannot redistribute multiple partition dims: {touched!r}."
        )

    dim = touched[0]
    new_labels = np.asarray(indexers[dim])
    if new_labels.ndim != 1:
        raise NotImplementedError(
            f"New {dim!r} labels must be 1-D; got {new_labels.shape!r}."
        )
    _agree(
        mpi_context,
        (
            "reindex",
            dim,
            int(new_labels.shape[0]),
            str(method),
            str(tolerance),
        ),
    )

    comm = _dim_comm(meta, dim, mpi_context)
    old_coord_local = np.asarray(value[dim].values)
    old_full_coord = np.concatenate(gather_v(old_coord_local, comm))
    old_index = pd.Index(old_full_coord)
    old_pos = old_index.get_indexer(new_labels, method=method, tolerance=tolerance)
    old_pos = old_pos.astype(np.int64)

    return mpp_redistribute(
        mpi_context,
        value,
        meta,
        dim,
        new_coord=new_labels,
        old_pos=old_pos,
        fill_value=fill_value,
    )


def mpp_sortby(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    by: Hashable | xr.DataArray | Sequence[Hashable | xr.DataArray],
    *,
    ascending: bool = True,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Sort ``value`` by one or more keys, redistributing if needed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to sort; distributed or replicated.
    by : Hashable, DataArray, or sequence of these
        Sort key(s): variable/coordinate name(s) or explicit DataArray(s), exactly as
        ``xarray.Dataset.sortby``/ ``DataArray.sortby`` accepts.
    ascending : bool, optional
        Sort order.
    chunk_info : mapping, optional
        Reserved for parity with ``repartition``'s signature; not consulted by the
        redistributing path, which always balances.
    log_partitions : bool, optional
        Currently unused by the redistributing path.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The sorted object: rank-local (metadata preserved) if no sort key varies along a
        partitioned dimension; freshly, memory-scalably redistributed otherwise -- see
        ``mpp_redistribute``.

    Raises
    ------
    NotImplementedError
        If the sort key(s) together vary along more than one active partition dimension
        under a multi-dimensional (Cartesian) partition, or a key is not one-dimensional
        along the partition dimension it varies along.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        return value.sortby(by, ascending=ascending)

    keys = list(by) if isinstance(by, (list, tuple)) else [by]
    touched_dims: set[str] = set()
    for key in keys:
        if isinstance(key, xr.DataArray):
            touched_dims.update(str(d) for d in key.dims)
            continue
        try:
            touched_dims.update(str(d) for d in value[key].dims)
        except (KeyError, TypeError):
            continue

    partition_dims = meta["dims"]
    touched = tuple(str(d) for d in partition_dims if d in touched_dims)

    if not touched:
        result = strip_mpi_meta(value).sortby(by, ascending=ascending)
        reattach_meta(result, meta)
        return result

    if len(touched) > 1:
        raise NotImplementedError(
            f"Sort keys span multiple partition dims: {touched!r}."
        )

    dim = touched[0]
    local_len = int(value.sizes[dim])
    key_arrays_local: list[np.ndarray[Any, Any]] = []
    for key in keys:
        arr = np.asarray(
            key.values if isinstance(key, xr.DataArray) else value[key].values
        )
        if arr.ndim != 1 or arr.shape[0] != local_len:
            raise NotImplementedError(
                f"Sort key {key!r} must be 1-D along {dim!r}; got {arr.shape!r}."
            )
        key_arrays_local.append(arr)

    key_signature = tuple(
        "<dataarray>" if isinstance(key, xr.DataArray) else str(key) for key in keys
    )
    _agree(mpi_context, ("sortby", dim, key_signature, bool(ascending)))

    comm = _dim_comm(meta, dim, mpi_context)
    full_keys = [np.concatenate(gather_v(arr, comm)) for arr in key_arrays_local]
    old_full_coord = np.concatenate(gather_v(np.asarray(value[dim].values), comm))
    # np.lexsort sorts by the *last* array primarily; reverse so the
    # first key in `by` is primary, matching xarray.sortby's own order.
    order = np.lexsort(tuple(reversed(full_keys)))
    if not ascending:
        order = order[::-1]
    old_pos = order.astype(np.int64)
    new_coord = old_full_coord[order]

    return mpp_redistribute(
        mpi_context,
        value,
        meta,
        dim,
        new_coord=new_coord,
        old_pos=old_pos,
        fill_value=np.nan,
    )
