"""Provide MPI-aware elementwise, scan, and order-statistic operations."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v
from ..mpp.ext_domains import dim_comm as _dim_comm
from ..mpp.mpp import mpp_broadcast
from .chunks import prune_chunk_info
from .halo import mpp_halo_exchange
from .meta import (
    mpp_concat_along,
    mpp_get_meta,
    mpp_partition_meta,
    mpp_redefine_domain,
    mpp_update_meta,
    reattach_meta,
    strip_mpi_meta,
)
from .planning import _agree, guarded

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from ..mpi.context import MPIContext

#: Sentinel distinguishing "no fill value given" from a genuine ``other=None``.
_UNSET = object()


import ast
import operator
from collections.abc import Callable

from .meta import _partitions_match, mpp_operand_meta
from .planning import mpp_comm_reduce, mpp_resolve_comm

_MATMUL_CALLABLES: frozenset[Callable[..., Any]] = frozenset(
    {operator.matmul, np.matmul}
)
# Handle ``@`` separately because contracting a partition dimension requires MPI.
_AST_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

_AST_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_AST_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}


_AST_BOOL_OPS: dict[type[ast.boolop], Callable[[list[Any]], Any]] = {
    ast.And: lambda values: all(values),
    ast.Or: lambda values: any(values),
}


def mpp_where(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    cond: Any,
    other: Any = np.nan,
    *,
    drop: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Elementwise selection (``value.where(cond, other)``), MPI-safe.

    Raises
    ------
    ValueError
        If ``drop=True`` is requested on a distributed object, or the operands are
        distributed over incompatible partitions (see
        :meth:`~.arithmetic.Arithmetic.apply`).

    """
    operands = (value, cond, other)
    meta, reference = mpp_check_operands_distribution(mpi_context, operands)
    if meta is not None and drop:
        raise ValueError("drop=True is unsupported for distributed data.")

    _agree(
        mpi_context,
        (
            "where",
            None if meta is None else (str(meta["dim"]), int(meta["global_size"])),
        ),
    )
    result = value.where(cond) if other is _UNSET else value.where(cond, other)
    if meta is None:
        return result
    check_partition_preserved(result, meta, reference)
    return reattach_meta(result, meta)


def _prefix_scan(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    meta: Mapping[str, Any],
    *,
    product: bool,
    skipna: bool | None,
    keep_attrs: bool | None,
) -> xr.Dataset | xr.DataArray:
    """Cross-rank prefix-scan core of :func:`mpp_cumsum` and :func:`mpp_cumprod`.

    Each rank scans locally, then an exclusive scan of the rank totals supplies
    the offset its own prefix is missing.
    """
    operation = "cumprod" if product else "cumsum"

    def local() -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
        """Return this rank's local cumulative result and its total."""
        scan = (value.cumprod if product else value.cumsum)(
            dim, skipna=skipna, keep_attrs=keep_attrs
        )
        total = (value.prod if product else value.sum)(dim, skipna=skipna)
        return scan, total

    parts, error = guarded(local)
    mpi_context.raise_if_error(
        error, f"MPI xarray {operation}", signature=(operation, str(dim))
    )
    local_scan, local_total = parts
    # `.prod(dim)` alone does not force a still-lazy dask-backed `value` to
    # compute, and `comm.exscan` would pickle the graph as-is.
    local_total = local_total.load()

    comm = _dim_comm(meta, dim, mpi_context)
    # EXSCAN gives exclusive prefixes; rank 0 receives None and takes the
    # operator's identity instead.
    prefix = comm.exscan(local_total, op=MPI.PROD if product else MPI.SUM)
    if prefix is None:
        prefix = (xr.ones_like if product else xr.zeros_like)(local_total)

    return local_scan * prefix if product else local_scan + prefix


def _cumulative(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    product: bool,
    skipna: bool | None,
    keep_attrs: bool | None,
) -> xr.Dataset | xr.DataArray:
    """Shared implementation for :func:`mpp_cumsum` and :func:`mpp_cumprod`."""
    operation = "cumprod" if product else "cumsum"
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        method = value.cumprod if product else value.cumsum
        return method(dim, skipna=skipna, keep_attrs=keep_attrs)

    _agree(mpi_context, (operation, str(dim), int(meta["global_size"])))

    scan = functools.partial(
        _prefix_scan,
        mpi_context,
        dim=dim,
        meta=meta,
        product=product,
        skipna=skipna,
        keep_attrs=keep_attrs,
    )

    if not isinstance(value, xr.Dataset):
        return reattach_meta(scan(value), meta)

    # Only variables carrying ``dim`` are scanned; replicated ones pass through.
    touched = [name for name, var in value.data_vars.items() if dim in var.dims]
    if not touched:
        return strip_mpi_meta(value.copy(deep=False))
    untouched = [name for name in value.data_vars if name not in touched]
    scanned = scan(value[touched])
    result = (
        xr.merge([scanned, value[untouched]], combine_attrs="no_conflicts")
        if untouched
        else scanned
    )
    result.attrs = dict(value.attrs)
    return reattach_meta(result, meta)


def mpp_cumsum(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Compute a cumulative sum along a distributed dimension.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Input object.
    dim : Hashable
        Cumulative-sum dimension.
    skipna : bool or None, optional
        Skip missing values according to xarray semantics.
    keep_attrs : bool or None, optional
        Preserve attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Cumulative sum with the original partition layout.
    """
    return _cumulative(
        mpi_context, value, dim, product=False, skipna=skipna, keep_attrs=keep_attrs
    )


def mpp_cumprod(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Compute a cumulative product along a distributed dimension.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Input object.
    dim : Hashable
        Cumulative-product dimension.
    skipna : bool or None, optional
        Skip missing values according to xarray semantics.
    keep_attrs : bool or None, optional
        Preserve attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Cumulative product with the original partition layout.
    """
    return _cumulative(
        mpi_context, value, dim, product=True, skipna=skipna, keep_attrs=keep_attrs
    )


def mpp_ffill(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    limit: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Forward-fill along ``dim``, correct when ``dim`` is distributed."""
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        return value.ffill(dim, limit=limit)

    if limit is not None:
        _agree(mpi_context, ("ffill", str(dim), int(limit)))
        padded, left_pad, _right_pad = mpp_halo_exchange(
            mpi_context, value, dim, before=limit, after=0
        )
        filled = padded.ffill(dim, limit=limit)
        local_len = int(value.sizes[dim])
        trimmed = filled.isel({dim: slice(left_pad, left_pad + local_len)})
        return reattach_meta(trimmed, meta)

    _agree(mpi_context, ("ffill", str(dim), None))
    return reattach_meta(_fill_scan(mpi_context, value, dim, meta, forward=True), meta)


def mpp_bfill(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    limit: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Backward-fill along ``dim``, correct when ``dim`` is distributed."""
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        return value.bfill(dim, limit=limit)

    if limit is not None:
        _agree(mpi_context, ("bfill", str(dim), int(limit)))
        padded, _left_pad, right_pad = mpp_halo_exchange(
            mpi_context, value, dim, before=0, after=limit
        )
        filled = padded.bfill(dim, limit=limit)
        local_len = int(value.sizes[dim])
        trimmed = filled.isel({dim: slice(0, local_len)})
        return reattach_meta(trimmed, meta)

    _agree(mpi_context, ("bfill", str(dim), None))
    return reattach_meta(_fill_scan(mpi_context, value, dim, meta, forward=False), meta)


def _fill_scan(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    meta: Mapping[str, Any],
    *,
    forward: bool,
) -> xr.Dataset | xr.DataArray:
    """Unbounded ffill/bfill core: per-element carry from the neighbouring ranks.

    Each rank fills locally, then every position still missing at its edge
    takes the nearest valid value held by a rank on the upstream side. The
    carry is resolved per element: a column that is missing at one rank's
    edge must not hide valid edges of the other columns, nor stop an
    earlier rank's value from passing through.
    """
    comm = _dim_comm(meta, dim, mpi_context)

    def _local() -> tuple[xr.Dataset | xr.DataArray, Any]:
        """Return this rank's locally filled array and its boundary slice."""
        if int(value.sizes[dim]) == 0:
            return value, None
        local_filled = value.ffill(dim) if forward else value.bfill(dim)
        edge_slice = local_filled.isel({dim: -1 if forward else 0}, drop=True)
        # Materialize edge slices before object-based collectives to avoid
        # pickling lazy Dask graphs.
        return local_filled, edge_slice.load()

    local_or_none, error = guarded(_local)
    mpi_context.raise_if_error(
        error,
        "MPI xarray ffill/bfill",
        signature=("fill_scan", str(dim), forward),
        comm=comm,
    )
    local_filled, edge_slice = local_or_none

    edges = comm.allgather(edge_slice)
    rank = comm.rank
    # Upstream ranks ordered from the nearest to the farthest.
    upstream = edges[:rank][::-1] if forward else edges[rank + 1 :]
    carry = None
    for edge in upstream:
        if edge is None:
            continue
        carry = edge if carry is None else carry.fillna(edge)

    if carry is None:
        return local_filled
    return local_filled.fillna(carry)


#: Source points each interpolation method needs on either side of a target.
_INTERP_STENCIL = {"nearest": 1, "linear": 1, "slinear": 2, "quadratic": 3, "cubic": 3}


def _interp_halo_width(
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    new_coord: Any,
    meta: Mapping[str, Any],
    comm: MPI.Comm,
    method: str,
) -> int | None:
    """Return the halo that covers every rank's interpolation targets.

    Only the source coordinate is gathered, which costs one value per point
    along ``dim`` rather than one per array element. From it each rank works
    out how far outside its own compute domain its targets reach; the widest
    such reach, agreed across ranks, is the halo that serves all of them.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object being interpolated.
    dim : Hashable
        Dimension being interpolated along.
    new_coord : array-like
        This rank's target coordinate values.
    meta : mapping
        Distribution metadata for ``value``.
    comm : mpi4py.MPI.Comm
        Communicator varying along ``dim``.
    method : str
        Interpolation method, which sets how many source points a target
        needs on each side.

    Returns
    -------
    int or None
        Halo width to exchange, or None when no bounded halo suffices and
        the caller must reassemble the axis instead.
    """
    source = np.concatenate(gather_v(np.asarray(value[dim].values), comm))
    order = np.argsort(source, kind="stable")
    if not np.array_equal(order, np.arange(source.size)):
        # An unsorted source axis gives no locality to exploit.
        return None

    targets = np.asarray(new_coord)
    if targets.size == 0:
        reach = 0
    else:
        pad = _INTERP_STENCIL.get(method, 3)
        low = int(np.searchsorted(source, np.nanmin(targets), side="left")) - pad
        high = int(np.searchsorted(source, np.nanmax(targets), side="right")) + pad
        start, stop = int(meta["starts"][dim]), int(meta["stops"][dim])
        reach = max(start - low, high - stop, 0)

    agreed = np.empty(2, dtype=np.int64)
    comm.Allreduce(
        np.array([reach, -int(value.sizes[dim])], dtype=np.int64), agreed, op=MPI.MAX
    )
    width, shortest = int(agreed[0]), -int(agreed[1])
    # A halo can never exceed the narrowest compute domain on the axis.
    return width if width <= shortest else None


def mpp_interp(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    new_coord: Any,
    method: str = "linear",
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Interpolate onto ``new_coord`` along ``dim``, correct when distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to interpolate.
    dim : Hashable
        Dimension to interpolate along.
    new_coord : array-like
        This rank's own local slice of the new target coordinate along ``dim`` (not the
        global target grid -- exactly as this rank's own local ``value`` is its slice of
        the source, not the global source).
    method : str, optional
        As in ``xarray.DataArray.interp``.
    **kwargs : Any
        Forwarded to ``xarray.DataArray.interp``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Interpolated onto this rank's ``new_coord``, with ``.meta`` recomputed for the
        new length along ``dim`` (an allgather of each rank's own new local length, the
        same mechanism :func:`diff`/:func:`~.arithmetic.coarsen_reduce` use for their
        own length-changing case).

    """
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        return value.interp({dim: new_coord}, method=method, **kwargs)

    _agree(mpi_context, ("interp", str(dim), method))
    comm = _dim_comm(meta, dim, mpi_context)

    # Interpolation is local: a target point only needs the source points
    # bracketing it. Gathering the whole field would make every rank hold the
    # global array, so only the source coordinate is gathered -- one value per
    # point rather than one per element -- and the data it points at arrives
    # through a halo.
    width = _interp_halo_width(value, dim, new_coord, meta, comm, method)
    if width is not None:
        padded, _before, _after = mpp_halo_exchange(
            mpi_context, value, dim, before=width, after=width
        )
        result = padded.interp({dim: new_coord}, method=method, **kwargs)
    else:
        # A rank is asking for targets far outside its own span, so no
        # bounded halo can serve it; fall back to reassembling the axis.
        full = mpp_concat_along(gather_v(value, comm), value, dim)
        result = full.interp({dim: new_coord}, method=method, **kwargs)

    return mpp_redefine_domain(mpi_context, result, meta, dim)


def _order_statistic(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    operation: str,
    reduce_full: Callable[[xr.Dataset | xr.DataArray], xr.Dataset | xr.DataArray],
) -> xr.Dataset | xr.DataArray:
    """Compute an order statistic over a partitioned dimension.

    Unlike a sum or an extremum, an order statistic cannot be combined from
    per-rank partials: it needs every value along ``dim`` at once. The axis is
    gathered onto the root of its sub-communicator, reduced there with
    ordinary xarray, and broadcast back. Where other partition dimensions
    survive, only the root keeps the real result and the rest are emptied, so
    the value is owned exactly once.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce; ``dim`` must be partitioned.
    dim : Hashable
        Dimension to reduce over.
    operation : str
        Name used for collective agreement and diagnostics.
    reduce_full : callable
        Applied on the root to the reassembled object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object with distribution metadata for the surviving
        dimensions.
    """
    meta = mpp_get_meta(value)
    _agree(mpi_context, (operation, str(dim), int(meta["global_size"])))
    comm = _dim_comm(meta, dim, mpi_context)
    # Materialize before the object collective so a lazy Dask graph is never
    # pickled onto the wire.
    value = value.load()
    pieces = gather_v(value, comm, root=0)

    def _on_root() -> xr.Dataset | xr.DataArray:
        """Reassemble the axis and reduce it."""
        return reduce_full(mpp_concat_along(pieces, value, dim))

    result, error = guarded(_on_root) if comm.rank == 0 else (None, None)
    mpi_context.raise_if_error(
        error, f"MPI xarray {operation}", signature=(operation, str(dim)), comm=comm
    )
    result = strip_mpi_meta(mpp_broadcast(result, comm, root=0))

    remaining = tuple(d for d in meta["dims"] if d != dim)
    if not remaining:
        return result

    start = {d: int(meta["starts"][d]) for d in remaining}
    stop = {d: int(meta["stops"][d]) for d in remaining}
    if comm.rank != 0:
        # Replicas hold the same values; empty them so ownership stays unique.
        empty_dim = remaining[0]
        result = result.isel({empty_dim: slice(0, 0)})
        stop[empty_dim] = start[empty_dim]

    mpp_update_meta(
        result,
        dim=remaining,
        global_size={d: int(meta["global_sizes"][d]) for d in remaining},
        start=start,
        stop=stop,
        chunk_info=prune_chunk_info(meta["chunk_info"], result),
        cart=None,
    )
    return result


def mpp_median(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Median over ``dim``, correct when ``dim`` is distributed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object. Under a single partition dimension, fully
        replicated (``.meta`` is None) since nothing remains
        distributed. Under a multi-dimensional partition, metadata is
        reattached for whichever dimension(s) survive ``dim`` being
        reduced away, with no duplicated ownership: exactly one rank
        per distinct surviving range keeps the real result; every
        other rank that shared that range before the reduction is left
        with a genuinely empty (``start == stop``) slice instead of a
        redundant copy.

    """
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        return value.median(dim, skipna=skipna, keep_attrs=keep_attrs)
    return _order_statistic(
        mpi_context,
        value,
        dim,
        "median",
        lambda full: full.median(dim, skipna=skipna, keep_attrs=keep_attrs),
    )


def mpp_quantile(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    q: float | Iterable[float],
    dim: Hashable,
    *,
    method: str = "linear",
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Quantile(s) over ``dim``, correct when ``dim`` is distributed.

    Structurally identical to :func:`mpp_median` (a quantile is the same
    order statistic generalized from the 0.5 point to an arbitrary ``q``,
    and needs the same full view of ``dim`` to compute correctly) --
    gather this rank's local slice along ``dim`` onto ``dim``'s
    sub-communicator's root, compute the ordinary ``xarray`` quantile
    there, broadcast back, then apply the exact same no-duplicate-
    ownership dedup :func:`mpp_median` uses for the surviving
    dimensions. See that function's docstring for the full reasoning
    behind each step; only the reduction call itself and the extra
    ``quantile`` dimension a sequence ``q`` adds differ here.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object, with a new ``quantile`` dimension/coordinate when
        ``q`` is a sequence of more than one value. Under a single
        partition dimension, fully replicated (``.meta`` is None) since
        nothing remains distributed; under a multi-dimensional partition,
        exactly one rank per distinct surviving range keeps the real
        result, matching :func:`mpp_median`.

    """
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        return value.quantile(
            q, dim, method=method, skipna=skipna, keep_attrs=keep_attrs
        )
    return _order_statistic(
        mpi_context,
        value,
        dim,
        "quantile",
        lambda full: full.quantile(
            q, dim, method=method, skipna=skipna, keep_attrs=keep_attrs
        ),
    )


def mpp_diff(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    n: int = 1,
    *,
    label: Literal["upper", "lower"] = "upper",
) -> xr.Dataset | xr.DataArray:
    """``n``-th order difference along ``dim``, correct when ``dim`` is distributed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The differenced object, ``n`` elements shorter along ``dim`` globally -- and,
        when ``dim`` is the partition dimension, at exactly one rank (0 for "upper", the
        last rank for "lower") locally; every other rank's local length is unchanged.

    Raises
    ------
    ValueError
        If ``n`` is negative, ``label`` is not "upper"/"lower", or any rank's local
        length along ``dim`` is shorter than ``n`` (this last case is caught by
        :meth:`~.arithmetic.mpp_halo_exchange` itself, which checks every rank's local
        length together via a synchronized ``allgather`` before raising, so the error is
        consistent and every rank raises together rather than some hanging).

    """
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        return value.diff(dim, n=n, label=label)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n!r}")
    if label not in ("upper", "lower"):
        raise ValueError(f"label must be 'upper' or 'lower', got {label!r}")
    if n == 0:
        return reattach_meta(value.diff(dim, n=0, label=label), meta)

    before, after = (n, 0) if label == "upper" else (0, n)
    padded, _left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after
    )
    diffed = padded.diff(dim, n=n, label=label)

    return mpp_redefine_domain(mpi_context, diffed, meta, dim)


def mpp_shift(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    periods: int = 1,
    *,
    fill_value: Any = _UNSET,
) -> xr.Dataset | xr.DataArray:
    """Shift ``value`` by ``periods`` along ``dim``, correct when ``dim`` is
    distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to shift.
    dim : Hashable
        Dimension to shift along.
    periods : int, optional
        Number of positions to shift by; positive shifts values toward higher indices
        (as in ``xarray.DataArray.shift``).
    fill_value : Any, optional
        As in ``xarray.DataArray.shift``; defaults to xarray's own dtype-aware NA fill
        when omitted.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The shifted object, same shape and distribution as the input.

    """
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        kwargs = {} if fill_value is _UNSET else {"fill_value": fill_value}
        return value.shift({dim: periods}, **kwargs)
    if periods == 0:
        return value

    before, after = (periods, 0) if periods > 0 else (0, -periods)
    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after
    )
    kwargs = {} if fill_value is _UNSET else {"fill_value": fill_value}
    shifted = padded.shift({dim: periods}, **kwargs)

    local_len = int(value.sizes[dim])
    trimmed = shifted.isel({dim: slice(left_pad, left_pad + local_len)})
    return reattach_meta(trimmed, meta)


def mpp_pad(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    pad_width: tuple[int, int],
    *,
    mode: str = "constant",
    constant_values: Any = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Pad a distributed dimension at its global edges.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to pad.
    dim : Hashable
        Dimension to pad.
    pad_width : tuple[int, int]
        Number of values added before and after ``dim``.
    mode : str, default "constant"
        Xarray padding mode. Distributed dimensions support only ``"constant"``.
    constant_values : Any, optional
        Fill value for constant padding.
    keep_attrs : bool or None, optional
        Preserve attributes when supported by xarray.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Padded object with updated partition metadata.

    Raises
    ------
    NotImplementedError
        If a distributed dimension uses a non-constant padding mode.
    """
    before, after = pad_width
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        kwargs = {} if mode != "constant" else {"constant_values": constant_values}
        return value.pad({dim: pad_width}, mode=mode, keep_attrs=keep_attrs, **kwargs)
    if before == 0 and after == 0:
        return value
    if mode != "constant":
        raise NotImplementedError(
            f"Distributed pad supports only mode='constant'; got {mode!r}."
        )

    _agree(mpi_context, ("pad", str(dim), int(before), int(after), mode))

    start = int(meta["starts"][dim])
    stop = int(meta["stops"][dim])
    global_size = int(meta["global_sizes"][dim])
    is_lower_edge = start == 0
    is_upper_edge = stop == global_size

    side: tuple[int, int] = (
        before if is_lower_edge else 0,
        after if is_upper_edge else 0,
    )
    result = (
        value.pad(
            {dim: side},
            mode="constant",
            constant_values=constant_values,
            keep_attrs=keep_attrs,
        )
        if side != (0, 0)
        else value
    )

    new_start = 0 if is_lower_edge else start + before
    new_stop = stop + before + (after if is_upper_edge else 0)

    # Preserve metadata for every other partition dimension unchanged.
    global_sizes = dict(meta["global_sizes"])
    starts = dict(meta["starts"])
    stops = dict(meta["stops"])
    global_sizes[dim] = global_size + before + after
    starts[dim] = new_start
    stops[dim] = new_stop

    mpp_update_meta(
        result,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=prune_chunk_info(meta["chunk_info"], result),
        cart=meta.get("cart"),
    )
    return result


def mpp_roll(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    shift: int,
) -> xr.Dataset | xr.DataArray:
    """Circularly shift ``value`` by ``shift`` along ``dim``, wrapping at the edge."""
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        return value.roll({dim: shift}, roll_coords=False)

    global_size = int(meta["global_sizes"][dim])
    if global_size > 0:
        # Normalize periodic shifts to the smallest equivalent magnitude to minimize
        # halo width.
        shift = shift % global_size
        if shift > global_size // 2:
            shift -= global_size
    if shift == 0:
        return value

    before, after = (shift, 0) if shift > 0 else (0, -shift)
    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after, periodic=True
    )
    shifted = padded.shift({dim: shift})

    local_len = int(value.sizes[dim])
    trimmed = shifted.isel({dim: slice(left_pad, left_pad + local_len)})
    # Restore original dtypes after periodic shift because halo padding supplies real
    # values, not NaNs.
    if isinstance(value, xr.Dataset):
        original_dtypes = {name: var.dtype for name, var in value.variables.items()}
        for name, dtype in original_dtypes.items():
            if name in trimmed.variables and trimmed.variables[name].dtype != dtype:
                trimmed[name] = trimmed[name].astype(dtype)
    elif trimmed.dtype != value.dtype:
        trimmed = trimmed.astype(value.dtype)
    return reattach_meta(trimmed, meta)


def mpp_differentiate(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    coord: Hashable,
    edge_order: Literal[1, 2] = 1,
    datetime_unit: Any = None,
) -> xr.Dataset | xr.DataArray:
    """Differentiate ``value`` along ``coord``, correct when ``coord`` is distributed.

    Raises
    ------
    ValueError
        If any rank's local length along ``coord`` is shorter than 1 (see
        ``mpp_halo_exchange``'s own synchronized length check) or too short overall for
        ``edge_order`` (raised by xarray itself).

    """
    meta = mpp_partition_meta(value, coord)
    if meta is None:
        return value.differentiate(
            coord, edge_order=edge_order, datetime_unit=datetime_unit
        )

    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, coord, before=1, after=1
    )
    # Rechunk the local haloed axis once because ``dask.gradient`` requires sufficiently
    # wide chunks.
    if padded.chunks:
        padded = padded.chunk({coord: -1})
    derivative = padded.differentiate(
        coord, edge_order=edge_order, datetime_unit=datetime_unit
    )

    local_len = int(value.sizes[coord])
    trimmed = derivative.isel({coord: slice(left_pad, left_pad + local_len)})
    return reattach_meta(trimmed, meta)


def _realign_distributed(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    target_dim: str,
) -> xr.Dataset | xr.DataArray:
    """Move an already-distributed object onto the standard split of ``target_dim``.

    Two operands split differently have to be brought onto a common
    decomposition before they can be combined. Reassembling each one globally
    to do that would make every rank hold the whole field; this sends each
    rank only the elements it will own, which is what
    :func:`~xrmpi.core.halo.mpp_redistribute` is for.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.Dataset or xarray.DataArray
        Distributed object to move.
    meta : mapping
        Its current distribution metadata.
    target_dim : str
        Dimension the result is partitioned on.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The object on the standard decomposition of ``target_dim``.
    """
    from .halo import mpp_redistribute

    global_size = int(meta["global_sizes"][target_dim])
    coord = (
        np.concatenate(
            gather_v(
                np.asarray(value[target_dim].values),
                _dim_comm(meta, target_dim, mpi_context),
            )
        )
        if target_dim in value.coords
        else np.arange(global_size)
    )
    # Element order is unchanged; only which rank holds each element moves.
    return mpp_redistribute(
        mpi_context,
        value,
        meta,
        target_dim,
        new_coord=coord,
        old_pos=np.arange(global_size, dtype=np.int64),
        fill_value=np.nan,
    )


def _gather_full(
    mpi_context: MPIContext, value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]
) -> xr.Dataset | xr.DataArray:
    """Reconstruct ``value``'s full, replicated extent on every rank."""
    dim = meta["dim"]
    if len(meta["dims"]) > 1:
        raise NotImplementedError(
            f"Gathering partition dims {meta['dims']!r} is unsupported."
        )
    pieces = gather_v(value, mpi_context.comm)
    return strip_mpi_meta(mpp_concat_along(pieces, value, dim))


def _align_replicated(
    mpi_context: MPIContext,
    other: Any,
    meta: dict[str, Any],
    partner: xr.Dataset | xr.DataArray | None = None,
) -> Any:
    """Slice a replicated operand onto an already-distributed partner's bounds."""
    if not isinstance(other, (xr.Dataset, xr.DataArray)):
        return other
    shared_dims = tuple(dim for dim in meta["dims"] if dim in other.dims)
    if not shared_dims:
        return other

    indexers: dict[Hashable, slice] = {}
    for dim in shared_dims:
        length = int(other.sizes[dim])
        global_size = int(meta["global_sizes"][dim])
        if length != global_size:
            raise ValueError(
                f"Cannot align {dim!r}: length {length}, expected {global_size}."
            )
        indexers[dim] = slice(meta["starts"][dim], meta["stops"][dim])
    sliced = other.isel(indexers)

    if partner is not None:
        for dim in shared_dims:
            if dim not in getattr(partner, "indexes", {}) or dim not in getattr(
                sliced, "indexes", {}
            ):
                continue
            try:
                xr.align(partner, sliced, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Operand {dim!r} coordinates do not match this rank."
                ) from exc

    return reattach_meta(sliced, meta)


def mpp_align(
    mpi_context: MPIContext,
    left: xr.Dataset | xr.DataArray,
    right: xr.Dataset | xr.DataArray,
    dim: Hashable | Literal["auto"] | None = None,
    *,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
    """Return ``(left, right)`` partitioned identically across ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    left : xarray.Dataset or xarray.DataArray
        Left operand to align.
    right : xarray.Dataset or xarray.DataArray
        Right operand to align.
    dim : hashable or {"auto"}, optional
        Dimension to partition both operands along when neither is currently
        distributed, or the shared dimension to reconcile onto when both are already
        distributed differently.
    chunk_info : mapping, optional
        Forwarded to ``repartition``.
    log_partitions : bool, optional
        Forwarded to ``repartition``.

    Returns
    -------
    tuple of xarray.Dataset or xarray.DataArray
        ``(left, right)``, each carrying matching distribution metadata (or neither
        carrying any, if both remain replicated).

    Raises
    ------
    ValueError
        If neither operand is distributed and ``dim`` is omitted.

    """
    from .distribute import mpp_repartition

    left_meta = mpp_operand_meta(left)
    right_meta = mpp_operand_meta(right)

    if left_meta is not None and right_meta is not None:
        if _partitions_match(left_meta, right_meta):
            return left, right
        target_dim = dim if dim is not None else left_meta["dim"]
        if (
            len(left_meta["dims"]) == 1
            and len(right_meta["dims"]) == 1
            and target_dim in left_meta["dims"]
            and target_dim in right_meta["dims"]
        ):
            return (
                _realign_distributed(mpi_context, left, left_meta, target_dim),
                _realign_distributed(mpi_context, right, right_meta, target_dim),
            )
        full_left = _gather_full(mpi_context, left, left_meta)
        full_right = _gather_full(mpi_context, right, right_meta)
        return (
            mpp_repartition(
                mpi_context,
                full_left,
                target_dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            mpp_repartition(
                mpi_context,
                full_right,
                target_dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
        )

    if left_meta is not None:
        return left, _align_replicated(mpi_context, right, left_meta, partner=left)

    if right_meta is not None:
        return _align_replicated(mpi_context, left, right_meta, partner=right), right

    if dim is None:
        return left, right

    if (
        isinstance(left, (xr.Dataset, xr.DataArray))
        and isinstance(right, (xr.Dataset, xr.DataArray))
        and dim in getattr(left, "indexes", {})
        and dim in getattr(right, "indexes", {})
    ):
        try:
            xr.align(left, right, join="exact")
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Cannot align {dim!r}: coordinate labels differ."
            ) from exc

    return (
        mpp_repartition(
            mpi_context, left, dim, chunk_info=chunk_info, log_partitions=log_partitions
        ),
        mpp_repartition(
            mpi_context,
            right,
            dim,
            chunk_info=chunk_info,
            log_partitions=log_partitions,
        ),
    )


def mpp_check_operands_distribution(
    mpi_context: MPIContext, operands: Iterable[Any]
) -> tuple[dict[str, Any] | None, Any]:
    """Return the mpi_meta to attach to a multi-operand call's result.

    Returns
    -------
    tuple[dict[str, Any] | None, Any]
        ``(meta, reference)``: metadata to reattach to the result (or None when no
        operand is distributed) together with the first distributed operand itself, used
        by :meth:`apply` as the coordinate baseline for post-call validation.

    Raises
    ------
    ValueError
        If two operands are distributed over different partitions, if a replicated
        operand carries the distributed dimension at a different length than the
        partition owns, if a replicated operand's coordinate labels along the
        distributed dimension do not match the distributed partition's labels for this
        rank's slice (equal length alone does not imply equal coordinates), or (on more
        than one rank) if that coordinate check cannot even run because either side has
        no coordinate for the distributed dimension -- equal length alone is not enough
        evidence the operand is genuinely this rank's own data rather than another
        rank's same-length slice by coincidence.

    """
    operands = list(operands)
    metas = [mpp_operand_meta(item) for item in operands]

    ref_index = next((i for i, item in enumerate(metas) if item is not None), None)
    if ref_index is None:
        return None, None
    meta = metas[ref_index]
    reference = operands[ref_index]

    for other, other_meta in zip(operands, metas, strict=True):
        if other_meta is not None:
            if not _partitions_match(meta, other_meta):
                raise ValueError("Operands have different partition ownership.")
            continue

        for dim in meta["dims"]:
            if not (
                isinstance(other, (xr.Dataset, xr.DataArray)) and dim in other.dims
            ):
                continue
            owned = meta["stops"][dim] - meta["starts"][dim]
            local = int(other.sizes[dim])
            if local != owned:
                raise ValueError(
                    f"Operand {dim!r} length is {local}; expected {owned}."
                )
            reference_indexed = dim in getattr(reference, "indexes", {})
            other_indexed = dim in getattr(other, "indexes", {})
            if reference_indexed and other_indexed:
                try:
                    xr.align(reference, other, join="exact")
                except (ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Operand {dim!r} coordinates do not match this rank."
                    ) from exc
            elif mpi_context.comm.size > 1:
                # Without coordinates, equal local lengths cannot prove cross-rank
                # alignment; reject the ambiguous case.
                missing = [
                    name
                    for name, indexed in (
                        ("the distributed side", reference_indexed),
                        ("the operand", other_indexed),
                    )
                    if not indexed
                ]
                raise ValueError(
                    f"Cannot verify {dim!r} alignment: missing coordinate on "
                    + f"{' and '.join(missing)}."
                )
    return meta, reference


def check_partition_preserved(
    result: Any, meta: Mapping[str, Any], reference: Any
) -> None:
    """Verify ``result`` still owns the same partition-dimension slice.

    Parameters
    ----------
    result : Any
        The value returned by the callable.
    meta : Mapping[str, Any]
        The distribution metadata captured before the call.
    reference : Any
        The distributed operand the metadata was taken from, used as the coordinate
        baseline for the label check below.

    Raises
    ------
    ValueError
        If the distributed dimension is missing from ``result``, its local length
        changed, or its coordinate labels no longer match this rank's owned interval.

    """
    if not isinstance(result, (xr.Dataset, xr.DataArray)):
        return

    for dim in meta["dims"]:
        owned = meta["stops"][dim] - meta["starts"][dim]

        if dim not in result.dims:
            raise ValueError(f"Callable removed distributed dimension {dim!r}.")

        local = int(result.sizes[dim])
        if local != owned:
            raise ValueError(
                f"Callable changed local {dim!r} length from {owned} to {local}."
            )

        if (
            isinstance(reference, (xr.Dataset, xr.DataArray))
            and dim in getattr(reference, "indexes", {})
            and dim in getattr(result, "indexes", {})
        ):
            try:
                xr.align(reference, result, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Callable changed {dim!r} coordinates.") from exc


def mpp_apply(
    mpi_context: MPIContext, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    func : callable
        Any partition-preserving, rank-local function of the given ``args`` and
        ``kwargs``.
    *args : Any
        Positional arguments to ``func``: xarray Datasets or DataArrays (distributed or
        not) or plain scalars and arrays, in any mix.
    **kwargs : Any
        Keyword arguments to ``func``, checked for distribution metadata exactly like
        ``args``.

    Returns
    -------
    Any
        The result of ``func(*args, **kwargs)``.

    Raises
    ------
    ValueError
        If the xarray arguments are distributed over incompatible partitions or their
        coordinates disagree, or if the callable's result no longer represents the same
        owned partition (missing dimension, changed local length, or changed coordinate
        labels).

    """
    if func in _MATMUL_CALLABLES and not kwargs and len(args) == 2:
        return mpp_matmul(mpi_context, *args)

    return _apply_generic(mpi_context, func, args, kwargs)


def _apply_generic(
    mpi_context: MPIContext,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run the shared partition-preserving callable path."""
    meta, reference = mpp_check_operands_distribution(
        mpi_context, (*args, *kwargs.values())
    )

    _agree(
        mpi_context,
        (
            "apply",
            getattr(func, "__name__", repr(func)),
            None
            if meta is None
            else (
                tuple(str(d) for d in meta["dims"]),
                tuple(int(meta["global_sizes"][d]) for d in meta["dims"]),
            ),
        ),
    )

    result = func(*args, **kwargs)
    if meta is None:
        return result
    check_partition_preserved(result, meta, reference)
    return reattach_meta(result, meta)


def mpp_matmul(mpi_context: MPIContext, left: xr.DataArray, right: Any) -> xr.DataArray:
    """Matrix multiplication (``left @ right``), correct under MPI.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    left : xarray.DataArray
        Left operand.
    right : Any
        Right operand: an ``xarray.DataArray`` (distributed or not) or a plain
        array/scalar ``left`` can be matrix-multiplied with.

    Returns
    -------
    xarray.DataArray
        The matrix product.

    Raises
    ------
    ValueError
        If ``left``/``right`` are distributed over incompatible partitions (see
        :meth:`apply`).
    TypeError
        If the dtype involved has no MPI reduction datatype, when the distributed
        dimension is contracted.

    """
    meta, _reference = mpp_check_operands_distribution(mpi_context, (left, right))
    if meta is None:
        return _apply_generic(mpi_context, operator.matmul, (left, right), {})

    contracted = tuple(
        d
        for d in meta["dims"]
        if d in getattr(left, "dims", ()) and d in getattr(right, "dims", ())
    )
    if not contracted:
        # No partition dimension is contracted, so matrix multiplication is rank-local.
        return _apply_generic(mpi_context, operator.matmul, (left, right), {})
    if len(contracted) > 1:
        raise NotImplementedError(
            f"Cannot contract multiple partition dims: {contracted!r}."
        )
    dim = contracted[0]
    other_axes = tuple(d for d in meta["dims"] if d != dim)
    replicated = tuple(
        d
        for d in other_axes
        if not (d in getattr(left, "dims", ()) and d in getattr(right, "dims", ()))
    )
    if replicated:
        raise NotImplementedError(
            f"Cannot contract {dim!r}; operand is replicated over {replicated!r}."
        )

    _agree(mpi_context, ("matmul", str(dim), int(meta["global_sizes"][dim])))

    partial = operator.matmul(left, right)
    total = mpp_comm_reduce(
        mpi_context,
        partial,
        MPI.SUM,
        phase="MPI xarray distributed matrix multiplication",
        comm=mpp_resolve_comm(mpi_context, meta, (dim,)),
    )
    return strip_mpi_meta(total)


def _eval_ast_node(
    mpi_context: MPIContext, node: ast.expr, variables: Mapping[str, Any]
) -> Any:
    """Recursively evaluate one parsed expression node."""
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.MatMult):
            left = _eval_ast_node(mpi_context, node.left, variables)
            right = _eval_ast_node(mpi_context, node.right, variables)
            return mpp_matmul(mpi_context, left, right)

        function = _AST_BINARY_OPS.get(type(node.op))
        if function is None:
            raise ValueError(
                f"Unsupported expression operator: {type(node.op).__name__}."
            )
        left = _eval_ast_node(mpi_context, node.left, variables)
        right = _eval_ast_node(mpi_context, node.right, variables)
        return mpp_apply(mpi_context, function, left, right)

    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        last_val = None
        for val_node in node.values:
            last_val = _eval_ast_node(mpi_context, val_node, variables)
            if isinstance(last_val, (xr.Dataset, xr.DataArray)):
                raise TypeError("Use '&' or '|' for array boolean expressions.")
            if is_and and not last_val:
                return last_val
            if not is_and and last_val:
                return last_val
        return last_val

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError(
                "Chained comparisons are unsupported; combine separate comparisons."
            )
        function = _AST_COMPARE_OPS.get(type(node.ops[0]))
        if function is None:
            raise ValueError(
                f"Unsupported comparison operator: {type(node.ops[0]).__name__}."
            )
        left = _eval_ast_node(mpi_context, node.left, variables)
        right = _eval_ast_node(mpi_context, node.comparators[0], variables)
        return mpp_apply(mpi_context, function, left, right)

    if isinstance(node, ast.UnaryOp):
        function = _AST_UNARY_OPS.get(type(node.op))
        if function is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}.")
        operand = _eval_ast_node(mpi_context, node.operand, variables)
        return mpp_apply(mpi_context, function, operand)

    if isinstance(node, ast.Name):
        try:
            return variables[node.id]
        except KeyError:
            raise NameError(f"Undefined expression name {node.id!r}.") from None

    if isinstance(node, ast.Constant):
        return node.value

    raise ValueError(f"Unsupported expression node: {type(node).__name__}.")


def mpp_evaluate(mpi_context: MPIContext, expression: str, /, **variables: Any) -> Any:
    """Evaluate a string expression, respecting normal operator precedence.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    expression : str
        A Python expression referencing ``variables`` by name, for example ``"(a + b) *
        c - d / e"``.
    **variables : Any
        Values bound to the names used in ``expression``: xarray Datasets/DataArrays
        (distributed or not) or plain scalars.

    Returns
    -------
    Any
        The expression's value.

    Raises
    ------
    ValueError
        If ``expression`` fails to parse, uses an unsupported operator or expression
        element, or chains comparisons.
    NameError
        If ``expression`` references a name not present in ``variables``.

    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse expression {expression!r}: {exc}") from exc
    return _eval_ast_node(mpi_context, tree.body, variables)
