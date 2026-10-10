"""Provide distributed groupby and resample reductions."""

from __future__ import annotations

import functools
from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v, reduce_scatter

if TYPE_CHECKING:
    from typing import Any

    from ..mpi.context import MPIContext

from ..mpp.mpp import extreme_identity
from .chunks import get_effective_chunk_size
from .meta import mpp_get_meta, mpp_update_meta, strip_mpi_meta
from .planning import (
    dataset_result,
    finish_local_reduction,
    local_reduction_meta,
    mpp_comm_reduce,
    mpp_finish,
    mpp_reduction_plan,
    mpp_resolve_comm,
    partial_dtype,
    skipna_enabled,
)

_GROUP_DIM = "_mpi_group"
_GROUP_OPS = ("sum", "mean", "count", "min", "max")


def _balanced_counts(total: int, size: int) -> list[int]:
    """Near-equal split of ``total`` into ``size`` nonnegative integer counts."""
    base, rem = divmod(total, size)
    return [base + (1 if r < rem else 0) for r in range(size)]


def _group_reduce_local(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    group: xr.DataArray,
    *,
    op: str,
    skipna: bool | None,
) -> xr.DataArray:
    """Apply a grouped reduction locally, preserving variables without ``dim``."""
    if dim not in variable.dims:
        return variable
    grouped = variable.groupby(group)
    if op == "count":
        return grouped.count(dim=dim, keep_attrs=False)
    method = getattr(grouped, op)
    return method(dim=dim, skipna=skipna, keep_attrs=False)


def _reduce_groups(
    mpi_context: MPIContext,
    local: xr.DataArray,
    op: MPI.Op,
    expect_dtype: Any,
    *,
    comm: MPI.Comm,
    counts: list[int] | None,
    labels: np.ndarray[Any, Any],
    replica_count: int,
    phase: str,
) -> xr.DataArray:
    """Reduce one per-group partial across ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    local : xarray.DataArray
        Rank-local partial, already reindexed onto the global label set.
    op : mpi4py.MPI.Op
        Reduction operator.
    expect_dtype : numpy.dtype
        Dtype the reduction is performed in.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    counts : list of int or None
        Groups each rank keeps. None reduces to every rank instead.
    labels : numpy.ndarray
        Group labels this rank keeps.
    replica_count : int
        Number of duplicate replicas included in a SUM.
    phase : str
        Collective diagnostic label.

    Returns
    -------
    xarray.DataArray
        Reduced partial indexed by ``labels``.
    """
    if counts is None:
        return mpp_comm_reduce(
            mpi_context,
            local,
            op,
            expect_dtype=expect_dtype,
            phase=phase,
            comm=comm,
            replica_count=replica_count,
        )
    axis = local.get_axis_num(_GROUP_DIM)
    raw = reduce_scatter(np.asarray(local.values), op, comm, counts, axis=axis)
    return xr.DataArray(raw, dims=local.dims).assign_coords({_GROUP_DIM: labels})


def _group_combine(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    group: xr.DataArray,
    global_labels: np.ndarray[Any, Any],
    *,
    op: str,
    skipna: bool | None,
    comm: MPI.Comm,
    replica_count: int = 1,
    counts: list[int] | None = None,
) -> xr.DataArray:
    """Combine rank-local per-group partials into a global result.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    variable : xarray.DataArray
        Variable being grouped.
    dim : Hashable
        Dimension grouped over.
    group : xarray.DataArray
        Group label per position along ``dim``.
    global_labels : numpy.ndarray
        Sorted union of labels across ranks.
    op : {"sum", "mean", "count", "min", "max"}
        Reduction applied within each group.
    skipna : bool or None
        Missing-value behavior, following xarray semantics.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    replica_count : int, default 1
        Number of duplicate replicas included in a SUM.
    counts : list of int or None, optional
        Groups each rank keeps, scattering the result. None keeps all
        groups on every rank.

    Returns
    -------
    xarray.DataArray
        Result indexed by the groups this rank keeps.
    """
    labels = global_labels
    if counts is not None:
        start = sum(counts[: comm.rank])
        labels = global_labels[start : start + counts[comm.rank]]

    def partial(kind: str, fill: Any, na: bool | None) -> xr.DataArray:
        """Reduce locally and align onto the global label set."""
        local = _group_reduce_local(
            mpi_context, variable, dim, group, op=kind, skipna=na
        )
        return local.reindex({_GROUP_DIM: global_labels}, fill_value=fill)

    def reduce(
        local: xr.DataArray, mpi_op: MPI.Op, dtype: Any, kind: str
    ) -> xr.DataArray:
        """Send one aligned partial through the collective."""
        return _reduce_groups(
            mpi_context,
            local,
            mpi_op,
            dtype,
            comm=comm,
            counts=counts,
            labels=labels,
            replica_count=replica_count,
            phase=f"MPI xarray groupby {kind} reduction",
        )

    if op == "mean":
        total = reduce(
            partial("sum", 0, skipna),
            MPI.SUM,
            partial_dtype(variable.dtype.str, "sum", skipna),
            "sum",
        )
        n = reduce(
            partial("count", 0, None),
            MPI.SUM,
            partial_dtype(variable.dtype.str, "count", None),
            "count",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = total / n
        # Match xarray groupby-mean promotion: preserve floating and complex
        # dtypes, promote everything else.
        if variable.dtype.kind in "fc" and result.dtype != variable.dtype:
            result = result.astype(variable.dtype, keep_attrs=True)
        return result.where(n > 0)

    if op in ("sum", "count"):
        return reduce(
            partial(op, 0, skipna),
            MPI.SUM,
            partial_dtype(variable.dtype.str, op, skipna),
            op,
        )

    minimum = op == "min"
    identity = extreme_identity(variable.dtype, minimum=minimum)
    mpi_op = MPI.MIN if minimum else MPI.MAX
    local = partial(op, identity, skipna)
    if variable.dtype.kind != "f":
        return reduce(local, mpi_op, variable.dtype, op)

    # A group present on a rank but entirely missing there reduces to NaN,
    # and MPI leaves MIN/MAX with a NaN operand undefined. Neutralise those
    # entries, then restore NaN where the group has no valid value anywhere
    # (skipna) or holds any NaN (no skipna).
    missing = local.isnull()
    result = reduce(local.fillna(identity), mpi_op, variable.dtype, op)
    if skipna_enabled(variable.dtype, skipna):
        valid = reduce(
            partial("count", 0, None),
            MPI.SUM,
            partial_dtype(variable.dtype.str, "count", None),
            "count",
        )
        return result.where(valid > 0)
    any_missing = reduce(
        missing.astype(np.int32), MPI.MAX, np.dtype(np.int32), "missing"
    )
    return result.where(any_missing == 0)


def _resample_bin_labels(
    timestamps: pd.DatetimeIndex, freq: str, comm: MPI.Comm
) -> pd.DatetimeIndex:
    """Rank-consistent resample bin-start label per element of ``timestamps``."""
    offset = pd.tseries.frequencies.to_offset(freq)

    # Detect fixed-duration offsets via ``.nanos``; pandas ``Tick`` subclasses are
    # incomplete for this purpose.
    try:
        delta_ns = int(offset.nanos)
        fixed_duration = True
    except (ValueError, AttributeError):
        fixed_duration = False

    if not fixed_duration:
        # Calendar offsets (weeks, months, ...) are labelled by pandas at the bin
        # edge it chooses (right edge for ``W``, ``ME``, ``QE``, ``YE``), and a
        # multiple such as ``2MS`` starts counting at the first timestamp. Label
        # every element with xarray's own resampler, seeded with the global
        # earliest timestamp so all ranks agree on the origin.
        local_min_ns = (
            int(timestamps.as_unit("ns").asi8.min())
            if len(timestamps)
            else np.iinfo(np.int64).max
        )
        global_min_ns = comm.allreduce(local_min_ns, op=MPI.MIN)
        if global_min_ns == np.iinfo(np.int64).max:
            return timestamps
        seed = np.datetime64(global_min_ns, "ns").astype(timestamps.dtype)
        combined = pd.DatetimeIndex(
            np.unique(np.concatenate([np.asarray(timestamps.values), [seed]]))
        )
        probe = xr.DataArray(np.arange(len(combined)), dims="t", coords={"t": combined})
        bin_labels = np.empty(len(combined), dtype=combined.dtype)
        for label, selection in probe.resample(t=freq).groups.items():
            bin_labels[selection] = label
        positions = combined.searchsorted(timestamps)
        return pd.DatetimeIndex(bin_labels[positions])

    # Derive one global start-day anchor with ``MPI.MIN`` and normalize timestamps to
    # nanoseconds.
    timestamps_ns = timestamps.as_unit("ns")
    local_min_ns = (
        int(timestamps_ns.asi8.min()) if len(timestamps_ns) else np.iinfo(np.int64).max
    )
    global_min_ns = comm.allreduce(local_min_ns, op=MPI.MIN)
    if global_min_ns == np.iinfo(np.int64).max:
        # No rank holds any timestamps at all; nothing to label.
        return timestamps
    anchor = pd.Timestamp(global_min_ns, unit="ns").normalize()

    if delta_ns <= 0:
        raise ValueError(f"non-positive Tick frequency {freq!r}")

    offsets_ns = timestamps_ns.asi8 - anchor.value
    bin_index = offsets_ns // delta_ns  # floor division: works for negatives too
    label_ns = anchor.value + bin_index.astype(np.int64) * delta_ns
    return pd.DatetimeIndex(label_ns.astype("datetime64[ns]")).as_unit(timestamps.unit)


def mpp_groupby_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    labels: xr.DataArray | np.ndarray[Any, Any],
    op: Literal["sum", "mean", "count", "min", "max"] = "mean",
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Reduce ``value`` over ``dim``, grouped by ``labels``.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : Hashable
        Dimension being grouped and reduced.
    labels : array-like
        Group key for every position along this rank's local ``dim`` axis.
        Need not be sorted or unique.
    op : {"sum", "mean", "count", "min", "max"}, optional
        Reduction applied within each group.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable, {"auto"}, or None, optional
        Partition placement after grouping. ``"auto"`` may place the result
        on the new group dimension.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced over ``dim``, carrying a new dimension of the same name
        indexed by the sorted, global set of group labels.
    """
    if op not in _GROUP_OPS:
        raise ValueError(f"Unsupported groupby op: {op!r}. Supported: {_GROUP_OPS}.")
    dims = (dim,)
    group = xr.DataArray(np.asarray(labels), dims=dim, name=_GROUP_DIM)
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)

    def with_attrs(result: xr.DataArray, source: xr.DataArray) -> xr.DataArray:
        """Reattach source attributes when the caller asked for them."""
        if keep_attrs:
            result.attrs.update(source.attrs)
        return result

    if local_meta is not None:
        if isinstance(value, xr.DataArray):
            result = with_attrs(
                _group_reduce_local(
                    mpi_context, value, dim, group, op=op, skipna=skipna
                ),
                value,
            )
        else:
            result = value.map(
                functools.partial(_group_reduce_local, mpi_context),
                dim=dim,
                group=group,
                op=op,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
        return finish_local_reduction(result, old_meta=local_meta)

    plan = mpp_reduction_plan(mpi_context, value, dims, old_meta, operation=op)
    labels_comm = mpp_resolve_comm(mpi_context, old_meta, (dim,))
    global_labels = np.unique(
        np.concatenate(gather_v(np.unique(group.values), labels_comm))
    )

    entries = [entry for entry in plan if entry.dims and entry.distributed]
    comms = {
        entry.name: mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        for entry in entries
    }
    # Scatter the combined groups only when the result will be partitioned on
    # the new group dimension and every variable reduces over one shape.
    can_scatter = (
        old_meta is not None
        and not tuple(d for d in old_meta["dims"] if d != dim)
        and partition_dim is not None
        and len(global_labels) > 1
        and bool(comms)
        and len({c.size for c in comms.values()}) == 1
        and all(entry.replica_count == 1 for entry in entries)
    )

    counts = start = stop = None
    scatter_comm = None
    if can_scatter:
        scatter_comm = next(iter(comms.values()))
        counts = _balanced_counts(len(global_labels), scatter_comm.size)
        start = sum(counts[: scatter_comm.rank])
        stop = start + counts[scatter_comm.rank]

    def combine(variable: xr.DataArray, entry: Any) -> xr.DataArray:
        """Reduce one distributed variable across its communicator."""
        return with_attrs(
            _group_combine(
                mpi_context,
                variable,
                dim,
                group,
                global_labels,
                op=op,
                skipna=skipna,
                comm=comms[entry.name],
                replica_count=entry.replica_count,
                counts=counts,
            ),
            variable,
        )

    def replicated(variable: xr.DataArray) -> xr.DataArray:
        """Reduce a variable every rank already holds in full."""
        full = _group_reduce_local(
            mpi_context, variable, dim, group, op=op, skipna=skipna
        )
        if counts is None:
            return with_attrs(full, variable)
        fill = (
            extreme_identity(variable.dtype, minimum=(op == "min"))
            if op in ("min", "max")
            else 0
        )
        sliced = full.reindex({_GROUP_DIM: global_labels[start:stop]}, fill_value=fill)
        return with_attrs(sliced, variable)

    if isinstance(value, xr.DataArray):
        result = combine(value, plan[0])
    else:
        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
            elif entry.distributed:
                variables[entry.name] = combine(variable, entry)
            else:
                variables[entry.name] = replicated(variable)
        result = dataset_result(value, dims, variables)

    if counts is None:
        return mpp_finish(
            mpi_context,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=frozenset({_GROUP_DIM}),
        )

    mpp_update_meta(
        result,
        dim=_GROUP_DIM,
        global_size=len(global_labels),
        start=start,
        stop=stop,
        chunk_info={
            str(other): get_effective_chunk_size(int(length), None, scatter_comm.size)
            for other, length in result.sizes.items()
        },
    )
    return result


def mpp_resample_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    freq: str,
    op: Literal["sum", "mean", "count", "min", "max"] = "mean",
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Resample a datetime dimension to ``freq``, then reduce."""
    timestamps = pd.DatetimeIndex(value[dim].values)
    labels = _resample_bin_labels(timestamps, freq, mpi_context.comm)
    result = mpp_groupby_reduce(
        mpi_context,
        value,
        dim,
        labels,
        op,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )

    # Rename the internal group dimension back to the resampled source
    # dimension, carrying any partition metadata across the rename.
    if _GROUP_DIM not in getattr(result, "dims", ()):
        return result
    meta = mpp_get_meta(result)
    renamed = strip_mpi_meta(result).rename({_GROUP_DIM: dim})
    if meta is not None:
        rename = {d: (dim if d == _GROUP_DIM else d) for d in meta["dims"]}
        mpp_update_meta(
            renamed,
            dim=tuple(rename.values()),
            global_size={new: meta["global_sizes"][old] for old, new in rename.items()},
            start={new: meta["starts"][old] for old, new in rename.items()},
            stop={new: meta["stops"][old] for old, new in rename.items()},
            chunk_info=meta["chunk_info"],
            cart=meta.get("cart"),
        )
    return renamed
