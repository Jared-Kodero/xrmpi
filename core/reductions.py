"""Distributed numerical, logical and positional reductions.

Every reduction here returns a result that is replicated across the ranks
that took part, or repartitioned onto a surviving dimension, with no
duplicated contributions. See :func:`~.planning.mpp_finish` for the exact
guarantee.
"""

from __future__ import annotations

import functools
from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import reduce_scatter

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from ..mpp.mpp import _mpp_reduce, extreme_identity
from .meta import mpp_get_meta, mpp_partition_meta
from .planning import (
    _FLAG_ENCODING,
    ReduceContext,
    guarded,
    mpp_comm_reduce,
    mpp_count_valid_values,
    mpp_global_reduce,
    mpp_sum_and_count,
    op_name,
    partial_dtype,
    residual_shape,
    skipna_enabled,
)

# Leading dimension the packed integer companions of a reproducing product
# travel under, so mantissa and companions can each go through one collective.
_PROD_FIELD_DIM = "_mpp_prod_field"


from .halo import mpp_halo_exchange
from .meta import mpp_operand_meta, mpp_redefine_domain, reattach_meta
from .planning import _agree


def _combine_sum_or_prod(
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial: xr.DataArray,
    dims: tuple[Hashable, ...],
    op: MPI.Op,
    *,
    skipna: bool | None,
    min_count: int | None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, list[int]] | None = None,
) -> xr.DataArray:
    """Combine rank-local sum or product partials."""
    if op_name(op) == "PROD":
        result = _combine_prod(
            mpi_context,
            value,
            partial,
            dims,
            skipna=skipna,
            error=error,
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )
    else:
        result = mpp_comm_reduce(
            mpi_context,
            partial,
            op,
            expect_dtype=partial_dtype(value.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray sum/prod reduction",
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
            expect_shape=residual_shape(value, dims),
        )
    global_count = None
    if min_count is not None and skipna_enabled(value.dtype, skipna):
        global_count = mpp_count_valid_values(
            mpi_context,
            value,
            dims,
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )
    if global_count is not None:
        # where() introduces NaN, which requires a floating result. Restore
        # the partial's own dtype so a float32 field stays float32.
        masked = result.where(global_count >= min_count)
        result = (
            masked
            if masked.dtype == result.dtype or result.dtype.kind not in "fc"
            else masked.astype(result.dtype, keep_attrs=True)
        )
    return result


def _combine_prod(
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None,
    error: BaseException | None,
    comm: MPI.Comm | None,
    replica_count: int,
    scatter: tuple[Hashable, list[int]] | None,
) -> xr.DataArray:
    """Combine rank-local products with explicit overflow handling.

    All of the decomposed fields are integers reduced with ``SUM``, so one
    collective replaces the separate mantissa and tally reductions and the
    result no longer depends on the rank count.
    """
    from ..mpp.ext_efp import prod_decompose, prod_recombine

    fields_da: xr.DataArray | None = None
    if error is None and partial is not None:
        try:
            axes = tuple(value.dims.index(d) for d in dims)
            fields = prod_decompose(np.asarray(value.values), axes)
            fields_da = xr.DataArray(
                fields,
                dims=(_PROD_FIELD_DIM, *partial.dims),
                coords={
                    d: partial.coords[d] for d in partial.dims if d in partial.coords
                },
            )
        except BaseException as exc:
            error = exc

    global_fields = mpp_comm_reduce(
        mpi_context,
        fields_da,
        MPI.SUM,
        expect_dtype=np.dtype(np.int64),
        error=error,
        phase="MPI xarray prod reduction",
        comm=comm,
        scatter=scatter,
    )

    values = np.asarray(global_fields.values)
    if replica_count != 1:
        # Every field is additive in log/exponent space, so the replica factor
        # divides exactly.
        values = values // replica_count

    expect = partial_dtype(value.dtype.str, "prod", skipna)
    combined = prod_recombine(values, expect)
    return global_fields.isel({_PROD_FIELD_DIM: 0}, drop=True).copy(data=combined)


def _global_valid_count(
    mpi_context: MPIContext,
    value: xr.DataArray,
    template: xr.DataArray,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None,
    comm: MPI.Comm | None,
    replica_count: int,
    scatter: tuple[Hashable, list[int]] | None,
) -> xr.DataArray:
    """Compute the global valid-value count when communication is required."""
    if skipna_enabled(value.dtype, skipna):
        return mpp_count_valid_values(
            mpi_context,
            value,
            dims,
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )

    meta = mpp_get_meta(value)
    global_sizes = dict(meta["global_sizes"]) if meta is not None else {}
    total = 1
    for reduced in dims:
        total *= int(global_sizes.get(reduced, value.sizes[reduced]))
    return xr.full_like(template, total, dtype=np.int64)


def _combine_mean(
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial_sum: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None = None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, list[int]] | None = None,
) -> xr.DataArray:
    """Combine rank-local sums and counts into a global mean."""
    global_sum, fused_count = mpp_sum_and_count(
        mpi_context,
        value,
        partial_sum,
        dims,
        skipna=skipna,
        sum_dtype=partial_dtype(value.dtype.str, "sum", skipna),
        error=error,
        phase="MPI xarray mean reduction",
        comm=comm,
        replica_count=replica_count,
        scatter=scatter,
    )
    global_count = (
        fused_count
        if fused_count is not None
        else _global_valid_count(
            mpi_context,
            value,
            global_sum,
            dims,
            skipna=skipna,
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )
    )
    # Match xarray mean promotion: full real reductions promote to float64; partial real
    # and complex reductions preserve floating dtype.
    kind = value.dtype.kind
    if kind not in "fc":
        target = np.dtype(np.float64)
    elif kind == "c":
        target = value.dtype
    else:
        is_full_reduction = set(dims) == set(value.dims)
        target = np.dtype(np.float64) if is_full_reduction else value.dtype
    divisor = (
        global_count.astype(target, keep_attrs=False)
        if target.kind in "fc"
        else global_count
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = global_sum / divisor
    result = result.where(global_count != 0)
    if result.dtype != target:
        result = result.astype(target, keep_attrs=True)
    return result


def _local_extreme_partial(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    variable_dims: tuple[Hashable, ...],
    *,
    empty: bool,
    minimum: bool,
    skipna: bool | None,
    keep_attrs: bool | None,
) -> xr.DataArray:
    """Compute a rank-local min/max partial."""
    if empty:
        identity = extreme_identity(variable.dtype, minimum=minimum)
        template = variable.sum(dim=variable_dims, skipna=False, keep_attrs=keep_attrs)
        return xr.full_like(template, identity, dtype=variable.dtype)
    method = variable.min if minimum else variable.max
    return method(dim=variable_dims, skipna=skipna, keep_attrs=keep_attrs)


def _combine_extreme(
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    minimum: bool,
    skipna: bool | None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
    scatter: tuple[Hashable, list[int]] | None = None,
) -> xr.DataArray:
    """Combine rank-local minimum or maximum partials."""
    # Use the agreed dtype so empty partitions cannot alter collective control flow.
    operation = "min" if minimum else "max"
    expect_dtype = value.dtype
    kind = value.dtype.kind
    reduced_shape = residual_shape(value, dims)
    if kind == "b":
        return mpp_comm_reduce(
            mpi_context,
            partial,
            MPI.LAND if minimum else MPI.LOR,
            expect_dtype=expect_dtype,
            error=error,
            phase=f"MPI xarray {operation} reduction",
            comm=comm,
            scatter=scatter,
            expect_shape=reduced_shape,
        )

    op = MPI.MIN if minimum else MPI.MAX
    if kind != "f":
        return mpp_comm_reduce(
            mpi_context,
            partial,
            op,
            expect_dtype=expect_dtype,
            error=error,
            phase=f"MPI xarray {operation} reduction",
            comm=comm,
            scatter=scatter,
            expect_shape=reduced_shape,
        )

    # Floating reductions carry validity beside the extreme so empty or all-NaN
    # partitions can use an identity without confusing it with real infinity.
    # Encoding the flag in the same buffer avoids a second boolean collective.
    send: np.ndarray[Any, Any] | None = None
    template: xr.DataArray | None = None
    use_skipna = skipna_enabled(value.dtype, skipna)
    # ANY valid rank suffices under skipna; without it every rank must be
    # NaN-free for the result to be defined.
    flip = -1.0 if ((not minimum) != use_skipna) else 1.0
    identity = extreme_identity(expect_dtype, minimum=minimum)
    healthy_flag, failed_flag = _FLAG_ENCODING[op_name(op)]

    if error is None:
        try:
            if use_skipna:
                good = value.count(dim=dims, keep_attrs=False) > 0
            else:
                good = ~value.isnull().any(dim=dims, keep_attrs=False)
            safe_partial = partial.where(good, other=identity)
            if safe_partial.dtype != expect_dtype:
                safe_partial = safe_partial.astype(expect_dtype, keep_attrs=True)
            template = safe_partial

            values = np.ascontiguousarray(
                np.asarray(safe_partial.values, dtype=expect_dtype)
            )
            flags = np.where(
                np.asarray(good.values, dtype=bool),
                np.asarray(flip, dtype=expect_dtype),
                np.zeros((), dtype=expect_dtype),
            )
            # One trailing column carries whether every rank built a buffer,
            # so the health check rides in this reduction instead of costing
            # a separate agreement round. Row 1 there takes the operator's
            # identity, which cannot perturb the flag in row 0.
            width = values.size + (0 if scatter is not None else 1)
            send = np.empty((2, width), dtype=expect_dtype)
            send[0, : values.size] = np.reshape(values, values.size)
            send[1, : values.size] = np.reshape(flags, values.size)
            if scatter is None:
                send[0, values.size] = healthy_flag
                send[1, values.size] = identity
        except BaseException as exc:
            error = exc
            send = None
            template = None

    if scatter is None and comm is not None and comm.size > 1:
        local_size = int(np.prod(reduced_shape, dtype=np.int64) or 1)
        if send is None:
            # A rank whose local reduction raised still enters the collective
            # its peers are committed to, flagging itself in the last column.
            send = np.full((2, local_size + 1), identity, dtype=expect_dtype)
            send[0, local_size] = failed_flag
        recv = _mpp_reduce(send, op, comm)
        if np.real(recv[0, local_size]) != healthy_flag:
            mpi_context.raise_if_error(
                error, f"MPI xarray {operation} reduction", None, comm=comm
            )
            raise AssertionError("MPI xarray reduction buffer is missing.")
        if template is None:
            raise AssertionError("MPI xarray reduction buffer is missing.")
        return _finish_extreme(
            recv[:, :local_size], template, expect_dtype, minimum=minimum, flip=flip
        )

    signature = (
        None
        if send is None
        else (
            operation,
            send.dtype.str,
            tuple(int(length) for length in send.shape),
            None
            if scatter is None
            else (str(scatter[0]), tuple(int(c) for c in scatter[1])),
        )
    )
    mpi_context.raise_if_error(
        error, f"MPI xarray {operation} reduction", signature, comm=comm
    )
    if send is None or template is None:
        raise AssertionError("MPI xarray reduction buffer is missing.")

    resolved_comm = comm if comm is not None else mpi_context.comm
    if scatter is not None:
        target, counts = scatter
        axis = 1 + template.get_axis_num(target)
        recv = reduce_scatter(send, op, resolved_comm, counts, axis=axis)
        start = sum(counts[: resolved_comm.rank])
        stop = start + counts[resolved_comm.rank]
        template = template.isel({target: slice(start, stop)})
    else:
        recv = _mpp_reduce(send, op, resolved_comm)

    return _finish_extreme(recv, template, expect_dtype, minimum=minimum, flip=flip)


def _finish_extreme(
    recv: np.ndarray[Any, Any],
    template: xr.DataArray,
    expect_dtype: np.dtype[Any],
    *,
    minimum: bool,
    flip: float,
) -> xr.DataArray:
    """Rebuild an extremum from its reduced value and validity rows.

    Positions no rank held a real value for carry the operator's identity,
    which is indistinguishable from a genuine infinity, so the validity row
    decides which ones become NaN.
    """
    shape = tuple(int(length) for length in template.shape)
    combined = np.asarray(recv[0]).reshape(shape)
    valid = (np.asarray(recv[1]).reshape(shape) * flip) > 0
    masked = np.where(valid, combined, np.asarray(np.nan, dtype=expect_dtype))
    return template.copy(data=np.asarray(masked, dtype=expect_dtype).reshape(shape))


def mpp_sum_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    min_count: int | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Sum a distributed xarray object over one or more dimensions."""
    return _sum_prod(
        mpi_context,
        value,
        dim,
        op=MPI.SUM,
        product=False,
        skipna=skipna,
        min_count=min_count,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def mpp_prod_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    min_count: int | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Multiply a distributed xarray object over one or more dimensions."""
    return _sum_prod(
        mpi_context,
        value,
        dim,
        op=MPI.PROD,
        product=True,
        skipna=skipna,
        min_count=min_count,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _sum_prod(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    op: MPI.Op,
    product: bool,
    skipna: bool | None,
    min_count: int | None,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
) -> xr.Dataset | xr.DataArray:
    """Implement distributed sum and product reductions."""

    def serial(obj: Any, dims: Any) -> Any:
        """Reduce without communication."""
        method = obj.prod if product else obj.sum
        return method(
            dim=dims, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs
        )

    def combine(variable: xr.DataArray, ctx: ReduceContext) -> xr.DataArray:
        """Reduce one distributed variable across ranks."""
        method = variable.prod if product else variable.sum
        local, error = guarded(
            lambda: method(
                dim=ctx.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        return _combine_sum_or_prod(
            mpi_context,
            variable,
            local,
            ctx.dims,
            op,
            skipna=skipna,
            min_count=min_count,
            error=error,
            comm=ctx.comm,
            replica_count=ctx.entry.replica_count,
            scatter=ctx.scatter,
        )

    return mpp_global_reduce(
        mpi_context,
        value,
        dim,
        operation="prod" if product else "sum",
        serial=serial,
        combine=combine,
        partition_dim=partition_dim,
    )


def _materialize_local(value: xr.DataArray) -> xr.DataArray:
    """Materialize a local Dask-backed array in memory."""
    return value.load() if getattr(value, "chunks", None) is not None else value


def mpp_mean_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the mean of a distributed xarray object.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable, {"auto"}, or None, optional
        Where to repartition the result."""

    def serial(obj: Any, dims: Any) -> Any:
        """Reduce without communication."""
        return obj.mean(dim=dims, skipna=skipna, keep_attrs=keep_attrs)

    def combine(variable: xr.DataArray, ctx: ReduceContext) -> xr.DataArray:
        """Reduce one distributed variable across ranks."""
        variable = _materialize_local(variable)
        local_sum, error = guarded(
            lambda: variable.sum(
                dim=ctx.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        return _combine_mean(
            mpi_context,
            variable,
            local_sum,
            ctx.dims,
            skipna=skipna,
            error=error,
            comm=ctx.comm,
            replica_count=ctx.entry.replica_count,
            scatter=ctx.scatter,
        )

    return mpp_global_reduce(
        mpi_context,
        value,
        dim,
        operation="mean",
        serial=serial,
        combine=combine,
        partition_dim=partition_dim,
    )


def mpp_min_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the minimum of a distributed xarray object."""
    return _min_max(
        mpi_context,
        value,
        dim,
        minimum=True,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def mpp_max_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the maximum of a distributed xarray object."""
    return _min_max(
        mpi_context,
        value,
        dim,
        minimum=False,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _min_max(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    minimum: bool,
    skipna: bool | None,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
) -> xr.Dataset | xr.DataArray:
    """Implement distributed minimum and maximum reductions."""
    partition_dims = (mpp_get_meta(value) or {}).get("dims", ())

    def serial(obj: Any, dims: Any) -> Any:
        """Reduce without communication."""
        method = obj.min if minimum else obj.max
        return method(dim=dims, skipna=skipna, keep_attrs=keep_attrs)

    def combine(variable: xr.DataArray, ctx: ReduceContext) -> xr.DataArray:
        """Reduce one distributed variable across ranks."""
        empty = any(
            d in variable.dims and int(variable.sizes[d]) == 0 for d in partition_dims
        )
        local, error = guarded(
            lambda: _local_extreme_partial(
                mpi_context,
                variable,
                ctx.dims,
                empty=empty,
                minimum=minimum,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
        )
        return _combine_extreme(
            mpi_context,
            variable,
            local,
            ctx.dims,
            minimum=minimum,
            skipna=skipna,
            error=error,
            comm=ctx.comm,
            scatter=ctx.scatter,
        )

    return mpp_global_reduce(
        mpi_context,
        value,
        dim,
        operation="min" if minimum else "max",
        serial=serial,
        combine=combine,
        partition_dim=partition_dim,
    )


def mpp_any_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Return whether any value is true over the requested dimensions.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Logical OR over the requested dimensions -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.

    """
    return _logical(
        mpi_context,
        value,
        dim,
        op=MPI.LOR,
        all_values=False,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def mpp_all_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Return whether all values are true over the requested dimensions.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Logical AND over the requested dimensions -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.

    """
    return _logical(
        mpi_context,
        value,
        dim,
        op=MPI.LAND,
        all_values=True,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _logical(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    op: MPI.Op,
    all_values: bool,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
) -> xr.Dataset | xr.DataArray:
    """Implement distributed logical reductions."""
    operation = "all" if all_values else "any"

    def serial(obj: Any, dims: Any) -> Any:
        """Reduce without communication."""
        method = obj.all if all_values else obj.any
        return method(dim=dims, keep_attrs=keep_attrs)

    def combine(variable: xr.DataArray, ctx: ReduceContext) -> xr.DataArray:
        """Reduce one distributed variable across ranks."""
        method = variable.all if all_values else variable.any
        local, error = guarded(lambda: method(dim=ctx.dims, keep_attrs=keep_attrs))
        return mpp_comm_reduce(
            mpi_context,
            local,
            op,
            expect_dtype=partial_dtype(variable.dtype.str, operation, None),
            error=error,
            phase=f"MPI xarray {operation} reduction",
            comm=ctx.comm,
        )

    return mpp_global_reduce(
        mpi_context,
        value,
        dim,
        operation=operation,
        serial=serial,
        combine=combine,
        partition_dim=partition_dim,
        allow_scatter=False,
    )


def _first_last_local(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rank-local first/last valid value along ``dim``, and its any-valid mask (both
    without ``dim``)."""
    size = int(variable.sizes[dim])
    if size == 0:
        template = variable.isel({dim: slice(0, 0)}).sum(
            dim=dim, skipna=False, keep_attrs=False
        )
        return template, xr.zeros_like(template, dtype=bool)

    if not skipna_enabled(variable.dtype, skipna):
        index = 0 if want_first else size - 1
        picked = variable.isel({dim: index}, drop=True)
        return picked, xr.ones_like(picked, dtype=bool)

    mask = variable.notnull()
    if want_first:
        index = mask.argmax(dim=dim)
    else:
        index = (size - 1) - mask.isel({dim: slice(None, None, -1)}).argmax(dim=dim)
    return variable.isel({dim: index}, drop=True), mask.any(dim=dim)


def _first_last_pick(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
) -> xr.DataArray:
    """Rank-local first/last, used when ``dim`` is not the partition dimension."""
    picked, any_valid = _first_last_local(
        mpi_context, variable, dim, skipna=skipna, want_first=want_first
    )
    return picked.where(any_valid) if variable.dtype.kind in "fc" else picked


def _first_last_combine(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
    comm: MPI.Comm | None = None,
) -> xr.DataArray:
    """Combine rank-local first/last candidates into a global result."""
    candidate, any_valid = _first_last_local(
        mpi_context, variable, dim, skipna=skipna, want_first=want_first
    )
    active_comm = mpi_context.comm if comm is None else comm
    rank, size = active_comm.rank, active_comm.size
    sentinel = size if want_first else -1
    owner, error = guarded(lambda: xr.where(any_valid, rank, sentinel).astype(np.int32))
    owner = mpp_comm_reduce(
        mpi_context,
        owner,
        MPI.MIN if want_first else MPI.MAX,
        expect_dtype=np.dtype(np.int32),
        error=error,
        phase="MPI xarray first/last owner election",
        comm=comm,
    )
    is_owner = owner == rank

    kind = variable.dtype.kind
    neutral = False if kind == "b" else np.zeros((), dtype=variable.dtype).item()
    payload, error = guarded(lambda: candidate.where(is_owner, other=neutral))
    combined = mpp_comm_reduce(
        mpi_context,
        payload,
        MPI.LOR if kind == "b" else MPI.SUM,
        expect_dtype=variable.dtype,
        error=error,
        phase="MPI xarray first/last value reduction",
        comm=comm,
    )
    result = combined.where(owner != sentinel) if kind in "fc" else combined

    # Elect first/last coordinates with the same owner selected for the data value.
    index_coords = {
        name: coord for name, coord in variable.coords.items() if dim in coord.dims
    }
    if index_coords:
        combined_coords: dict[Hashable, xr.DataArray] = {}
        for name, coord in index_coords.items():
            local_coord = candidate.coords[name]
            coord_kind = local_coord.dtype.kind
            # Reduce datetime/timedelta coordinates through lossless int64 views.
            as_int = coord_kind in "mM"
            reducible = local_coord.astype(np.int64) if as_int else local_coord
            reducible_kind = reducible.dtype.kind
            coord_neutral = (
                False
                if reducible_kind == "b"
                else np.zeros((), dtype=reducible.dtype).item()
            )
            coord_payload, coord_error = guarded(
                lambda reducible=reducible: reducible.where(
                    is_owner, other=coord_neutral
                )
            )
            coord_combined = mpp_comm_reduce(
                mpi_context,
                coord_payload,
                MPI.LOR if reducible_kind == "b" else MPI.SUM,
                expect_dtype=reducible.dtype,
                error=coord_error,
                phase="MPI xarray first/last coordinate reduction",
                comm=comm,
            )
            if reducible_kind in "fc":
                coord_combined = coord_combined.where(owner != sentinel)
            combined_coords[name] = (
                coord_combined.astype(local_coord.dtype) if as_int else coord_combined
            )
        result = result.assign_coords(combined_coords)
    return result


def mpp_first_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Select the first valid value along one dimension.

    Returns
    -------
    xr.Dataset | xr.DataArray
        First valid value along the requested dimension -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.

    """
    return _first_or_last(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        want_first=True,
    )


def mpp_last_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Select the last valid value along one dimension.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Last valid value along the requested dimension -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.

    """
    return _first_or_last(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        want_first=False,
    )


def _first_or_last(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
    want_first: bool,
) -> xr.Dataset | xr.DataArray:
    """Shared implementation for :meth:`first` and :meth:`last`."""
    if not isinstance(dim, str):
        raise TypeError("MPI xarray first/last reduce exactly one dimension.")

    def pick(variable: xr.DataArray, combined: bool, comm: MPI.Comm | None) -> Any:
        """Select the edge value, locally or across ranks."""
        chooser = _first_last_combine if combined else _first_last_pick
        extra = {"comm": comm} if combined else {}
        result = chooser(
            mpi_context, variable, dim, skipna=skipna, want_first=want_first, **extra
        )
        if keep_attrs:
            result.attrs.update(variable.attrs)
        return result

    def serial(obj: Any, dims: Any) -> Any:
        """Select the edge value without communication."""
        if isinstance(obj, xr.DataArray):
            return pick(obj, combined=False, comm=None)
        return obj.map(
            functools.partial(_first_last_pick, mpi_context),
            dim=dim,
            skipna=skipna,
            want_first=want_first,
            keep_attrs=keep_attrs,
        )

    def combine(variable: xr.DataArray, ctx: ReduceContext) -> xr.DataArray:
        """Select the edge value across ranks."""
        return pick(variable, combined=True, comm=ctx.comm)

    return mpp_global_reduce(
        mpi_context,
        value,
        dim,
        operation="first" if want_first else "last",
        serial=serial,
        combine=combine,
        partition_dim=partition_dim,
        allow_scatter=False,
    )


def _var_or_std(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    skipna: bool | None,
    ddof: int,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
    root: bool,
) -> xr.Dataset | xr.DataArray:
    """Shared implementation for :func:`mpp_var` and :func:`mpp_std`."""
    cached: list[Any] = []

    def global_mean() -> xr.Dataset | xr.DataArray:
        """Return the global mean, computed once and reused per variable."""
        if not cached:
            cached.append(
                mpp_mean_reduce(
                    mpi_context,
                    value,
                    dim,
                    skipna=skipna,
                    keep_attrs=False,
                    partition_dim=None,
                )
            )
        return cached[0]

    def serial(obj: Any, dims: Any) -> Any:
        """Reduce without communication."""
        method = obj.std if root else obj.var
        return method(dim=dims, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs)

    def combine(variable: xr.DataArray, ctx: ReduceContext) -> xr.DataArray:
        """Combine local squared deviations into a global variance."""
        mean = global_mean()
        if not isinstance(mean, xr.DataArray):
            mean = mean[ctx.entry.name]
        deviation = variable - mean
        # Squared deviations carry ``deviation.dtype`` because integer inputs
        # are promoted before reduction.
        partial, error = guarded(
            lambda: (deviation * deviation).sum(
                dim=ctx.dims, skipna=skipna, min_count=None, keep_attrs=False
            )
        )
        # The squared-deviation sum and the count it is divided by are both
        # SUM reductions of the same shape, so they share one collective.
        total, fused_count = mpp_sum_and_count(
            mpi_context,
            variable,
            partial,
            ctx.dims,
            skipna=skipna,
            sum_dtype=partial_dtype(deviation.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray variance reduction",
            comm=ctx.comm,
            replica_count=ctx.entry.replica_count,
            scatter=ctx.scatter,
        )
        count = (
            fused_count
            if fused_count is not None
            else mpp_count_valid_values(
                mpi_context,
                variable,
                ctx.dims,
                comm=ctx.comm,
                replica_count=ctx.entry.replica_count,
                scatter=ctx.scatter,
            )
        )
        denominator = count - ddof
        target = np.asarray(np.var(np.zeros(1, dtype=variable.dtype))).dtype
        divisor = (
            denominator.astype(target, keep_attrs=False)
            if target.kind in "fc"
            else denominator
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = total / divisor
        result = result.where(denominator > 0)
        if result.dtype != target:
            result = result.astype(target, keep_attrs=True)
        if root:
            result = np.sqrt(result)
        if keep_attrs:
            result.attrs.update(variable.attrs)
        return result

    return mpp_global_reduce(
        mpi_context,
        value,
        dim,
        operation="std" if root else "var",
        serial=serial,
        combine=combine,
        partition_dim=partition_dim,
    )


def mpp_var(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    ddof: int = 0,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the variance of a distributed xarray object."""
    return _var_or_std(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        ddof=ddof,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        root=False,
    )


def mpp_std(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    ddof: int = 0,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the standard deviation of a distributed xarray object."""
    return _var_or_std(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        ddof=ddof,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        root=True,
    )


def mpp_rolling_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    window: int,
    reduce: str = "mean",
    *,
    center: bool = True,
    min_periods: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Windowed reduction along ``dim``, correct when ``dim`` is distributed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rolled-and-reduced result, with the same local length and distribution
        metadata as the input when ``dim`` is the partition dimension.

    """
    meta = mpp_operand_meta(value)
    if meta is None or dim not in meta["dims"]:
        rolled = value.rolling({dim: window}, center=center, min_periods=min_periods)
        return getattr(rolled, reduce)()

    # Match xarray centered windows: even windows place the extra cell on the left.
    before = window // 2 if center else window - 1
    after = (window - 1) - before if center else 0

    # Halo coordinates are unused; restore the original compute-domain coordinate after
    # trimming.
    dim_coords = {
        name: coord for name, coord in value.coords.items() if dim in coord.dims
    }
    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after, exchange_coords=False
    )
    rolled = padded.rolling({dim: window}, center=center, min_periods=min_periods)
    reduced = getattr(rolled, reduce)()

    local_len = int(value.sizes[dim])
    trimmed = reduced.isel({dim: slice(left_pad, left_pad + local_len)})
    if dim_coords:
        trimmed = trimmed.assign_coords(dim_coords)
    return reattach_meta(trimmed, meta)


def mpp_coarsen_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    window: int,
    reduce: str = "mean",
    *,
    boundary: str = "exact",
    side: str = "left",
    coord_func: str = "mean",
) -> xr.Dataset | xr.DataArray:
    """Block reduction along ``dim``, correct when ``dim`` is distributed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The coarsened-and-reduced result, correctly distributed along the now
        block-reduced ``dim``.

    Raises
    ------
    ValueError
        If ``boundary="exact"`` and the global size is not evenly divisible by
        ``window``.
    NotImplementedError
        If ``side="right"`` is requested on a distributed ``dim``.

    """
    meta = mpp_partition_meta(value, dim)
    if meta is None:
        coarsened = value.coarsen(
            {dim: window}, boundary=boundary, side=side, coord_func=coord_func
        )
        return getattr(coarsened, reduce)()

    if side != "left":
        raise NotImplementedError("Distributed searchsorted supports only side='left'.")

    _agree(
        mpi_context,
        ("coarsen_reduce", str(dim), int(window), boundary, side),
    )

    global_size = int(meta["global_sizes"][dim])
    start = int(meta["starts"][dim])
    stop = int(meta["stops"][dim])
    remainder = global_size % window

    if boundary == "exact" and remainder != 0:
        raise ValueError(
            f"Size {global_size} is not divisible by window {window} "
            + "with boundary='exact'."
        )

    is_left_edge = start == 0
    is_right_edge = stop == global_size

    before_needed = 0 if is_left_edge else start % window
    after_needed = 0 if is_right_edge else (window - stop % window) % window

    # Request the common upper-bound halo ``window - 1`` on all ranks, then trim
    # locally.
    request = max(window - 1, 0)
    padded, left_pad, right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=request, after=request
    )
    # left_pad/right_pad are what was actually fetched (0 at a true
    # global edge, `request` everywhere else); keep only the slice
    # closest to this rank's own data on each side.
    padded = padded.isel(
        {
            dim: slice(
                left_pad - before_needed,
                left_pad + int(value.sizes[dim]) + after_needed,
            )
        }
    )

    local_boundary = "exact"
    if is_right_edge and remainder != 0:
        if boundary == "trim":
            trim_len = int(padded.sizes[dim]) - remainder
            padded = padded.isel({dim: slice(0, trim_len)})
        else:  # "pad": only the true global edge ever needs a synthetic
            # (non-neighbor-sourced) pad -- every interior boundary block
            # already got real data from mpp_halo_exchange above.
            local_boundary = "pad"

    coarsened = getattr(
        padded.coarsen(
            {dim: window}, boundary=local_boundary, side="left", coord_func=coord_func
        ),
        reduce,
    )()

    if before_needed > 0:
        # This rank's own first block started inside the left neighbor's
        # unpadded range (see the ownership rule in the docstring); the
        # left neighbor computes and reports the identical block itself.
        coarsened = coarsened.isel({dim: slice(1, None)})

    # Recompute global bounds after coarsen because the distributed dimension length
    # changes.
    return mpp_redefine_domain(mpi_context, coarsened, meta, dim)
