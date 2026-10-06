"""Plan distributed reductions and execute MPI collectives."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import reduce_scatter
from ..mpp.ext_domains import get_cartesian_domain

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from ..mpp.mpp import _mpp_reduce, extreme_identity
from .chunks import get_chunk_bounds, get_effective_chunk_size, prune_chunk_info
from .meta import choose_partition_dim, mpp_get_meta, mpp_update_meta, strip_mpi_meta

_OP_NAMES: tuple[str, ...] = ("SUM", "PROD", "MIN", "MAX", "LAND", "LOR")

MPI_REDUCIBLE_KINDS = "biufc"

# Check rank agreement before collectives so mismatched plans fail instead of
# deadlocking.
CHECK_COLLECTIVE_AGREEMENT = True


def op_name(op: MPI.Op) -> str:
    """Return a rank-stable label for an MPI reduction operation."""
    for name in _OP_NAMES:
        if op == getattr(MPI, name):
            return name
    return "OP"


@cache
def mpi_representable(dtype_string: str) -> bool:
    """Return whether a NumPy dtype has a usable predefined MPI datatype."""
    # Imported here: ``mpi4py.util.dtlib`` imports ``mpi4py.MPI`` (MPI_Init).
    from mpi4py.util import dtlib

    dtype = np.dtype(dtype_string)
    try:
        datatype = dtlib.from_numpy_dtype(dtype)
    except BaseException:
        return False
    try:
        return int(datatype.Get_size()) > 0
    except BaseException:
        return False


@cache
def partial_dtype(
    dtype_string: str, operation: str, skipna: bool | None
) -> np.dtype[Any]:
    """Return the dtype of a rank-local xarray reduction."""
    probe = xr.DataArray(np.zeros((1,), dtype=np.dtype(dtype_string)), dims=("_probe",))
    if operation == "count":
        return cast("np.dtype[Any]", probe.count(dim="_probe").dtype)
    if operation in ("any", "all"):
        method = probe.all if operation == "all" else probe.any
        return cast("np.dtype[Any]", method(dim="_probe").dtype)

    method = getattr(probe, operation)
    if operation in ("sum", "prod"):
        result = method(dim="_probe", skipna=skipna, min_count=None)
    else:
        result = method(dim="_probe", skipna=skipna)
    return cast("np.dtype[Any]", result.dtype)


class PlanEntry(NamedTuple):
    """Describe one variable in a rank-independent reduction plan.

    Attributes
    ----------
    name : Hashable
        Variable name.
    dims : tuple[Hashable, ...]
        Reduced dimensions present on the variable.
    distributed : bool
        Whether the reduction requires MPI communication.
    dtype : numpy.dtype
        Variable dtype.
    shape : tuple[tuple[str, int], ...]
        Global dimensions and lengths surviving the reduction.
    comm_axes : frozenset[str]
        Partition axes included in the collective.
    replica_count : int
        Number of replicated copies included in a SUM collective.
    """

    name: Hashable
    dims: tuple[Hashable, ...]
    distributed: bool
    dtype: np.dtype[Any]
    shape: tuple[tuple[str, int], ...]
    comm_axes: frozenset[str] = frozenset()
    replica_count: int = 1


def normalize_dim(
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
) -> tuple[Any, tuple[Hashable, ...]]:
    """Normalize a reduction dimension specification."""
    if not isinstance(value, (xr.DataArray, xr.Dataset)):
        raise TypeError("MPI xarray operations require an xarray DataArray or Dataset.")
    if dim is None or dim is ...:
        return dim, tuple(value.dims)
    if isinstance(dim, str):
        return dim, (dim,)
    dims = tuple(dim)
    return dims, dims


def skipna_enabled(dtype: np.dtype[Any], skipna: bool | None) -> bool:
    """Return the effective dtype-aware ``skipna`` setting."""
    if skipna is not None:
        return skipna
    return dtype.kind in "fc"


def _check_reducible(dtype: np.dtype[Any], operation: str) -> None:
    """Validate that a dtype supports the requested MPI reduction."""
    if operation in ("any", "all"):
        return
    if dtype.kind not in MPI_REDUCIBLE_KINDS:
        raise TypeError(f"No predefined MPI datatype for {dtype}.")
    if not mpi_representable(dtype.str):
        # Reject dtypes without predefined MPI types before entering a collective.
        raise TypeError(f"No predefined MPI datatype for {dtype}.")
    if operation in ("min", "max") and dtype.kind == "c":
        name = "minimum" if operation == "min" else "maximum"
        raise TypeError(f"MPI {name} is not defined for complex xarray data.")


def local_reduction_meta(
    meta: Mapping[str, Any] | None,
    dims: tuple[Hashable, ...],
    *,
    partition_dim: Hashable | Literal["auto"] | None,
) -> Mapping[str, Any] | None:
    """Return metadata when a reduction remains rank-local."""
    if meta is None or any(dim in dims for dim in meta["dims"]):
        return None
    if partition_dim not in (None, "auto"):
        raise ValueError(
            "New partition_dim requires reducing the active partition dimension."
        )
    return meta


def finish_local_reduction(
    result: xr.Dataset | xr.DataArray, *, old_meta: Mapping[str, Any]
) -> xr.Dataset | xr.DataArray:
    """Restore metadata after a rank-local reduction."""
    dims = tuple(dim for dim in old_meta["dims"] if dim in result.dims)
    if not dims:
        return strip_mpi_meta(result)
    mpp_update_meta(
        result,
        dim=dims,
        global_size={dim: int(old_meta["global_sizes"][dim]) for dim in dims},
        start={dim: int(old_meta["starts"][dim]) for dim in dims},
        stop={dim: int(old_meta["stops"][dim]) for dim in dims},
        chunk_info=prune_chunk_info(old_meta["chunk_info"], result),
        cart=old_meta.get("cart"),
    )
    return result


def _agree(mpi_context: MPIContext, signature: tuple[Any, ...]) -> None:
    """Verify that all ranks entered the same reduction plan."""
    if not CHECK_COLLECTIVE_AGREEMENT or mpi_context.comm.size == 1:
        return
    digest = hashlib.blake2b(repr(signature).encode(), digest_size=16).digest()
    mpi_context.raise_if_error(
        None,
        "MPI xarray reduction planning",
        signature=("xarray_reduction_plan", digest),
    )


def mpp_reduction_plan(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dims: tuple[Hashable, ...],
    meta: Mapping[str, Any] | None,
    *,
    operation: str,
) -> tuple[PlanEntry, ...]:
    """Build and validate the rank-independent reduction plan."""
    if isinstance(value, xr.DataArray):
        items: tuple[tuple[Hashable, xr.DataArray], ...] = ((value.name, value),)
    else:
        items = tuple((name, value[name]) for name in value.data_vars)

    partition_dims: tuple[Hashable, ...] = () if meta is None else meta["dims"]
    grid_shape_by_dim: dict[Hashable, int] = {}
    if meta is not None and "cart" in meta:
        grid_shape_by_dim = dict(
            zip(meta["dims"], meta["cart"]["grid_shape"], strict=True)
        )

    entries = []
    for name, variable in items:
        variable_dims = tuple(dim for dim in dims if dim in variable.dims)
        if variable_dims:
            _check_reducible(variable.dtype, operation)

        owned = tuple(dim for dim in partition_dims if dim in variable.dims)
        replicated = tuple(dim for dim in partition_dims if dim not in variable.dims)
        reduced = tuple(dim for dim in variable_dims if dim in owned)
        comm_axes = (
            frozenset(reduced) | frozenset(replicated) if reduced else frozenset()
        )
        replica_count = (
            math.prod(grid_shape_by_dim.get(dim, 1) for dim in replicated)
            if reduced and replicated
            else 1
        )
        if replica_count != 1 and operation == "prod":
            # Replicated products cannot be deduplicated reliably because inversion
            # requires an n-th root.
            raise NotImplementedError(
                "Product reduction cannot remove replicated dimensions "
                + f"{tuple(str(d) for d in replicated)!r}."
            )

        entries.append(
            PlanEntry(
                name=name,
                dims=variable_dims,
                distributed=bool(comm_axes),
                dtype=variable.dtype,
                shape=tuple(
                    (
                        str(dim),
                        # Use global sizes for surviving partition dimensions because
                        # local extents may differ by rank.
                        int(
                            meta["global_sizes"][dim]
                            if meta is not None and dim in partition_dims
                            else value.sizes[dim]
                        ),
                    )
                    for dim in variable.dims
                    if dim not in variable_dims
                ),
                comm_axes=comm_axes,
                replica_count=int(replica_count),
            )
        )

    plan = tuple(entries)
    _agree(
        mpi_context,
        (
            operation,
            tuple(str(dim) for dim in dims),
            tuple(
                (
                    str(entry.name),
                    tuple(str(dim) for dim in entry.dims),
                    entry.distributed,
                    str(entry.dtype),
                    entry.shape,
                    tuple(sorted(str(d) for d in entry.comm_axes)),
                    entry.replica_count,
                )
                for entry in plan
            ),
        ),
    )
    return plan


def mpp_resolve_comm(
    mpi_context: MPIContext,
    meta: Mapping[str, Any] | None,
    comm_axes: Iterable[Hashable],
) -> MPI.Comm:
    """Return the communicator a plan entry's collective should use."""
    axes = frozenset(comm_axes)
    if meta is None or not axes or "cart" not in meta or len(meta["dims"]) <= 1:
        return mpi_context.comm
    topology = get_cartesian_domain(
        mpi_context.comm, meta["dims"], meta["global_sizes"]
    )
    return topology.sub_comm(axes)


def guarded(function: Any) -> tuple[Any, BaseException | None]:
    """Run a local operation and defer any exception for synchronization."""
    try:
        return function(), None
    except BaseException as exc:
        return None, exc


#: What the health flag contributes under each operator: a healthy rank sends
#: the operator's identity, so it cannot perturb the reduced flag, and a
#: failed rank sends a value the operator preserves.
_FLAG_ENCODING: dict[str, tuple[float, float]] = {
    "SUM": (0.0, 1.0),
    "PROD": (1.0, 0.0),
    "MIN": (0.0, -1.0),
    "MAX": (0.0, 1.0),
    "LAND": (1.0, 0.0),
    "LOR": (0.0, 1.0),
}


def residual_shape(value: xr.DataArray, dims: tuple[Hashable, ...]) -> tuple[int, ...]:
    """Return the local shape a reduction of ``value`` over ``dims`` leaves.

    Derived from the input's sizes alone, so a rank whose local reduction
    raised can still build a correctly shaped neutral buffer and enter the
    collective its peers are committed to.
    """
    reduced = set(dims)
    return tuple(int(value.sizes[d]) for d in value.dims if d not in reduced)


def _reduction_identity(name: str, dtype: np.dtype[Any]) -> Any:
    """Return the value that leaves a reduction under ``name`` unchanged."""
    if name in ("SUM", "LOR"):
        return np.zeros((), dtype=dtype)
    if name in ("PROD", "LAND"):
        return np.ones((), dtype=dtype)
    return extreme_identity(dtype, minimum=(name == "MIN"))


def mpp_comm_reduce(
    mpi_context: MPIContext,
    value: xr.DataArray | None,
    op: MPI.Op,
    *,
    expect_dtype: np.dtype[Any] | None = None,
    error: BaseException | None = None,
    phase: str = "MPI xarray reduction buffer preparation",
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, Sequence[int]] | None = None,
    expect_shape: tuple[int, ...] | None = None,
) -> xr.DataArray:
    """Reduce a validated DataArray buffer across ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.DataArray or None
        Collective buffer.
    op : mpi4py.MPI.Op
        Reduction operation.
    expect_dtype : numpy.dtype, optional
        Expected buffer dtype.
    error : BaseException, optional
        Deferred local error.
    phase : str
        Collective diagnostic label.
    comm : mpi4py.MPI.Comm, optional
        Reduction communicator.
    replica_count : int, default 1
        Number of duplicate replicas included in a SUM.
    scatter : tuple[Hashable, Sequence[int]], optional
        Target dimension and per-rank counts for ``Reduce_scatter``.

    Returns
    -------
    xarray.DataArray
        Globally reduced result or this rank's scattered slice.
    """
    send: np.ndarray[Any, Any] | None = None
    if error is None:
        try:
            if value is None:
                raise AssertionError("MPI xarray reduction buffer is missing.")
            send = np.asarray(value.values)
            if expect_dtype is not None and send.dtype != np.dtype(expect_dtype):
                send = send.astype(expect_dtype)
            if not send.flags.c_contiguous:
                send = np.ascontiguousarray(send)
            if send.dtype.kind not in MPI_REDUCIBLE_KINDS:
                raise TypeError(f"No predefined MPI datatype for {send.dtype}.")
            if not mpi_representable(send.dtype.str):
                raise TypeError(f"No predefined MPI datatype for {send.dtype}.")
        except BaseException as exc:
            error = exc
            send = None

    resolved_comm = comm if comm is not None else mpi_context.comm
    name = op_name(op)
    fused = (
        scatter is None
        and expect_shape is not None
        and name in _FLAG_ENCODING
        and resolved_comm.size > 1
        and (send is None or send.shape == expect_shape)
    )

    if fused:
        # One collective carries both the payload and whether every rank
        # produced one, instead of a separate agreement round beforehand.
        dtype = np.dtype(expect_dtype) if expect_dtype is not None else send.dtype
        healthy_flag, failed_flag = _FLAG_ENCODING[name]
        count = int(np.prod(expect_shape, dtype=np.int64)) if expect_shape else 1
        buffer = np.empty(count + 1, dtype=dtype)
        if send is None:
            buffer[:count] = _reduction_identity(name, dtype)
            buffer[count] = failed_flag
        else:
            buffer[:count] = send.reshape(count)
            buffer[count] = healthy_flag
        reduced = _mpp_reduce(buffer, op, resolved_comm)
        if np.real(reduced[count]) != healthy_flag:
            # Rare. Every rank read the same flag, so they all reach the
            # descriptive protocol together.
            mpi_context.raise_if_error(error, phase, None, comm=comm)
            raise AssertionError("MPI xarray reduction buffer is missing.")
        if value is None:
            raise AssertionError("MPI xarray reduction buffer is missing.")
        result = value.copy(data=reduced[:count].reshape(expect_shape))
        return _apply_replica_count(result, op, replica_count)

    signature = (
        None
        if send is None
        else (name, send.dtype.str, tuple(int(length) for length in send.shape))
    )
    mpi_context.raise_if_error(error, phase, signature, comm=comm)
    if send is None or value is None:
        raise AssertionError("MPI xarray reduction buffer is missing.")

    if scatter is not None:
        target, counts = scatter
        axis = value.dims.index(target)
        recv = reduce_scatter(send, op, resolved_comm, counts, axis=axis)
        start = sum(counts[: resolved_comm.rank])
        stop = start + counts[resolved_comm.rank]
        result = value.isel({target: slice(start, stop)}).copy(data=recv)
    else:
        recv = _mpp_reduce(send, op, resolved_comm)
        result = value.copy(data=recv)
    return _apply_replica_count(result, op, replica_count)


def _apply_replica_count(
    result: xr.DataArray, op: MPI.Op, replica_count: int
) -> xr.DataArray:
    """Undo the duplication a replicated SUM introduced."""
    if replica_count != 1 and op == MPI.SUM:
        # Duplicate contributions make the factor exact.
        if result.dtype.kind in "iu":
            result = result // replica_count
        else:
            result = result / replica_count
    return result


def mpp_can_fuse_count(dtype: np.dtype[Any]) -> bool:
    """Return whether a valid-value count can share a buffer of ``dtype``.

    ``float32`` carries integers exactly only to ``2**24``, which a single
    global field routinely exceeds, so 4-byte reductions keep their own
    collective rather than silently rounding a mean's denominator.
    """
    return dtype.itemsize >= 8 and dtype.kind in "fic"


def mpp_sum_and_count(
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial_sum: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None,
    sum_dtype: np.dtype[Any],
    error: BaseException | None = None,
    phase: str = "MPI xarray sum reduction",
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, Sequence[int]] | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Reduce a partial sum and its valid-value count together.

    Both are ``SUM`` reductions of the same shape over the same communicator,
    so they travel in one buffer and cost one collective instead of two.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.DataArray
        Reduction input, used for the count and the residual shape.
    partial_sum : xarray.DataArray or None
        Rank-local sum, or None if the local reduction raised.
    dims : tuple of Hashable
        Dimensions being reduced.
    skipna : bool or None
        Missing-value behavior, following xarray semantics.
    sum_dtype : numpy.dtype
        Dtype the summed buffer is reduced in.
    error : BaseException, optional
        Deferred local error.
    phase : str
        Collective diagnostic label.
    comm : mpi4py.MPI.Comm, optional
        Reduction communicator.
    replica_count : int, default 1
        Duplicate replicas included in the SUM.
    scatter : tuple, optional
        Target dimension and per-rank counts for ``Reduce_scatter``.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray or None]
        Global sum, and the count when it was fused into the same
        collective. A None count means the caller should obtain it its own
        way, which for ``skipna=False`` needs no communication at all.
    """
    resolved_comm = comm if comm is not None else mpi_context.comm
    shape = residual_shape(value, dims)
    fusable = (
        scatter is None
        and resolved_comm.size > 1
        and error is None
        and partial_sum is not None
        and mpp_can_fuse_count(sum_dtype)
        and skipna_enabled(value.dtype, skipna)
    )

    if fusable:
        local_count, count_error = guarded(
            lambda: value.count(dim=dims, keep_attrs=False)
        )
        if count_error is None and local_count is not None:
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            # [ sum | count | health flag ], one contiguous SUM reduction.
            buffer = np.empty(2 * count + 1, dtype=sum_dtype)
            buffer[:count] = np.asarray(partial_sum.values, dtype=sum_dtype).reshape(
                count
            )
            buffer[count : 2 * count] = np.asarray(
                local_count.values, dtype=sum_dtype
            ).reshape(count)
            buffer[2 * count] = 0.0
            reduced = _mpp_reduce(buffer, MPI.SUM, resolved_comm)
            if np.real(reduced[2 * count]) == 0.0:
                total = partial_sum.copy(data=reduced[:count].reshape(shape))
                counted = local_count.copy(
                    data=np.asarray(reduced[count : 2 * count].reshape(shape)).astype(
                        np.int64
                    )
                )
                return (
                    _apply_replica_count(total, MPI.SUM, replica_count),
                    _apply_replica_count(counted, MPI.SUM, replica_count),
                )
        error = count_error

    global_sum = mpp_comm_reduce(
        mpi_context,
        partial_sum,
        MPI.SUM,
        expect_dtype=sum_dtype,
        error=error,
        phase=phase,
        comm=comm,
        replica_count=replica_count,
        scatter=scatter,
        expect_shape=shape,
    )
    return global_sum, None


def mpp_count_valid_values(
    mpi_context: MPIContext,
    value: xr.DataArray,
    dims: tuple[Hashable, ...],
    *,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, Sequence[int]] | None = None,
) -> xr.DataArray:
    """Count valid values globally across the requested dimensions."""
    count: xr.DataArray | None = None
    error: BaseException | None = None
    try:
        count = value.count(dim=dims, keep_attrs=False)
    except BaseException as exc:
        error = exc
    return mpp_comm_reduce(
        mpi_context,
        count,
        MPI.SUM,
        expect_dtype=partial_dtype(value.dtype.str, "count", None),
        error=error,
        phase="MPI xarray count reduction",
        comm=comm,
        replica_count=replica_count,
        scatter=scatter,
        expect_shape=residual_shape(value, dims),
    )


def dataset_result(
    value: xr.Dataset,
    dims: tuple[Hashable, ...],
    variables: Mapping[Hashable, xr.DataArray],
) -> xr.Dataset:
    """Rebuild a Dataset from reduced data variables."""
    reduced = set(dims)
    coords = {
        name: coord
        for name, coord in value.coords.items()
        if not reduced & set(coord.dims)
    }
    return xr.Dataset(dict(variables), coords=coords, attrs=dict(value.attrs))


def repartition_candidates(plan: tuple[PlanEntry, ...]) -> frozenset[Hashable]:
    """Return dimensions eligible for post-reduction repartition."""
    return frozenset(
        dim for entry in plan if entry.distributed for dim, _ in entry.shape
    )


def mpp_scatter_target(
    *,
    old_meta: Mapping[str, Any] | None,
    dims: tuple[Hashable, ...],
    partition_dim: Hashable | Literal["auto"] | None,
    auto_candidates: frozenset[Hashable],
    result_sizes: Mapping[Hashable, int],
    comm: MPI.Comm,
    replica_count: int,
) -> tuple[Hashable, list[int]] | None:
    """Choose a target for ``Reduce_scatter`` after a reduction.

    Parameters
    ----------
    old_meta : mapping or None
        Input distribution metadata.
    dims : tuple[Hashable, ...]
        Reduced dimensions.
    partition_dim : Hashable, {"auto"}, or None
        Requested output partition dimension.
    auto_candidates : frozenset[Hashable]
        Dimensions eligible for automatic partitioning.
    result_sizes : mapping[Hashable, int]
        Global result sizes.
    comm : mpi4py.MPI.Comm
        Candidate communicator.
    replica_count : int
        Number of replicated copies in the communicator.

    Returns
    -------
    tuple[Hashable, list[int]] or None
        Target dimension and per-rank counts, or None when scattering is unsuitable.
    """
    if (
        partition_dim != "auto"
        or old_meta is None
        or replica_count != 1
        or comm.size <= 1
        or any(dim not in dims for dim in old_meta["dims"])
    ):
        return None
    sizes = {
        dim: int(length)
        for dim, length in result_sizes.items()
        if dim in auto_candidates
    }
    if not any(length > 1 for length in sizes.values()):
        return None
    target = choose_partition_dim(sizes, comm.size, rank=comm.rank)
    length = sizes[target]
    chunk_size = get_effective_chunk_size(length, None, comm.size)
    counts = [
        stop - start
        for start, stop in (
            get_chunk_bounds(length, chunk_size, rank, comm.size)
            for rank in range(comm.size)
        )
    ]
    return target, counts


def mpp_finish_scatter(
    result: xr.Dataset | xr.DataArray,
    *,
    target: Hashable,
    counts: Sequence[int],
    comm: MPI.Comm,
) -> xr.Dataset | xr.DataArray:
    """Attach distribution metadata after ``Reduce_scatter``.

    Parameters
    ----------
    result : xarray.Dataset or xarray.DataArray
        Rank-local scattered result.
    target : Hashable
        Scattered dimension.
    counts : sequence of int
        Per-rank lengths along ``target``.
    comm : mpi4py.MPI.Comm
        Scatter communicator.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Result with updated MPI metadata.
    """
    rank = comm.rank
    start = sum(counts[:rank])
    stop = start + counts[rank]
    chunk_info = {
        str(dim): get_effective_chunk_size(int(length), None, comm.size)
        for dim, length in result.sizes.items()
    }
    mpp_update_meta(
        result,
        dim=target,
        global_size=sum(counts),
        start=start,
        stop=stop,
        chunk_info=chunk_info,
    )
    return result


def mpp_scatter_replicated_slice(
    variable: xr.DataArray, target: Hashable, start: int, stop: int
) -> xr.DataArray:
    """Slice a replicated variable to a scattered target range.

    Parameters
    ----------
    variable : xarray.DataArray
        Unreduced replicated variable.
    target : Hashable
        Scattered dimension.
    start, stop : int
        Rank-local half-open target bounds.

    Returns
    -------
    xarray.DataArray
        Matching local slice, or ``variable`` unchanged if ``target`` is absent.
    """
    return (
        variable.isel({target: slice(start, stop)})
        if target in variable.dims
        else variable
    )


def mpp_plan_scatter_target(
    mpi_context: MPIContext,
    old_meta: Mapping[str, Any] | None,
    dims: tuple[Hashable, ...],
    partition_dim: Hashable | Literal["auto"] | None,
    reduce_plan: tuple[PlanEntry, ...],
) -> tuple[Hashable, list[int], MPI.Comm] | None:
    """Choose one shared scatter target for a reduction plan.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    old_meta : mapping or None
        Input distribution metadata.
    dims : tuple[Hashable, ...]
        Reduced dimensions.
    partition_dim : Hashable, {"auto"}, or None
        Requested output partition dimension.
    reduce_plan : tuple[PlanEntry, ...]
        Per-variable reduction plan.

    Returns
    -------
    tuple[Hashable, list[int], mpi4py.MPI.Comm] or None
        Shared target, counts, and communicator when scatter is valid.
    """
    distributed_entries = [e for e in reduce_plan if e.dims and e.distributed]
    if not distributed_entries:
        return None
    combine_comms = {
        entry.name: mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        for entry in distributed_entries
    }
    if len({c.size for c in combine_comms.values()}) != 1:
        return None
    if any(entry.replica_count != 1 for entry in distributed_entries):
        return None
    comm = next(iter(combine_comms.values()))
    result_sizes: dict[Hashable, int] = {}
    for entry in reduce_plan:
        result_sizes.update(dict(entry.shape))
    target = mpp_scatter_target(
        old_meta=old_meta,
        dims=dims,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(reduce_plan),
        result_sizes=result_sizes,
        comm=comm,
        replica_count=1,
    )
    if target is None:
        return None
    return (*target, comm)


@dataclass(frozen=True)
class ReduceContext:
    """What a combine step needs beyond the variable being reduced.

    Attributes
    ----------
    dims : tuple of Hashable
        Dimensions this variable reduces over.
    entry : PlanEntry
        The variable's entry in the reduction plan.
    comm : mpi4py.MPI.Comm
        Communicator the reduction runs on.
    scatter : tuple[Hashable, list[int]] or None
        Target dimension and per-rank counts when the result is scattered.
    """

    dims: tuple[Hashable, ...]
    entry: PlanEntry
    comm: MPI.Comm
    scatter: tuple[Hashable, list[int]] | None


def mpp_global_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    operation: str,
    serial: Callable[[Any, Any], Any],
    combine: Callable[..., xr.DataArray],
    partition_dim: Hashable | Literal["auto"] | None,
    allow_scatter: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Drive a distributed reduction from plan to finished result.

    Every global reduction follows the same course: decide whether the
    reduction touches a partitioned dimension at all, plan it, reduce each
    variable, then repartition the result. Only the rank-local step and the
    cross-rank combination differ between sum, product, mean, extremum,
    logical and positional reductions, so those two arrive as callables and
    everything around them is shared.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None
        Dimensions to reduce.
    operation : str
        Reduction name, used for planning and diagnostics.
    serial : callable
        ``serial(obj, dims)`` reducing ``obj`` without communication. Used
        wherever no partitioned dimension is involved.
    combine : callable
        ``combine(variable, context)`` reducing one distributed variable,
        where ``context`` is a :class:`ReduceContext`.
    partition_dim : Hashable, {"auto"}, or None
        Where to repartition the result once the active partition dimension
        is reduced away.
    allow_scatter : bool, default True
        Whether the result may be produced already split across ranks by a
        ``Reduce_scatter``. Positional and logical reductions do not.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object with updated distribution metadata.
    """
    local_dim, dims = normalize_dim(value, dim)
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        return finish_local_reduction(serial(value, local_dim), old_meta=local_meta)

    plan = mpp_reduction_plan(mpi_context, value, dims, old_meta, operation=operation)
    scattered = (
        mpp_plan_scatter_target(mpi_context, old_meta, dims, partition_dim, plan)
        if allow_scatter
        else None
    )
    scatter = None if scattered is None else scattered[:2]

    def comm_for(entry: PlanEntry) -> MPI.Comm:
        """Return the communicator this entry reduces over."""
        if scattered is not None:
            return scattered[2]
        return mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)

    if isinstance(value, xr.DataArray):
        if not dims:
            return serial(value, local_dim)
        result = combine(
            value, ReduceContext(dims, plan[0], comm_for(plan[0]), scatter)
        )
    else:
        start = stop = None
        if scattered is not None:
            counts, scatter_comm = scattered[1], scattered[2]
            start = sum(counts[: scatter_comm.rank])
            stop = start + counts[scatter_comm.rank]
        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = (
                    variable
                    if scattered is None
                    else mpp_scatter_replicated_slice(
                        variable, scattered[0], start, stop
                    )
                )
            elif not entry.distributed:
                variables[entry.name] = serial(variable, entry.dims)
            else:
                variables[entry.name] = combine(
                    variable,
                    ReduceContext(entry.dims, entry, comm_for(entry), scatter),
                )
        source = (
            value.isel({scattered[0]: slice(start, stop)})
            if scattered is not None and scattered[0] in value.dims
            else value
        )
        result = dataset_result(source, dims, variables)

    if scattered is not None:
        return mpp_finish_scatter(
            result, target=scattered[0], counts=scattered[1], comm=scattered[2]
        )
    return mpp_finish(
        mpi_context,
        result,
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(plan),
    )


def mpp_finish(
    mpi_context: MPIContext,
    result: xr.Dataset | xr.DataArray,
    *,
    old_meta: Mapping[str, Any] | None,
    partition_dim: Hashable | Literal["auto"] | None,
    auto_candidates: frozenset[Hashable],
) -> xr.Dataset | xr.DataArray:
    """Finalize metadata and optional repartition after a reduction.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Finalized distributed result. Fully replicated (``.meta`` is
        None) if every previous partition dimension was reduced away.
        Otherwise, metadata is reattached for whichever dimension(s)
        survive, with no duplicated ownership: exactly one rank per
        distinct surviving range keeps the real result; every other
        rank that shared that range before the reduction is left with
        a genuinely empty (``start == stop``) slice instead of a
        redundant copy.

    """
    pruned_chunks = (
        prune_chunk_info(old_meta["chunk_info"], result) if old_meta is not None else {}
    )
    result = strip_mpi_meta(result)
    old_dims: tuple[Hashable, ...] = () if old_meta is None else old_meta["dims"]
    remaining_dims = tuple(dim for dim in old_dims if dim in result.dims)
    partition_removed = old_meta is not None and not remaining_dims

    if partition_dim is None:
        return result

    if partition_dim == "auto" and remaining_dims:
        # Reattach surviving partition axes directly and rebuild reduced Cartesian
        # topology lazily.
        assert old_meta is not None  # remaining_dims is empty otherwise
        cart = old_meta.get("cart") if len(remaining_dims) == len(old_dims) else None
        start = {dim: int(old_meta["starts"][dim]) for dim in remaining_dims}
        stop = {dim: int(old_meta["stops"][dim]) for dim in remaining_dims}

        reduced_dims = frozenset(old_dims) - frozenset(remaining_dims)
        if reduced_dims:
            # Keep one copy per surviving range and mark subgroup replicas empty to
            # preserve non-overlapping ownership.
            comm = mpp_resolve_comm(mpi_context, old_meta, reduced_dims)
            if comm.rank != 0:
                empty_dim = remaining_dims[0]
                result = result.isel({empty_dim: slice(0, 0)})
                stop[empty_dim] = start[empty_dim]

        mpp_update_meta(
            result,
            dim=remaining_dims,
            global_size={
                dim: int(old_meta["global_sizes"][dim]) for dim in remaining_dims
            },
            start=start,
            stop=stop,
            chunk_info=pruned_chunks,
            cart=cart,
        )
        return result

    target = partition_dim
    if partition_dim == "auto":
        if not partition_removed:
            return result
        sizes = {
            dim: length
            for dim, length in result.sizes.items()
            if dim in auto_candidates
        }
        if not any(int(length) > 1 for length in sizes.values()):
            return result
        target = choose_partition_dim(
            sizes, mpi_context.comm.size, rank=mpi_context.comm.rank
        )
    elif partition_dim not in auto_candidates:
        raise ValueError(
            f"partition_dim={partition_dim!r} was not reduced collectively."
        )

    chunk_info = pruned_chunks
    from .distribute import mpp_repartition

    return mpp_repartition(mpi_context, result, target, chunk_info=chunk_info)
