"""Collectives with no FMS counterpart.

FMS covers the collectives a fixed decomposition needs. These cover what a
dynamic one needs as well: contributions whose sizes differ between ranks,
and rebuilding a decomposition after an operation changed how much each rank
holds.

Names here carry no ``mpp_`` prefix because they are not FMS routines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from collections.abc import Sequence


def reduce_scatter(
    local: np.ndarray[Any, Any],
    op: MPI.Op,
    comm: MPI.Comm,
    recvcounts: Sequence[int],
    *,
    axis: int = 0,
) -> np.ndarray[Any, Any]:
    """Reduce an array and retain each rank's contiguous slice.

    Parameters
    ----------
    local : numpy.ndarray
        Equal-shaped local reduction buffer on every rank.
    op : mpi4py.MPI.Op
        Reduction operator.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    recvcounts : sequence of int
        Elements retained by each rank along ``axis``.
    axis : int, default 0
        Axis split among ranks.

    Returns
    -------
    numpy.ndarray
        This rank's reduced slice.
    """
    moved = np.ascontiguousarray(np.moveaxis(local, axis, 0))
    per_slice = moved[0].size if moved.ndim > 1 else 1
    flat_counts = [c * per_slice for c in recvcounts]
    recvbuf = np.empty(flat_counts[comm.rank], dtype=moved.dtype)
    comm.Reduce_scatter(moved.reshape(-1), recvbuf, recvcounts=flat_counts, op=op)
    my_len = recvcounts[comm.rank]
    shape = (my_len, *moved.shape[1:]) if moved.ndim > 1 else (my_len,)
    return np.moveaxis(recvbuf.reshape(shape), 0, axis)


def partition_offsets(comm: MPI.Comm, local_length: int) -> tuple[int, int, int]:
    """Recompute distributed offsets after a local length change.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Partition communicator.
    local_length : int
        This rank's new local length.

    Returns
    -------
    tuple[int, int, int]
        Global size and this rank's half-open ownership bounds.
    """
    length = np.array([int(local_length)], dtype=np.int64)
    total = np.empty_like(length)
    comm.Allreduce(length, total, op=MPI.SUM)
    prefix = np.zeros_like(length)
    comm.Exscan(length, prefix, op=MPI.SUM)
    if comm.rank == 0:
        prefix[0] = 0  # Exscan leaves rank 0's receive buffer undefined.
    start = int(prefix[0])
    return int(total[0]), start, start + int(length[0])


def gather_v(
    local: Any, comm: MPI.Comm, *, root: int | None = None
) -> list[Any] | None:
    """Collect one contribution per rank when their sizes differ.

    The counterpart of :func:`mpp_gather` for ragged data, named after the
    ``v`` variants MPI uses for the same distinction. Contributions are sent
    in pickled form, so they may be arrays of differing length or arbitrary
    Python objects.

    Parameters
    ----------
    local : Any
        This rank's contribution.
    comm : mpi4py.MPI.Comm
        Communicator to gather over.
    root : int, optional
        Rank receiving the result. None gathers to every rank.

    Returns
    -------
    list or None
        One entry per rank in rank order, or None on non-root ranks when
        ``root`` is given.
    """
    if root is None:
        return comm.allgather(local)
    return comm.gather(local, root=root)


def scatter_v(value: list[Any] | None, comm: MPI.Comm, *, root: int = 0) -> Any:
    """Deal out one entry per rank when their sizes differ.

    The counterpart of :func:`mpp_scatter` for ragged data, named after the
    ``v`` variants MPI uses for the same distinction.

    Parameters
    ----------
    value : list or None
        One entry per rank on ``root``, ignored elsewhere.
    comm : mpi4py.MPI.Comm
        Communicator to scatter over.
    root : int, default 0
        Rank holding the entries.

    Returns
    -------
    Any
        This rank's entry.
    """
    return comm.scatter(value, root=root)
