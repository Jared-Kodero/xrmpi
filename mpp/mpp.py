"""Rank-level communication primitives.

Mirrors FMS ``mpp/mpp.F90``: collectives and communicator handling that
know nothing about how an array is decomposed. Anything that reasons about
a decomposition lives in :mod:`~xgeo.mpp.mpp_domains` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..mpi.mpi_init import MPI

#: Dtype kinds MPI has a datatype for.
MPI_REDUCIBLE_KINDS = "biufc"

if TYPE_CHECKING:
    from .mpp_domains import Domain


def _mpp_reduce(
    local: np.ndarray[Any, Any],
    op: MPI.Op,
    comm: MPI.Comm | None,
    domain: Domain | None = None,
) -> np.ndarray[Any, Any]:
    """Reduce rank-local arrays with an MPI reduction operator."""
    active_comm = (
        comm if comm is not None else (domain.comm if domain else MPI.COMM_WORLD)
    )
    recv = np.empty_like(local)
    active_comm.Allreduce(local, recv, op=op)
    return recv


def mpp_sum(
    local: np.ndarray[Any, Any],
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_sum``. Pass ``domain`` or ``comm``."""
    return _mpp_reduce(local, MPI.SUM, comm, domain)


def mpp_max(
    local: np.ndarray[Any, Any],
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_max``."""
    return _mpp_reduce(local, MPI.MAX, comm, domain)


def mpp_min(
    local: np.ndarray[Any, Any],
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_min``."""
    return _mpp_reduce(local, MPI.MIN, comm, domain)


def mpp_chksum(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm | None = None,
    *,
    mask_val: float | None = None,
) -> int:
    """Compute a rank-order-independent bitwise checksum.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local field.
    comm : mpi4py.MPI.Comm, optional
        Reduction communicator.
    mask_val : float, optional
        Sentinel excluded from the checksum.

    Returns
    -------
    int
        Global checksum.

    Raises
    ------
    TypeError
        If the element width is unsupported.
    """
    values = np.asarray(local)
    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    if mask_val is not None:
        keep = (
            ~np.isnan(values)
            if isinstance(mask_val, float) and np.isnan(mask_val)
            else values != mask_val
        )
        values = values[keep]

    # Reinterpret the bits and widen to int64 so the sum cannot overflow the
    # element type: FMS's TRANSFER, without a copy where NumPy allows it.
    width = values.dtype.itemsize
    if width not in (1, 2, 4, 8):
        raise TypeError(f"mpp_chksum: unsupported dtype {values.dtype}.")
    as_int = np.ascontiguousarray(values).view(f"i{width}")
    local_sum = np.int64(as_int.sum(dtype=np.int64))

    if comm is None or comm.size == 1:
        return int(local_sum)
    total = np.empty(1, dtype=np.int64)
    comm.Allreduce(np.array([local_sum], dtype=np.int64), total, op=MPI.SUM)
    return int(total[0])


def extreme_identity(dtype: np.dtype[Any], *, minimum: bool) -> Any:
    """Return the neutral value for a minimum or maximum reduction.

    A rank with nothing to contribute sends this instead, so it still enters
    the collective its peers are committed to rather than skipping it.

    Parameters
    ----------
    dtype : numpy.dtype
        Type being reduced.
    minimum : bool
        Whether the reduction is a minimum.

    Returns
    -------
    Any
        Value that leaves the reduction unchanged.

    Raises
    ------
    TypeError
        If the dtype has no defined ordering identity.
    """
    kind = dtype.kind
    if kind == "b":
        return bool(minimum)
    if kind in "iu":
        limits = np.iinfo(dtype)
        return limits.max if minimum else limits.min
    if kind == "f":
        return np.asarray(np.inf if minimum else -np.inf, dtype=dtype).item()
    name = "minimum" if minimum else "maximum"
    raise TypeError(f"MPI {name} is not defined for {dtype} data.")


def mpp_sync(comm: MPI.Comm) -> None:
    """Block until every rank in ``comm`` has arrived.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator to synchronise.
    """
    comm.Barrier()


def _buffered(value: Any) -> bool:
    """Return whether ``value`` can ride the MPI buffer interface.

    Only arrays of a type MPI has a datatype for qualify. A ``datetime64``
    array is still a NumPy array but has no MPI datatype, so it has to go the
    pickled route like any other Python object.
    """
    return isinstance(value, np.ndarray) and value.dtype.kind in MPI_REDUCIBLE_KINDS


def mpp_broadcast(
    value: np.ndarray[Any, Any], comm: MPI.Comm, *, root: int = 0
) -> np.ndarray[Any, Any]:
    """Send ``root``'s array to every rank.

    Parameters
    ----------
    value : numpy.ndarray or object
        Array to send on ``root``; a correctly shaped and typed buffer
        elsewhere. A non-array object is sent in pickled form instead, the
        way FMS overloads ``mpp_broadcast`` by type.
    comm : mpi4py.MPI.Comm
        Communicator to broadcast over.
    root : int, default 0
        Rank holding the source array.

    Returns
    -------
    numpy.ndarray
        The broadcast array, on every rank.
    """
    if not _buffered(value):
        # Anything MPI has no datatype for goes through the pickled form;
        # FMS overloads mpp_broadcast by type for the same reason.
        return comm.bcast(value, root=root)
    buffer = np.ascontiguousarray(value)
    comm.Bcast(buffer, root=root)
    return buffer


def mpp_gather(
    local: np.ndarray[Any, Any], comm: MPI.Comm, *, root: int | None = None
) -> np.ndarray[Any, Any] | None:
    """Collect equally sized rank-local arrays along a new leading axis.

    Parameters
    ----------
    local : numpy.ndarray
        This rank's contribution; must be the same shape on every rank. Use
        :func:`gather_v` when the contributions differ in size.
    comm : mpi4py.MPI.Comm
        Communicator to gather over.
    root : int, optional
        Rank receiving the result. None gathers to every rank.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(comm.size, *local.shape)``, or None on non-root
        ranks when ``root`` is given.
    """
    send = np.ascontiguousarray(local)
    if root is None:
        recv = np.empty((comm.size, *send.shape), dtype=send.dtype)
        comm.Allgather(send, recv)
        return recv
    recv = (
        np.empty((comm.size, *send.shape), dtype=send.dtype)
        if comm.rank == root
        else None
    )
    comm.Gather(send, recv, root=root)
    return recv


def mpp_scatter(
    value: np.ndarray[Any, Any] | None,
    comm: MPI.Comm,
    *,
    root: int = 0,
) -> np.ndarray[Any, Any]:
    """Deal out ``root``'s leading axis, one slice per rank.

    Parameters
    ----------
    value : numpy.ndarray or None
        Array of shape ``(comm.size, *rest)`` on ``root``, ignored elsewhere.
    comm : mpi4py.MPI.Comm
        Communicator to scatter over.
    root : int, default 0
        Rank holding the source array.

    Returns
    -------
    numpy.ndarray
        This rank's slice, of shape ``rest``.
    """
    shape, dtype = comm.bcast(
        (value.shape[1:], value.dtype) if comm.rank == root else None, root=root
    )
    recv = np.empty(shape, dtype=dtype)
    comm.Scatter(
        np.ascontiguousarray(value) if comm.rank == root else None, recv, root=root
    )
    return recv


def mpp_alltoall(send: np.ndarray[Any, Any], comm: MPI.Comm) -> np.ndarray[Any, Any]:
    """Exchange one equally sized slice with every rank.

    Parameters
    ----------
    send : numpy.ndarray
        Array of shape ``(comm.size, *rest)``; entry ``i`` goes to rank ``i``.
    comm : mpi4py.MPI.Comm
        Communicator to exchange over.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(comm.size, *rest)``; entry ``i`` came from rank
        ``i``.
    """
    buffer = np.ascontiguousarray(send)
    recv = np.empty_like(buffer)
    comm.Alltoall(buffer, recv)
    return recv
