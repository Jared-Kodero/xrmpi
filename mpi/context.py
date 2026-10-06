"""MPI communication utilities built on :mod:`mpi4py`.

Provides :class:`MPIContext` for point-to-point, collective, hierarchical,
and decorator-based MPI execution. The :data:`mpi` singleton uses
``MPI.COMM_WORLD`` by default.
"""

# context.py
from __future__ import annotations

import atexit
import functools
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from ...xgeo.core.climtools import LockFile, tmp
from .diagnostics import MPIDiagnostics, MPIError, get_tmpdir, tmp_cleanup
from .mpi_init import MPI, require_mpi, world_size

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class ToChildrenContext:
    """Parent-to-child communication operations.

    Provides collective operations from a parent communicator to logical child
    communicators created by :meth:`MPIContext.decompose`.

    Parameters
    ----------
    mpi_context : MPIContext
        Parent MPI context containing the child communicators.
    """

    def __init__(self, mpi_context: MPIContext) -> None:
        """Initialize parent-to-child communication operations."""
        self._runtime = mpi_context

    def broadcast(self, value: T | None, *, root: int = 0) -> T:
        """Broadcast a parent value to all child communicators.

        Parameters
        ----------
        value : T or None
            Value provided on ``root``. Non-root ranks may pass None.
        root : int, optional
            Source rank in the parent communicator.

        Returns
        -------
        T
            Broadcast value on every rank.

        Raises
        ------
        RuntimeError
            If the communicator has not been decomposed.
        TypeError
            If ``root`` is not an integer rank.
        ValueError
            If ``root`` is outside the parent communicator.
        MPIError
            If ranks disagree on the collective operation.
        """
        mpi_context = self._runtime
        _ = mpi_context.child
        comm = mpi_context.comm

        error: BaseException | None = None
        signature: tuple[str, int] | None = None
        try:
            if isinstance(root, bool) or not isinstance(root, Integral):
                raise TypeError("root must be a non-negative integer rank.")
            if root < 0 or root >= comm.size:
                raise ValueError(f"root {root} is outside [0, {comm.size}).")
            signature = ("to_children.broadcast", int(root))
        except BaseException as exc:
            error = exc

        mpi_context.raise_if_error(
            error,
            "mpi.to_children.broadcast",
            signature=signature,
        )
        return cast(
            "T",
            comm.bcast(value if mpi_context.is_root(root) else None, root=root),
        )

    def scatter(self, values: Sequence[T] | None, *, root: int = 0) -> T:
        """Scatter one value to each child communicator.

        Parameters
        ----------
        values : sequence of T or None
            One value per child on ``root``. Non-root ranks may pass None.
        root : int, optional
            Source rank in the parent communicator.

        Returns
        -------
        T
            Value assigned to this rank's child communicator.

        Raises
        ------
        RuntimeError
            If the communicator has not been decomposed.
        TypeError
            If ``root`` is not an integer rank.
        ValueError
            If ``root`` or ``values`` is invalid.
        MPIError
            If ranks disagree on the collective operation.
        """
        mpi_context = self._runtime
        child = mpi_context.child
        comm = mpi_context.comm
        ranks_per_child = child.comm.size
        ntasks = comm.size // ranks_per_child

        error: BaseException | None = None
        signature: tuple[str, int, int] | None = None
        send: list[T | None] | None = None

        try:
            if isinstance(root, bool) or not isinstance(root, Integral):
                raise TypeError("root must be a non-negative integer rank.")
            if root < 0 or root >= comm.size:
                raise ValueError(f"root {root} is outside [0, {comm.size}).")

            if mpi_context.is_root(root):
                if values is None:
                    raise ValueError("values cannot be None on the parent root.")
                if len(values) != ntasks:
                    raise ValueError(
                        f"values must contain exactly {ntasks} items, one per "
                        + f"child; got {len(values)}."
                    )

                send = [None] * comm.size
                for task, value in enumerate(values):
                    send[task * ranks_per_child] = value

            signature = ("to_children.scatter", int(root), ntasks)
        except BaseException as exc:
            error = exc

        mpi_context.raise_if_error(
            error,
            "mpi.to_children.scatter",
            signature=signature,
        )

        received = comm.scatter(send, root=root)
        return cast(
            "T",
            child.comm.bcast(
                received if child.is_root() else None,
                root=0,
            ),
        )


class FromChildrenContext:
    """Child-to-parent communication operations.

    Provides collective operations from logical child communicators created by
    :meth:`MPIContext.decompose` back to their parent communicator.

    Parameters
    ----------
    mpi_context : MPIContext
        Parent MPI context containing the child communicators.
    """

    def __init__(self, mpi_context: MPIContext) -> None:
        """Initialize child-to-parent communication operations."""
        self._runtime = mpi_context

    def gather(self, value: T, *, root: int = 0) -> list[T] | None:
        """Gather one value from each child communicator.

        Only each child leader contributes; results are ordered by child index.

        Parameters
        ----------
        value : T
            Child result. Only the child leader's value is collected.
        root : int, optional
            Destination rank in the parent communicator.

        Returns
        -------
        list of T or None
            Child-ordered values on ``root``; otherwise None.

        Raises
        ------
        RuntimeError
            If the communicator has not been decomposed.
        TypeError
            If ``root`` is not an integer rank.
        ValueError
            If ``root`` is outside the parent communicator.
        MPIError
            If ranks disagree on the collective operation.
        """
        mpi_context = self._runtime
        child = mpi_context.child
        comm = mpi_context.comm

        error: BaseException | None = None
        signature: tuple[str, int] | None = None
        try:
            if isinstance(root, bool) or not isinstance(root, Integral):
                raise TypeError("root must be a non-negative integer rank.")
            if root < 0 or root >= comm.size:
                raise ValueError(f"root {root} is outside [0, {comm.size}).")
            signature = ("from_children.gather", int(root))
        except BaseException as exc:
            error = exc

        mpi_context.raise_if_error(
            error, "mpi.from_children.gather", signature=signature
        )

        payload = (cast("int", child.task), value) if child.is_root() else None
        gathered = comm.gather(payload, root=root)

        if not mpi_context.is_root(root):
            return None

        results = [item for item in gathered if item is not None]
        results.sort(key=lambda item: item[0])
        return [item[1] for item in results]


class MPIContext(MPIDiagnostics):
    """MPI communication context.

    Wraps an MPI intracommunicator and provides point-to-point and collective
    communication, communicator decomposition, diagnostics, and MPI-aware
    function execution. The communicator is exposed through :attr:`comm`.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator. Defaults to ``MPI.COMM_WORLD``.
    """

    MPIError = MPIError

    # -------------------------------------------------------------------------
    # Runtime state and accessors
    # -------------------------------------------------------------------------

    def __init__(self, comm: MPI.Intracomm | None = None) -> None:
        """Initialize an MPI communication context."""
        # Before anything else: an MPI call made without MPI_Init does not
        # raise, it aborts the process, so the check cannot wrap the call.
        require_mpi()
        self.comm: MPI.Intracomm = comm if comm is not None else MPI.COMM_WORLD
        self._child: MPIContext | None = None
        self._to_children = ToChildrenContext(self)
        self._from_children = FromChildrenContext(self)
        self.info: tuple[int, ...] = ()
        self.task: int | None = None

        # get_tmpdir broadcasts, so it is a collective and must not run for a
        # process that is not part of an MPI job; constructing a context in a
        # notebook would otherwise post a bcast and create a directory under
        # $SCRATCH for a single kernel. Non-MPI use falls back to the plain
        # per-process scratch directory core.utils already made.
        if self.alive(self.comm):
            self._tmp: Path = get_tmpdir(self.comm)
            atexit.register(partial(tmp_cleanup, self.comm, self._tmp))
            self._install_abort_hook()
        else:
            self._tmp = tmp

        self._mpi_lock = LockFile(self._tmp / ".mpi.lock")

    @property
    def to_children(self) -> ToChildrenContext:
        """Parent-to-child communication operations."""
        return self._to_children

    @property
    def from_children(self) -> FromChildrenContext:
        """Child-to-parent communication operations."""
        return self._from_children

    @property
    def child(self) -> MPIContext:
        """Child MPI context for this rank.

        Returns
        -------
        MPIContext
            Context associated with this rank's child communicator.

        Raises
        ------
        RuntimeError
            If the communicator has not been decomposed.
        """
        if self._child is None:
            raise RuntimeError("MPI context has not been decomposed.")

        return self._child

    @property
    def launched(self) -> bool:
        """Whether this process appears to have been launched under MPI."""
        return self.alive(self.comm)

    @staticmethod
    def alive(comm: MPI.Comm) -> bool:
        """Check whether a communicator appears to be running under MPI.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Communicator to inspect.

        Returns
        -------
        bool
            Whether MPI execution is detected.
        """
        if comm.Get_size() > 1 or world_size() > 1:
            return True
        try:
            return MPI.Comm.Get_parent() != MPI.COMM_NULL
        except AttributeError, RuntimeError:
            return False

    def is_root(self, root: int = 0) -> bool:
        """Check whether this process is the specified root rank.

        Parameters
        ----------
        root : int, optional
            Root rank.

        Returns
        -------
        bool
            Whether the current rank equals ``root``.
        """
        return self.comm.rank == root

    # -------------------------------------------------------------------------
    # Child communicator decomposition
    # -------------------------------------------------------------------------

    def decompose(self, ntasks: int) -> None:
        """Split the communicator into equally sized child communicators.

        Each rank receives one context exposed through :attr:`child`. All ranks must
        call this collective operation with the same ``ntasks``.

        Parameters
        ----------
        ntasks : int
            Number of child communicators.

        Raises
        ------
        TypeError
            If ``ntasks`` is not an integer.
        ValueError
            If ``ntasks`` is invalid for the communicator size.
        MPIError
            If ranks provide inconsistent arguments.
        """
        comm = self.comm
        error: BaseException | None = None
        signature: tuple[str, int] | None = None

        try:
            if isinstance(ntasks, bool) or not isinstance(ntasks, Integral):
                raise TypeError("ntasks must be an integer.")
            if ntasks < 1:
                raise ValueError("ntasks must be at least 1.")
            if ntasks > comm.size:
                raise ValueError(
                    f"ntasks ({ntasks}) cannot exceed MPI size ({comm.size})."
                )
            if comm.size % ntasks != 0:
                raise ValueError(
                    f"MPI size ({comm.size}) must be divisible by ntasks ({ntasks})."
                )
            signature = ("decompose", int(ntasks))
        except BaseException as exc:
            error = exc

        self.raise_if_error(error, "mpi.decompose", signature=signature)

        ranks_per_task = comm.size // ntasks
        task = comm.rank // ranks_per_task
        start = task * ranks_per_task
        stop = start + ranks_per_task

        child_comm = comm.Split(color=task, key=comm.rank)
        child = MPIContext(child_comm)
        child.task = task
        child.info = tuple(range(start, stop))
        self._child = child

    # -------------------------------------------------------------------------
    # Point-to-point and collective object communication
    # -------------------------------------------------------------------------

    def send(self, value: T, dest: int, *, tag: int = 0) -> None:
        """Send a Python object to a rank.

        Parameters
        ----------
        value : T
            Object to send.
        dest : int
            Destination rank.
        tag : int, optional
            MPI message tag.
        """
        self.comm.send(value, dest=dest, tag=tag)

    def send_all(self, pieces: Mapping[int, Any], *, tag: int = 0) -> None:
        """Send one object to each of several ranks, then wait once.

        ``send`` blocks until its message is taken, so scattering P pieces
        with it serialises P handshakes on the sending rank and gets slower
        as the job grows. FMS's ``mpp_do_redistribute`` avoids the same
        pattern by posting every transfer before waiting on any of them;
        this does the same with ``isend``, leaving one wait for the whole
        set instead of one per destination.
        """
        requests = [
            self.comm.isend(piece, dest=dest, tag=tag)
            for dest, piece in sorted(pieces.items())
        ]
        MPI.Request.Waitall(requests)

    def receive(
        self,
        source: int | None = None,
        *,
        tag: int | None = None,
    ) -> T:
        """Receive a Python object from a rank.

        Parameters
        ----------
        source : int, optional
            Source rank. Defaults to ``MPI.ANY_SOURCE``.
        tag : int, optional
            MPI message tag. Defaults to ``MPI.ANY_TAG``.

        Returns
        -------
        T
            Received object.
        """
        source = MPI.ANY_SOURCE if source is None else source
        tag = MPI.ANY_TAG if tag is None else tag
        return cast("T", self.comm.recv(source=source, tag=tag))

    def broadcast(self, value: T | None, *, root: int = 0) -> T:
        """Broadcast a Python object from one rank.

        Parameters
        ----------
        value : T or None
            Object provided on ``root``. Non-root ranks may pass None.
        root : int, optional
            Source rank.

        Returns
        -------
        T
            Broadcast object.
        """
        return cast("T", self.comm.bcast(value, root=root))

    def scatter(self, values: Sequence[T] | None, *, root: int = 0) -> T:
        """Scatter Python objects across ranks.

        Parameters
        ----------
        values : sequence of T or None
            Rank-ordered objects on ``root``. Non-root ranks may pass None.
        root : int, optional
            Source rank.

        Returns
        -------
        T
            Object assigned to this rank.
        """
        return cast("T", self.comm.scatter(values, root=root))

    def scatterv(
        self,
        send: np.ndarray | None,
        counts: Sequence[int] | np.ndarray,
        local_shape: Sequence[int],
        dtype: npt.DTypeLike,
        *,
        root: int = 0,
    ) -> np.ndarray:
        """Scatter variable-sized array slabs across ranks.

        Parameters
        ----------
        send : numpy.ndarray or None
            Source array on ``root``, partitioned contiguously along axis 0.
        counts : sequence of int or numpy.ndarray
            Rank-wise axis-0 counts, identical on all ranks.
        local_shape : sequence of int
            Output shape for this rank; the leading dimension must equal its count.
        dtype : numpy.dtype-like
            Element dtype, identical on all ranks.
        root : int, optional
            Source rank.

        Returns
        -------
        numpy.ndarray
            Contiguous array assigned to this rank.

        Raises
        ------
        ValueError
            If ``send`` is missing on ``root``.
        """
        import numpy as np
        from mpi4py.util import dtlib

        dtype = np.dtype(dtype)
        counts = np.asarray(counts, dtype=np.int64)
        row_elems = (
            int(np.prod(local_shape[1:], dtype=np.int64)) if len(local_shape) > 1 else 1
        )
        elem_counts = counts * row_elems
        displs = np.zeros(len(counts), dtype=np.int64)
        np.cumsum(elem_counts[:-1], out=displs[1:])

        mpi_dtype = dtlib.from_numpy_dtype(dtype)
        local = np.empty(local_shape, dtype=dtype)

        sendbuf = None
        if self.comm.rank == root:
            if send is None:
                raise ValueError("`send` is required on the scattering root rank.")
            contiguous = np.ascontiguousarray(send, dtype=dtype)
            sendbuf = [contiguous, (elem_counts, displs), mpi_dtype]

        self.comm.Scatterv(sendbuf, [local, mpi_dtype], root=root)
        return local

    def gather(self, value: T, *, root: int = 0) -> list[T] | None:
        """Gather Python objects onto one rank.

        Parameters
        ----------
        value : T
            Object contributed by this rank.
        root : int, optional
            Destination rank.

        Returns
        -------
        list of T or None
            Rank-ordered objects on ``root``; otherwise None.
        """
        return cast("list[T] | None", self.comm.gather(value, root=root))

    # -------------------------------------------------------------------------
    # Decorator interface
    # -------------------------------------------------------------------------

    def __call__(
        self,
        function: Callable[P, R] | None = None,
        /,
        *,
        all_ranks: bool = False,
        broadcast: bool = False,
        root: int = 0,
    ) -> (
        Callable[P, R]
        | Callable[P, R | None]
        | Callable[[Callable[P, R]], Callable[P, R]]
        | Callable[[Callable[P, R]], Callable[P, R | None]]
    ):
        """Decorate a function for MPI-aware execution.

        By default, the function runs only on ``root``.

        Parameters
        ----------
        function : callable, optional
            Function to decorate.
        all_ranks : bool, optional
            Execute the function on every rank.
        broadcast : bool, optional
            Broadcast the root rank's result to every rank.
        root : int, optional
            Root rank.

        Returns
        -------
        callable
            Decorated function or configured decorator. In root-only mode, non-root
            ranks return None.

        Raises
        ------
        TypeError
            If ``function`` is not callable.
        ValueError
            If the execution options or root rank are invalid.
        MPIError
            If execution fails inconsistently across ranks.

        Examples
        --------
        >>> @mpi
        ... def compute():
        ...     ...
        >>> @mpi(all_ranks=True)
        ... def compute_local():
        ...     ...
        >>> @mpi(broadcast=True)
        ... def load():
        ...     ...
        """
        if function is None:
            return functools.partial(
                self, all_ranks=all_ranks, broadcast=broadcast, root=root
            )
        if not callable(function):
            raise TypeError("mpi's positional argument must be callable.")
        if broadcast and all_ranks:
            raise ValueError("broadcast=True is incompatible with all_ranks=True.")
        if isinstance(root, bool) or not isinstance(root, Integral) or root < 0:
            raise ValueError("root must be a non-negative integer rank.")

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            """Run the wrapped callable on the root rank only."""
            if root >= self.comm.size:
                raise ValueError(f"root {root} is outside [0, {self.comm.size}).")

            result: R | None = None
            error: BaseException | None = None
            if all_ranks or self.is_root(root):
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    error = exc

            self.raise_if_error(error, function.__name__)
            if broadcast:
                return cast("R", self.comm.bcast(result, root=root))
            return result

        wrapper.mpi = True  # type: ignore[attr-defined]
        return wrapper


def get_mpi_ctx() -> MPIContext:
    """Return the process-wide MPI context.

    Returns
    -------
    MPIContext
        Shared context; constructing it repeatedly is cheap.
    """
    return MPIContext()


__all__ = ["MPIContext", "get_mpi_ctx"]
