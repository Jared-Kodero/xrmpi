"""Cartesian process grids, extending :mod:`~xrmpi.mpp.mpp_domains`.

FMS reaches the ranks of a domain through pelists. Here the process grid is
an MPI Cartesian communicator instead, which gives the same neighbour and
sub-communicator queries directly from MPI.

Names here carry no ``mpp_`` prefix because they are not FMS routines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, cast

from ..mpi.mpi_init import MPI
from .mpp_domains import Domain, _no_proc_null
from .mpp_domains_define import mpp_compute_extent, mpp_define_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mpi4py.MPI import Cartcomm, Comm

    from ..mpi.context import MPIContext


@cache
def _topology_keyval() -> int:
    """Create the topology-cache keyval on first use, after ``MPI_Init``."""
    return MPI.Comm.Create_keyval()


def _define_layout_nd(extents: Sequence[int], ndivs: int) -> tuple[int, ...]:
    """Choose a process-grid shape for more than two partition dimensions.

    Assigns the prime factors of ``ndivs``, largest first, to whichever axis
    currently carries the most work per rank. Two-dimensional layouts use
    :func:`mpp_define_layout` instead, which follows FMS exactly.

    Parameters
    ----------
    extents : sequence of int
        Global length of each partitioned dimension.
    ndivs : int
        Number of ranks to divide among.

    Returns
    -------
    tuple of int
        Number of divisions along each axis.

    Raises
    ------
    ValueError
        If ``extents`` is empty, any extent is not positive, or ``ndivs`` is
        not positive.
    """
    if not extents:
        raise ValueError("requires at least one extent")
    if any(extent <= 0 for extent in extents):
        raise ValueError(f"All extents must be positive; got {tuple(extents)!r}.")
    if ndivs <= 0:
        raise ValueError(f"ndivs must be positive; got {ndivs}.")
    if len(extents) == 2:
        return mpp_define_layout(extents[0], extents[1], ndivs)

    factors: list[int] = []
    remaining, factor = ndivs, 2
    while factor * factor <= remaining:
        while remaining % factor == 0:
            factors.append(factor)
            remaining //= factor
        factor += 1
    if remaining > 1:
        factors.append(remaining)

    shape = [1] * len(extents)
    for f in sorted(factors, reverse=True):
        axis = max(range(len(extents)), key=lambda i: extents[i] / shape[i])
        shape[axis] *= f
    return tuple(shape)


@dataclass(frozen=True)
class CartesianDomain:
    """One rank's view of a multi-dimensional Cartesian process grid.

    Attributes
    ----------
    dims : tuple of str
        Partition dimension names, in Cartesian-axis order.
    grid_shape : tuple of int
        Number of process-grid divisions along each axis.
    coords : tuple of int
        This rank's position in the process grid, one entry per axis.
    cart_comm : mpi4py.MPI.Cartcomm
        The underlying Cartesian communicator. Rank order matches
        ``comm`` (``reorder=False``), so ``cart_comm.rank`` and the
        originating communicator's rank agree.
    bounds : dict of str to (int, int)
        Global half-open ``[start, stop)`` interval owned by this rank,
        per dimension.
    neighbors : dict of str to (int or None, int or None)
        Per-dimension ``(lower_rank, upper_rank)`` face neighbors in the
        *original* (non-Cartesian) communicator's rank numbering. None at
        a non-periodic global boundary.

    """

    dims: tuple[str, ...]
    grid_shape: tuple[int, ...]
    coords: tuple[int, ...]
    cart_comm: Cartcomm
    bounds: dict[str, tuple[int, int]]
    neighbors: dict[str, tuple[int | None, int | None]]
    _sub_comm_cache: dict[frozenset[str], Comm] = field(
        default_factory=dict, repr=False, compare=False
    )

    def as_meta_cart(self) -> dict[str, Any]:
        """Return the ``meta["cart"]`` descriptor for this topology.

        Returns
        -------
        dict[str, Any]
            Cartesian topology metadata descriptor.

        """
        return {
            "grid_shape": self.grid_shape,
            "coords": self.coords,
            "periods": (False,) * len(self.dims),
        }

    def sub_comm(self, merge_axes: Sequence[str]) -> Comm:
        """Return the communicator grouping ranks for a partial collective.

        Parameters
        ----------
        merge_axes : sequence of str
            Subset of :attr:`dims` to group ranks across.

        Returns
        -------
        mpi4py.MPI.Comm
            The (possibly cached) sub-communicator.

        """
        key = frozenset(merge_axes)
        cached = self._sub_comm_cache.get(key)
        if cached is not None:
            return cached
        remain = [dim in key for dim in self.dims]
        sub = self.cart_comm.Sub(remain)
        self._sub_comm_cache[key] = sub
        return sub


def define_cartesian_domain(
    comm: MPI.Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianDomain:
    """Build a rank's Cartesian topology for a multi-dimensional partition.

    Raises
    ------
    ValueError
        If fewer than two dimensions are given.

    """
    if len(dims) < 2:
        raise ValueError(
            "requires at least two partition dimensions; got " + f"{tuple(dims)!r}"
        )

    extents = [int(sizes[dim]) for dim in dims]
    grid_shape = _define_layout_nd(extents, comm.size)

    cart_comm = comm.Create_cart(
        dims=list(grid_shape),
        periods=[False] * len(dims),
        reorder=False,
    )
    coords = tuple(cart_comm.Get_coords(cart_comm.rank))

    bounds: dict[str, tuple[int, int]] = {}
    neighbors: dict[str, tuple[int | None, int | None]] = {}
    for axis, dim in enumerate(dims):
        bounds[dim] = mpp_compute_extent(extents[axis], coords[axis], grid_shape[axis])
        lower, upper = cart_comm.Shift(axis, 1)
        neighbors[dim] = (_no_proc_null(lower), _no_proc_null(upper))

    return CartesianDomain(
        dims=tuple(dims),
        grid_shape=grid_shape,
        coords=coords,
        cart_comm=cart_comm,
        bounds=bounds,
        neighbors=neighbors,
    )


def get_cartesian_domain(
    comm: MPI.Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianDomain:
    """Return (building and caching once) a rank's Cartesian topology."""
    dims = tuple(dims)
    # Include sizes in the cache key so same-named dimensions with different extents
    # cannot collide.
    cache_key = (dims, tuple(int(sizes[d]) for d in dims))
    cache = comm.Get_attr(_topology_keyval())
    if cache is None:
        cache = {}
        comm.Set_attr(_topology_keyval(), cache)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    topology = define_cartesian_domain(comm, dims, sizes)
    cache[cache_key] = topology
    return topology


def dim_comm(
    source: Domain | Mapping[str, Any],
    dim: str,
    mpi_context: MPIContext | None = None,
) -> Comm:
    """Return the communicator that varies only along ``dim``.

    Parameters
    ----------
    source : Domain or mapping
        A domain, or canonical MPI metadata describing one.
    dim : str
        Partition dimension.
    mpi_context : MPIContext, optional
        Required only when ``source`` is metadata, which carries no
        communicator of its own.

    Returns
    -------
    mpi4py.MPI.Comm
        The whole communicator for a one-axis partition, otherwise the
        Cartesian subcommunicator for ``dim``.
    """
    if isinstance(source, Domain):
        dims, sizes, comm = source.dims, source.global_sizes, source.comm
        cartesian = source.cart is not None
    else:
        if mpi_context is None:
            raise TypeError("dim_comm needs an MPIContext for metadata input.")
        dims, sizes = source["dims"], source["global_sizes"]
        comm = mpi_context.comm
        cartesian = "cart" in source

    if len(dims) <= 1 or not cartesian:
        return cast("Comm", comm)
    return get_cartesian_domain(comm, dims, sizes).sub_comm((dim,))


def slice_compute_domain(
    start: int,
    stop: int,
    requested_start: int,
    requested_stop: int,
) -> tuple[int, int, int]:
    """Intersect one compute domain with a global slice.

    Parameters
    ----------
    start, stop : int
        Rank-local global ownership bounds.
    requested_start, requested_stop : int
        Requested global half-open slice.

    Returns
    -------
    tuple[int, int, int]
        Local slice bounds and the surviving global start offset.
    """
    lower = max(requested_start, start)
    upper = max(lower, min(requested_stop, stop))
    below = max(0, min(requested_stop, start) - requested_start)
    return lower - start, upper - start, below
