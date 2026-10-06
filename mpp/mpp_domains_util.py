"""Query a domain and check fields laid out on it.

Mirrors FMS ``mpp/include/mpp_domains_util.inc``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .ext_domains import get_cartesian_domain
from .mpp_domains_define import mpp_compute_extent
from .mpp import mpp_chksum
from .mpp_domains import Domain, DomainMismatchError

if TYPE_CHECKING:
    pass


def mpp_get_compute_domain(domain: Domain, dim: str) -> tuple[int, int]:
    """Return the half-open bounds this rank owns along ``dim``.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple[int, int]
        ``(start, stop)`` of the compute domain.
    """
    return domain.starts[dim], domain.stops[dim]


def mpp_get_compute_domains(
    global_size: int,
    dim_size: int,
    *,
    min_partition_size: int | None = None,
) -> list[tuple[int, int]]:
    """Return balanced ownership bounds for every division of a dimension.

    Parameters
    ----------
    global_size : int
        Global dimension length.
    dim_size : int
        Number of divisions along the dimension.
    min_partition_size : int, optional
        Minimum non-empty local extent.

    Returns
    -------
    list[tuple[int, int]]
        Half-open bounds for each division.
    """

    return [
        mpp_compute_extent(int(global_size), rank, int(dim_size), min_partition_size)
        for rank in range(int(dim_size))
    ]


def mpp_get_data_domain(domain: Domain, dim: str) -> tuple[int, int]:
    """Return the bounds this rank holds along ``dim``, halo included.

    The data domain is the compute domain widened by the halo and clipped to
    the global domain, since an edge rank has no neighbour to receive from.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple[int, int]
        ``(start, stop)`` of the data domain.
    """
    before, after = domain.halo.get(dim, (0, 0))
    start, stop = domain.starts[dim], domain.stops[dim]
    return max(0, start - before), min(domain.global_sizes[dim], stop + after)


def mpp_get_global_domain(domain: Domain, dim: str) -> tuple[int, int]:
    """Return the global bounds of ``dim``.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple[int, int]
        ``(0, global_size)``.
    """
    return 0, domain.global_sizes[dim]


def mpp_get_layout(domain: Domain) -> tuple[int, ...]:
    """Return the process-grid shape the domain is divided over.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.

    Returns
    -------
    tuple of int
        Divisions along each partitioned dimension.
    """
    if domain.cart is not None:
        return tuple(int(n) for n in domain.cart["shape"])
    return (domain.comm.size,)


def mpp_get_pelist(domain: Domain) -> tuple[int, ...]:
    """Return the ranks the domain is distributed over.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.

    Returns
    -------
    tuple of int
        Ranks in the owning communicator.
    """
    return tuple(range(domain.comm.size))


def mpp_get_domain_extents(domain: Domain, dim: str) -> tuple[tuple[int, int], ...]:
    """Return every rank's compute-domain bounds along ``dim``.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple of tuple[int, int]
        ``(start, stop)`` per rank, in rank order.
    """
    return tuple(domain.comm.allgather((domain.starts[dim], domain.stops[dim])))


def mpp_get_domain_components(domain: Domain) -> dict[str, Domain]:
    """Split a multi-axis domain into one single-axis domain per dimension.

    FMS ``mpp_get_domain_components`` hands back the ``domain1D`` components
    of a ``domain2D``, which is how a routine that only works along one axis
    gets a domain it can use.

    Parameters
    ----------
    domain : Domain
        Domain to decompose.

    Returns
    -------
    dict[str, Domain]
        One domain per dimension, each carrying that axis alone and the
        communicator that varies along it.
    """
    return {
        dim: Domain(
            dims=(dim,),
            global_sizes={dim: domain.global_sizes[dim]},
            starts={dim: domain.starts[dim]},
            stops={dim: domain.stops[dim]},
            comm=domain.comm,
            halo={dim: domain.halo[dim]} if dim in domain.halo else {},
            cyclic={dim: domain.cyclic[dim]} if dim in domain.cyclic else {},
        )
        for dim in domain.dims
    }


def mpp_get_neighbor_pe(
    domain: Domain, dim: str, *, periodic: bool = False
) -> tuple[int | None, int | None]:
    """Return neighboring ranks along one partition dimension.

    Parameters
    ----------
    domain : Domain
        Rank-local domain descriptor.
    dim : str
        Partition dimension.
    periodic : bool, default False
        Wrap neighbors across global edges.

    Returns
    -------
    tuple[int or None, int or None]
        Lower- and upper-side neighbor ranks.
    """
    comm = domain.comm
    rank = comm.rank

    if len(domain.dims) > 1:
        topology = get_cartesian_domain(comm, domain.dims, domain.global_sizes)
        if periodic:
            axis = domain.dims.index(dim)
            axis_size = topology.grid_shape[axis]
            coords = list(topology.coords)
            coords[axis] = (topology.coords[axis] - 1) % axis_size
            left_rank = topology.cart_comm.Get_cart_rank(coords)
            coords[axis] = (topology.coords[axis] + 1) % axis_size
            right_rank = topology.cart_comm.Get_cart_rank(coords)
            return left_rank, right_rank
        return topology.neighbors[dim]

    size = comm.size
    if periodic:
        return (rank - 1) % size, (rank + 1) % size
    left_rank = rank - 1 if rank > 0 else None
    right_rank = rank + 1 if rank < size - 1 else None
    return left_rank, right_rank


def mpp_check_field(
    field: np.ndarray[Any, Any], domain: Domain, *, label: str = "field"
) -> None:
    """Verify every rank holds the same values where their domains overlap.

    FMS ``mpp_check_field`` is a debugging aid: it catches a halo update that
    silently failed to propagate, which otherwise shows up much later as a
    wrong answer.

    Parameters
    ----------
    field : numpy.ndarray
        Values to compare.
    domain : Domain
        Rank-local domain describing ``field``.
    label : str, default "field"
        Name used in the error message.

    Raises
    ------
    DomainMismatchError
        If any rank disagrees.
    """

    local = mpp_chksum(np.asarray(field))
    every = domain.comm.allgather(local)
    if len(set(every)) != 1:
        raise DomainMismatchError(
            f"{label} differs across ranks: {len(set(every))} distinct checksums."
        )
