"""Define how an array is divided across ranks.

Mirrors FMS ``mpp/include/mpp_domains_define.inc``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from .mpp_domains import Domain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..mpi.context import MPIContext


def mpp_compute_extent(
    length: int, rank: int, size: int, min_chunk: int | None = None
) -> tuple[int, int]:
    """Split ``length`` into ``size`` contiguous, near-equal ``[start, stop)`` slabs.

    FMS ``mpp_compute_extent``: the routine every domain definition goes
    through to decide which indices a rank owns.

    Parameters
    ----------
    length : int
        Total length to split.
    rank : int
        Current MPI rank.
    size : int
        Total number of MPI ranks.
    min_chunk : int or None, optional
        Guaranteed minimum local length for every rank that receives any
        data. At most ``max(1, length // min_chunk)`` ranks get a
        non-empty slab; the rest get an empty ``(length, length)`` slab.
        Set at or above the widest halo/window a distributed dimension
        will need (e.g. the largest ``rolling_reduce`` window) to avoid
        ``mpp_halo_exchange``'s "local partition shorter than the
        requested halo" error on that dimension.

    Returns
    -------
    tuple of int
        Start and stop indices for the given rank.

    """
    if min_chunk is not None and min_chunk > 0 and size > 1 and length > 0:
        active = max(1, min(size, length // min_chunk))
        if active < size:
            if rank >= active:
                return length, length
            return mpp_compute_extent(length, rank, active)

    quotient, remainder = divmod(length, size)
    start = rank * quotient + min(rank, remainder)
    return start, start + quotient + int(rank < remainder)


def mpp_define_layout(extent0: int, extent1: int, ndivs: int) -> tuple[int, int]:
    """Choose a two-dimensional process-grid layout.

    Parameters
    ----------
    extent0, extent1 : int
        Global grid extents.
    ndivs : int
        Number of MPI ranks.

    Returns
    -------
    tuple[int, int]
        Process-grid shape minimizing idle ranks, then halo perimeter.

    Raises
    ------
    ValueError
        If ``ndivs`` is not positive.
    """
    if ndivs < 1:
        raise ValueError(f"ndivs must be positive, got {ndivs}.")

    pairs = [(rows, ndivs // rows) for rows in range(1, ndivs + 1) if ndivs % rows == 0]

    def cost(layout: tuple[int, int]) -> tuple[int, float, int]:
        """Return idle-rank, halo-perimeter, and aspect-ratio costs."""
        rows, cols = layout
        # Ranks left with nothing dominate; then the halo perimeter of one
        # subdomain; then squareness, purely to make ties deterministic.
        idle = max(0, rows - extent0) * cols + max(0, cols - extent1) * rows
        perimeter = extent0 / rows + extent1 / cols
        return idle, perimeter, abs(rows - cols)

    return min(pairs, key=cost)


def mpp_define_domains(
    mpi_context: MPIContext,
    global_sizes: Mapping[str, int],
    dims: str | Sequence[str],
    *,
    min_partition_size: int | Mapping[str, int] | None = None,
    rank: int | None = None,
) -> Domain:
    """Define balanced rank-local compute domains.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    global_sizes : mapping[str, int]
        Global sizes of partitioned dimensions.
    dims : str or sequence of str
        Partition dimensions.
    min_partition_size : int or mapping, optional
        Minimum non-empty local extent.
    rank : int, optional
        Rank whose domain to compute; defaults to the caller.

    Returns
    -------
    Domain
        Rank-local domain descriptor.
    """

    comm = mpi_context.comm
    target_rank = comm.rank if rank is None else rank
    dim_tuple = (dims,) if isinstance(dims, str) else tuple(dims)

    def _min_chunk(d: str) -> int | None:
        """Return the minimum partition size requested for one dimension."""
        return (
            min_partition_size
            if not isinstance(min_partition_size, Mapping)
            else min_partition_size.get(d)
        )

    if len(dim_tuple) == 1:
        dim = dim_tuple[0]
        length = int(global_sizes[dim])
        start, stop = mpp_compute_extent(
            length, target_rank, comm.size, _min_chunk(dim)
        )
        return Domain(
            dims=dim_tuple,
            global_sizes={dim: length},
            starts={dim: start},
            stops={dim: stop},
            comm=comm,
        )

    sizes = {d: int(global_sizes[d]) for d in dim_tuple}

    if target_rank == comm.rank:
        # Imported here: ext_domains builds on this module, so importing it
        # at module scope would close a cycle.
        from .ext_domains import get_cartesian_domain

        topology = get_cartesian_domain(comm, dim_tuple, sizes)
        grid_shape = topology.grid_shape
        starts = {d: topology.bounds[d][0] for d in dim_tuple}
        stops = {d: topology.bounds[d][1] for d in dim_tuple}
        cart = topology.as_meta_cart()
    else:
        grid_shape = mpp_define_layout(
            sizes[dim_tuple[0]], sizes[dim_tuple[1]], comm.size
        )
        coords = tuple(int(c) for c in np.unravel_index(target_rank, grid_shape))
        starts, stops = {}, {}
        for axis, d in enumerate(dim_tuple):
            s, e = mpp_compute_extent(
                sizes[d], coords[axis], grid_shape[axis], _min_chunk(d)
            )
            starts[d], stops[d] = s, e
        cart = {
            "grid_shape": grid_shape,
            "coords": coords,
            "periods": (False,) * len(dim_tuple),
        }

    return Domain(
        dims=dim_tuple,
        global_sizes=sizes,
        starts=starts,
        stops=stops,
        comm=comm,
        cart=cart,
    )
