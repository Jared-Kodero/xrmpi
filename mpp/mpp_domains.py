"""Domain types shared by the ``mpp_domains`` modules.

Mirrors FMS ``mpp/mpp_domains.F90``, which declares the domain types and
leaves the routines that act on them to the files it includes:
:mod:`~xrmpi.mpp.mpp_domains_define`,
:mod:`~xrmpi.mpp.mpp_domains_util`,
:mod:`~xrmpi.mpp.mpp_do_update`,
:mod:`~xrmpi.mpp.mpp_group_update`,
:mod:`~xrmpi.mpp.mpp_global_field` and
:mod:`~xrmpi.mpp.mpp_global_reduce`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..mpi.diagnostics import MPIError
from ..mpi.mpi_init import MPI


class DomainMismatchError(MPIError):
    """Raised when rank-local domains do not tile the global domain."""


@dataclass(frozen=True)
class Domain:
    """Describe one rank's distributed compute domain.

    Attributes
    ----------
    dims : tuple[str, ...]
        Partitioned dimension names.
    global_sizes : dict[str, int]
        Global size of each partitioned dimension.
    starts, stops : dict[str, int]
        Rank-local half-open ownership bounds.
    comm : mpi4py.MPI.Comm
        Communicator owning the global array.
    cart : dict or None
        Cartesian topology descriptor for multi-dimensional partitions.
    halo : dict[str, tuple[int, int]]
        Halo widths held before and after the compute domain on each
        dimension. Empty means the data domain equals the compute domain.
    cyclic : dict[str, bool]
        Whether each dimension wraps at the global edges, so the first and
        last ranks are neighbours. FMS carries this on the domain as
        ``CYCLIC_GLOBAL_DOMAIN`` rather than passing it per call.
    fold : int
        Folded edges, from :mod:`~xrmpi.mpp.mpp_parameter`. Zero means
        none. A fold joins an axis to itself in reverse, closing a tripolar
        grid across the pole.

    Notes
    -----
    FMS distinguishes the *compute* domain, which a rank owns and is
    responsible for updating, from the *data* domain, which additionally
    covers the halo points it holds copies of. ``starts``/``stops`` describe
    the compute domain; :func:`mpp_get_data_domain` applies ``halo``.
    """

    dims: tuple[str, ...]
    global_sizes: dict[str, int]
    starts: dict[str, int]
    stops: dict[str, int]
    comm: MPI.Comm
    cart: dict[str, Any] | None = field(default=None)
    halo: dict[str, tuple[int, int]] = field(default_factory=dict)
    cyclic: dict[str, bool] = field(default_factory=dict)
    fold: int = 0

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any], comm: MPI.Comm) -> Domain:
        """Build a domain from climtools MPI metadata.

        Parameters
        ----------
        meta : mapping
            Canonical MPI metadata.
        comm : mpi4py.MPI.Comm
            Owning communicator.

        Returns
        -------
        Domain
            Rank-local domain descriptor.
        """
        dims = tuple(str(d) for d in meta["dims"])
        return cls(
            dims=dims,
            global_sizes={d: int(meta["global_sizes"][d]) for d in dims},
            starts={d: int(meta["starts"][d]) for d in dims},
            stops={d: int(meta["stops"][d]) for d in dims},
            comm=comm,
            cart=dict(meta["cart"]) if meta.get("cart") is not None else None,
        )


def _no_proc_null(rank: int) -> int | None:
    """Map ``MPI.PROC_NULL`` (no neighbor) to None."""
    return None if rank == MPI.PROC_NULL else int(rank)
