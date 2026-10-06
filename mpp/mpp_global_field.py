"""Assemble a distributed field into its global form.

Mirrors FMS ``mpp/include/mpp_global_field.fh``: every rank contributes the
slice it owns and the result is the whole field, either on one rank or on all
of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mpp_domains import Domain, DomainMismatchError


def mpp_global_field(
    field: np.ndarray[Any, Any],
    domain: Domain,
    dim: str,
    axis: int,
    *,
    root: int | None = None,
) -> np.ndarray[Any, Any] | None:
    """Gather the compute domains along ``dim`` into the global field.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's compute-domain values.
    domain : Domain
        Domain describing ``field``.
    dim : str
        Partitioned dimension to reassemble.
    axis : int
        Array axis corresponding to ``dim``.
    root : int, optional
        Rank receiving the result. None gathers to every rank, which is what
        FMS does by default.

    Returns
    -------
    numpy.ndarray or None
        The global field, or None on non-root ranks when ``root`` is given.

    Raises
    ------
    DomainMismatchError
        If the gathered slices do not tile the axis exactly.
    """
    comm = domain.comm
    piece = (domain.starts[dim], domain.stops[dim], np.asarray(field))
    pieces = comm.allgather(piece) if root is None else comm.gather(piece, root=root)
    if pieces is None:
        return None

    cursor = 0
    ordered = sorted(pieces, key=lambda item: item[0])
    for start, stop, values in ordered:
        if start != cursor:
            raise DomainMismatchError(
                f"{dim!r}: expected a slice starting at {cursor}, got {start}."
            )
        if values.shape[axis] != stop - start:
            raise DomainMismatchError(
                f"{dim!r}: slice length {values.shape[axis]} != {stop - start}."
            )
        cursor = stop
    if cursor != domain.global_sizes[dim]:
        raise DomainMismatchError(
            f"{dim!r}: slices cover {cursor} of {domain.global_sizes[dim]} points."
        )

    return np.concatenate([values for _, _, values in ordered], axis=axis)
