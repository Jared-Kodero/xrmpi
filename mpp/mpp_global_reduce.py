"""Reduce a distributed field over its whole global domain.

Mirrors FMS ``mpp/include/mpp_global_reduce.fh`` and ``mpp_global_sum.fh``.
Halo points are excluded so a value shared by two ranks is counted once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .mpp import extreme_identity, mpp_max, mpp_min, mpp_sum
from .mpp_domains import Domain

if TYPE_CHECKING:
    from collections.abc import Sequence


def _compute_slice(
    field: np.ndarray[Any, Any], domain: Domain, dims: Sequence[str]
) -> np.ndarray[Any, Any]:
    """Trim halo points off ``field`` so only compute-domain values remain."""
    if not domain.halo:
        return field
    index: list[slice] = [slice(None)] * field.ndim
    for axis, dim in enumerate(dims):
        before, after = domain.halo.get(dim, (0, 0))
        if before or after:
            index[axis] = slice(before, field.shape[axis] - after or None)
    return field[tuple(index)]


def _local_extreme(
    owned: np.ndarray[Any, Any], *, minimum: bool
) -> np.ndarray[Any, Any]:
    """Return this rank's extremum, or the identity if it owns nothing."""
    if owned.size == 0:
        return np.asarray(
            extreme_identity(owned.dtype, minimum=minimum), dtype=owned.dtype
        )
    return np.asarray(owned.min() if minimum else owned.max())


def mpp_global_sum(
    field: np.ndarray[Any, Any],
    domain: Domain,
    dims: Sequence[str],
    *,
    bitwise_exact: bool = False,
) -> Any:
    """Sum a distributed field over its whole global domain.

    Halo points are excluded, so a value shared by two ranks is counted once.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's data-domain values.
    domain : Domain
        Rank-local domain describing ``field``.
    dims : sequence of str
        Dimension name of each axis of ``field``.
    bitwise_exact : bool, default False
        Sum in extended fixed point, giving a result independent of the rank
        count. Mirrors the FMS ``BITWISE_EXACT_SUM`` flag.

    Returns
    -------
    Any
        Global sum.
    """
    owned = _compute_slice(np.asarray(field), domain, dims)
    if bitwise_exact:
        from .mpp_efp import mpp_reproducing_sum

        return mpp_reproducing_sum(owned.reshape(-1), domain.comm)
    return mpp_sum(np.asarray(owned.sum(), dtype=np.float64), comm=domain.comm)


def mpp_global_max(
    field: np.ndarray[Any, Any], domain: Domain, dims: Sequence[str]
) -> Any:
    """Return the maximum of a distributed field over its global domain.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's data-domain values.
    domain : Domain
        Rank-local domain describing ``field``.
    dims : sequence of str
        Dimension name of each axis of ``field``.

    Returns
    -------
    Any
        Global maximum, over compute-domain points only.
        Ranks owning no points contribute the operator's identity rather
        than raising, so an empty compute domain cannot deadlock the
        collective.
    """
    owned = _compute_slice(np.asarray(field), domain, dims)
    return mpp_max(_local_extreme(owned, minimum=False), comm=domain.comm)


def mpp_global_min(
    field: np.ndarray[Any, Any], domain: Domain, dims: Sequence[str]
) -> Any:
    """Return the minimum of a distributed field over its global domain.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's data-domain values.
    domain : Domain
        Rank-local domain describing ``field``.
    dims : sequence of str
        Dimension name of each axis of ``field``.

    Returns
    -------
    Any
        Global minimum, over compute-domain points only.
        Ranks owning no points contribute the operator's identity rather
        than raising, so an empty compute domain cannot deadlock the
        collective.
    """
    owned = _compute_slice(np.asarray(field), domain, dims)
    return mpp_min(_local_extreme(owned, minimum=True), comm=domain.comm)
