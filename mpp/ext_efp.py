"""Reproducible products, extending :mod:`~xgeo.mpp.mpp_efp`.

FMS provides a reproducible *sum* in extended fixed point but no
product. This builds one on the same machinery: exponents sum exactly
as integers and mantissa logarithms go through the EFP sum, so the
result does not depend on the rank count.

Names here carry no ``mpp_`` prefix because they have no FMS
counterpart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..mpi.mpi_init import MPI
from .mpp_efp import _NUMINT, _from_digits, _to_digits

if TYPE_CHECKING:
    from collections.abc import Sequence


MAX_PROD_RANKS = 1000


_PROD_BLOCK = 512


_PROD_FIELDS = 5


PROD_EXPONENT, PROD_NAN, PROD_INF, PROD_ZERO, PROD_NEGATIVE = range(5)


def _moved_to_front(
    values: np.ndarray[Any, Any], axes: Sequence[int]
) -> np.ndarray[Any, Any]:
    """Collapse ``axes`` into a single leading axis, preserving the rest."""
    ordered = tuple(a % values.ndim for a in axes)
    moved = np.moveaxis(values, ordered, range(len(ordered)))
    kept = moved.shape[len(ordered) :]
    return moved.reshape((-1, *kept))


def prod_decompose(
    local: np.ndarray[Any, Any],
    axes: int | Sequence[int],
) -> np.ndarray[Any, Any]:
    """Decompose a rank-local product into exactly summable integer fields.

    A floating product depends on the order and grouping of its factors, so
    multiplying rank-local partials would make the result vary with the rank
    count. Every factor is instead split by ``frexp`` into a power of two and
    a mantissa in ``[0.5, 1)``; the exponents sum exactly as integers, and the
    mantissas are carried as ``log2`` values summed in extended fixed point.
    Both are order-free, so the product is reproducible.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    axes : int or sequence of int
        Local axes to reduce.

    Returns
    -------
    numpy.ndarray
        Integer fields: exponent, the NaN/infinity/zero/negative tallies, and
        the extended-fixed-point digits of the summed mantissa logarithms.
        Every field reduces with ``SUM``.
    """
    axis_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
    work = _moved_to_front(np.asarray(local).astype(np.float64, copy=False), axis_tuple)

    is_nan = np.isnan(work)
    is_inf = np.isinf(work)
    is_zero = work == 0.0
    ordinary = ~(is_nan | is_inf | is_zero)
    # Non-ordinary factors contribute 1.0, whose frexp is (0.5, 1): the
    # log2 of -1 and the exponent of +1 cancel, leaving the product untouched.
    magnitude = np.where(ordinary, np.abs(work), 1.0)

    mantissa, exponent = np.frexp(magnitude)
    fields = np.empty((_PROD_FIELDS + _NUMINT, *work.shape[1:]), dtype=np.int64)
    fields[PROD_EXPONENT] = exponent.sum(axis=0, dtype=np.int64)
    fields[PROD_NAN] = np.count_nonzero(is_nan, axis=0)
    fields[PROD_INF] = np.count_nonzero(is_inf, axis=0)
    fields[PROD_ZERO] = np.count_nonzero(is_zero, axis=0)
    fields[PROD_NEGATIVE] = np.count_nonzero(np.signbit(work) & ~is_nan, axis=0)
    fields[_PROD_FIELDS:] = _to_digits(np.log2(mantissa), 0)
    return fields


def prod_recombine(
    fields: np.ndarray[Any, Any],
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Rebuild a product from the reduced fields of :func:`prod_decompose`.

    Parameters
    ----------
    fields : numpy.ndarray
        Globally summed exponent, exception tallies and mantissa-log digits.
    dtype : numpy.dtype, optional
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Reconstructed product with signed zero/infinity and NaN handling.
    """
    n_nan = fields[PROD_NAN]
    n_inf = fields[PROD_INF]
    n_zero = fields[PROD_ZERO]
    sign = np.where(fields[PROD_NEGATIVE] % 2 == 1, -1.0, 1.0)

    # Split the summed logarithm into a whole power of two and a remainder in
    # [0, 1), so the whole part joins the exact integer exponent and only the
    # remainder goes through exp2.
    log_mantissa = _from_digits(fields[_PROD_FIELDS:])
    whole = np.floor(log_mantissa)
    mantissa = np.exp2(log_mantissa - whole)

    # ldexp takes a C int exponent; clipping is safe because anything beyond
    # this range has already saturated the float64 result either way.
    exponent = np.clip(
        fields[PROD_EXPONENT] + whole.astype(np.int64), -32768, 32768
    ).astype(np.int32)
    with np.errstate(over="ignore"):
        result = sign * np.ldexp(mantissa, exponent)

    result = np.where(n_zero > 0, sign * 0.0, result)
    result = np.where(n_inf > 0, sign * np.inf, result)
    result = np.where((n_zero > 0) & (n_inf > 0), np.nan, result)
    result = np.where(n_nan > 0, np.nan, result)
    return result if dtype is None else result.astype(dtype, copy=False)


def reproducing_prod(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm,
    *,
    axis: int | Sequence[int] = 0,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Compute a rank-count-invariant distributed product.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    axis : int or sequence of int, default 0
        Local reduction axes.
    dtype : numpy.dtype, optional
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Reproducible global product.

    Raises
    ------
    ValueError
        If the communicator exceeds the supported rank limit.
    """
    if comm.size > MAX_PROD_RANKS:
        raise ValueError(f"reproducing_prod supports at most {MAX_PROD_RANKS} ranks.")
    values = np.asarray(local)
    out_dtype = np.dtype(dtype) if dtype is not None else values.dtype
    fields = prod_decompose(values, axis)
    reduced = np.empty_like(fields)
    comm.Allreduce(np.ascontiguousarray(fields), reduced, op=MPI.SUM)
    return prod_recombine(reduced, out_dtype)
