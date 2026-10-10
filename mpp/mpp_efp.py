"""Reproducible reductions in extended fixed point.

Mirrors FMS ``mpp/mpp_efp.F90``. A floating sum depends on the order the
terms arrive in, so a distributed sum would change with the rank count.
Values are accumulated as exact integer digits instead, making the result
independent of how the data was decomposed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mpi.mpi_init import MPI

_NUMBIT = 46


_NUMINT = 6


_PREC = float(2**_NUMBIT)


MAX_EFP_RANKS = 2 ** (63 - _NUMBIT) - 1


_SCALES = np.array([_PREC ** (2 - n) for n in range(_NUMINT)], dtype=np.float64)


_PREC_INT = 1 << _NUMBIT


_EFP_BLOCK = 1 << 16


def _carry_overflow(digits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Renormalise digits so each holds less than one unit of the next scale.

    FMS ``carry_overflow``. Without this the running accumulator overflows
    int64 silently once enough terms have been added, which corrupts the sum
    rather than reporting it.
    """
    for n in range(_NUMINT - 1, 0, -1):
        # Truncate toward zero, exactly, without going through float64.
        carry = np.sign(digits[n]) * (np.abs(digits[n]) >> _NUMBIT)
        digits[n] -= carry * _PREC_INT
        digits[n - 1] += carry
    return digits


#: Elements handled per pass, so the temporaries stay in cache. Larger local
#: axes are processed in slices of rows; the digits are exact integers, so the
#: slicing cannot change the result.
_EFP_CHUNK_ELEMENTS = 1 << 17

_INV_SCALES = tuple(1.0 / scale for scale in _SCALES)


def _to_digits(array: np.ndarray[Any, Any], axis: int) -> np.ndarray[Any, Any]:
    """Sum values into signed integer digits along ``axis``.

    Accumulates in blocks of :data:`_EFP_BLOCK` terms, renormalising after
    each, so an arbitrarily long local axis cannot overflow the accumulator.

    Every scale is a power of two, so ``residual * (1 / scale)`` and
    ``trunc`` are exact and give the same digit as ``sign * floor(|x| /
    scale)``. A scale larger than every magnitude in the slice contributes
    nothing, and the loop stops once the residual is exhausted, so typical
    data touches three of the six digits.
    """
    values = np.moveaxis(np.asarray(array, dtype=np.float64), axis, 0)
    digits = np.zeros((_NUMINT, *values.shape[1:]), dtype=np.int64)
    width = max(1, int(np.prod(values.shape[1:], dtype=np.int64)))
    rows = max(1, min(_EFP_BLOCK, _EFP_CHUNK_ELEMENTS // width))
    for start in range(0, values.shape[0], _EFP_BLOCK):
        block = values[start : start + _EFP_BLOCK]
        for first in range(0, block.shape[0], rows):
            residual = np.array(block[first : first + rows], dtype=np.float64)
            if residual.size == 0:
                continue
            peak = max(float(residual.max()), -float(residual.min()))
            for n, scale in enumerate(_SCALES):
                if scale > peak:
                    continue
                scaled = residual * _INV_SCALES[n]
                np.trunc(scaled, out=scaled)
                digits[n] += scaled.astype(np.int64).sum(axis=0)
                scaled *= scale
                residual -= scaled
                if n >= 2 and not residual.any():
                    break
        _carry_overflow(digits)
    return digits


def _from_digits(digits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Inverse of :func:`_to_digits`, summing smallest scale first."""
    total = np.zeros(digits.shape[1:], dtype=np.float64)
    for n in range(_NUMINT - 1, -1, -1):
        total += digits[n].astype(np.float64) * _SCALES[n]
    return total


def mpp_reproducing_sum(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm,
    *,
    axis: int | None = None,
) -> np.ndarray[Any, Any]:
    """Compute a rank-count-invariant distributed sum.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    axis : int or None, optional
        Local reduction axis before the global sum.

    Returns
    -------
    numpy.ndarray
        Reproducible global sum.

    Raises
    ------
    ValueError
        If the rank count is unsupported, the input contains non-finite
        values, or the values are too large to sum without overflow.
    """
    if comm.size > MAX_EFP_RANKS:
        # Every rank sees the same communicator size, so this raises on all of
        # them or none: safe to do before a collective.
        raise ValueError(f"mpp_reproducing_sum supports at most {MAX_EFP_RANKS} ranks.")

    flat = np.asarray(local).reshape(-1) if axis is None else local
    reduce_axis = 0 if axis is None else axis

    # FMS `prec_error`: the top digit must stay small enough that summing it
    # across every rank still fits in int64. Checking the inputs rather than
    # the converted digits keeps the conversion itself from overflowing.
    prec_error = (2**63 - 1) // comm.size
    finite = bool(np.all(np.isfinite(flat)))
    representable = finite and bool(
        np.all(np.abs(flat) < prec_error * _SCALES[0] / _EFP_BLOCK)
    )

    if representable:
        digits = _to_digits(flat, reduce_axis)
    else:
        # Contribute zeros so the collective keeps its shape and every rank
        # still reaches the Allreduce that carries the flags.
        shape = np.moveaxis(np.asarray(flat), reduce_axis, 0).shape[1:]
        digits = np.zeros((_NUMINT, *shape), dtype=np.int64)

    # Whether a rank holds non-finite or over-large input is a rank-local
    # fact, so raising on it directly would let one rank leave while the
    # others waited in the Allreduce below, deadlocking the job over a data
    # error. Both flags ride along in the reduction instead.
    payload = np.empty(digits.size + 2, dtype=np.int64)
    payload[:-2] = digits.reshape(-1)
    payload[-2] = 0 if finite else 1
    payload[-1] = (
        0 if representable and not np.any(np.abs(digits[0]) > prec_error) else 1
    )
    total = np.empty_like(payload)
    comm.Allreduce(payload, total, op=MPI.SUM)
    if total[-2]:
        raise ValueError(
            f"mpp_reproducing_sum requires finite input; {int(total[-2])} of "
            f"{comm.size} ranks hold NaN or infinity."
        )
    if total[-1]:
        raise ValueError(
            f"mpp_reproducing_sum overflowed on {int(total[-1])} of "
            f"{comm.size} ranks; the values are too large to sum reproducibly."
        )
    return _from_digits(total[:-2].reshape(digits.shape))


#: Index of each companion field carried alongside a product mantissa.
