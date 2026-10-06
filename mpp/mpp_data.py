"""Buffers shared between halo exchanges.

Mirrors FMS ``mpp/mpp_data.F90``, which holds ``mpp_domains_stack``: one
allocation reused by every update rather than a fresh one per call. A stencil
loop performs the same exchange every step, so without reuse each step
allocates and frees buffers whose size never changes.

FMS holds the stack as module state and exposes only
``mpp_domains_set_stack_size``, which keeps its FMS name here. The accessors
below have no FMS counterpart, so they carry no ``mpp_`` prefix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Largest pooled buffer kept per dtype, in elements. A request above this is
#: served by a plain allocation instead, so one unusually wide halo cannot
#: pin memory for the rest of the run.
MPP_STACK_LIMIT = 1 << 24

_stack: dict[np.dtype[Any], list[np.ndarray[Any, Any]]] = {}
_high_water: dict[np.dtype[Any], int] = {}


def get_stack(count: int, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """Return a buffer of at least ``count`` elements, reusing one if possible.

    Parameters
    ----------
    count : int
        Elements required.
    dtype : numpy.dtype
        Element type.

    Returns
    -------
    numpy.ndarray
        A view of exactly ``count`` elements. Its contents are undefined, as
        with ``numpy.empty``.
    """
    if count > MPP_STACK_LIMIT:
        return np.empty(count, dtype=dtype)
    pool = _stack.setdefault(dtype, [])
    for index, buffer in enumerate(pool):
        if buffer.size >= count:
            return pool.pop(index)[:count]
    _high_water[dtype] = max(_high_water.get(dtype, 0), count)
    return np.empty(count, dtype=dtype)


def put_stack(buffer: np.ndarray[Any, Any]) -> None:
    """Return a buffer to the pool once its exchange has completed.

    Parameters
    ----------
    buffer : numpy.ndarray
        Buffer obtained from :func:`get_stack`. Passing a buffer
        that is still in flight corrupts the next exchange to reuse it.
    """
    base = buffer.base if buffer.base is not None else buffer
    if base.size > MPP_STACK_LIMIT:
        return
    _stack.setdefault(base.dtype, []).append(base)


def mpp_domains_set_stack_size(elements: int) -> None:
    """Cap the pooled buffer size, as FMS ``mpp_domains_set_stack_size`` does.

    Parameters
    ----------
    elements : int
        Largest buffer, in elements, that may be retained per dtype.
        Anything larger is allocated and freed per call.
    """
    global MPP_STACK_LIMIT
    MPP_STACK_LIMIT = int(elements)
    for dtype, pool in list(_stack.items()):
        _stack[dtype] = [b for b in pool if b.size <= MPP_STACK_LIMIT]


def stack_size() -> dict[str, int]:
    """Report the high-water mark reached per dtype.

    FMS reports the same figure so a run can be told what
    ``mpp_domains_set_stack_size`` it actually needed.

    Returns
    -------
    dict[str, int]
        Largest element count requested, keyed by dtype name.
    """
    return {str(dtype): count for dtype, count in _high_water.items()}


def free_stack() -> None:
    """Drop every pooled buffer, releasing the memory."""
    _stack.clear()
