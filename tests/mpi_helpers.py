"""Helpers for the MPI case scripts in ``mpi_cases/``.

Each script runs on every rank under ``mpiexec``. Results are compared with
serial xarray/NumPy references and the process exits non-zero when any rank
recorded a failure, so ``test_mpi.py`` can assert on the exit status.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap
import numpy as np
import xarray as xr
import xrmpi as xm
from xrmpi.core.core import MPIXarray
from xrmpi.mpi.mpi_init import MPI

_bootstrap.relax_parallel_probe()

# Names the case scripts pull in with ``from mpi_helpers import *``.
__all__ = [
    "MPI",
    "Filler",
    "MPIXarray",
    "case",
    "check",
    "comm",
    "finish",
    "gather_global",
    "gdata",
    "make",
    "mpi",
    "np",
    "xm",
    "xr",
]

mpi = xm.MPIContext()
comm = mpi.comm

_failures: list[str] = []
_passed = 0


def gdata(shape, seed=0, nan=False):
    """Return a reproducible random array, optionally with 20 % NaN."""
    rng = np.random.default_rng(seed)
    out = rng.normal(size=shape)
    if nan:
        out[rng.random(shape) < 0.2] = np.nan
    return out


class Filler:
    """Picklable fill function slicing a global array by partition bounds."""

    def __init__(self, arr, dims, parts):
        self.arr, self.dims, self.parts = arr, tuple(dims), list(parts)

    def __call__(self, *args):
        idx = [slice(None)] * self.arr.ndim
        for i, part in enumerate(self.parts):
            idx[self.dims.index(part)] = slice(args[2 * i], args[2 * i + 1])
        return self.arr[tuple(idx)]


def make(arr, dims, parts, coords=None, name="v"):
    """Distribute ``arr`` over ``parts`` with ``xrmpi.new_dataarray``."""
    parts = [parts] if isinstance(parts, str) else list(parts)
    return xm.new_dataarray(
        mpi,
        Filler(arr, dims, parts),
        tuple(dims),
        shape=dict(zip(dims, arr.shape, strict=True)),
        dim=parts if len(parts) > 1 else parts[0],
        dtype=arr.dtype,
        coords=coords,
        name=name,
    )


def gather_global(obj):
    """Assemble the global NumPy array behind a distributed result."""
    if not isinstance(obj, MPIXarray):
        return np.asarray(obj)
    data, meta = obj.data, obj.meta
    local = np.asarray(data.values)
    if meta is None:
        return local
    dims = list(meta["dims"])
    pieces = comm.allgather(
        ({k: (meta["starts"][k], meta["stops"][k]) for k in dims}, local)
    )
    shape = list(local.shape)
    for k in dims:
        shape[data.dims.index(k)] = meta["global_sizes"][k]
    out = (
        np.full(shape, np.nan)
        if local.dtype.kind == "f"
        else np.zeros(shape, dtype=local.dtype)
    )
    for bounds, arr in pieces:
        idx = [slice(None)] * len(shape)
        for k in dims:
            if k in data.dims:
                idx[data.dims.index(k)] = slice(*bounds[k])
        out[tuple(idx)] = arr
    return out


def check(label, got, want, tol=1e-9):
    """Record whether ``got`` matches ``want`` (same shape, close values)."""
    global _passed
    try:
        g, w = np.asarray(got), np.asarray(want)
        ok = g.shape == w.shape and bool(
            np.allclose(g, w, rtol=tol, atol=tol, equal_nan=True)
        )
        detail = "" if ok else f"shape {g.shape} vs {w.shape}"
    except Exception as exc:
        ok, detail = False, repr(exc)
    if ok:
        _passed += 1
    else:
        _failures.append(f"{label}: {detail}")
    return ok


def case(label, function):
    """Run ``function`` and record an exception as a failure."""
    try:
        function()
    except Exception:
        _failures.append(f"{label}: raised\n{traceback.format_exc(limit=6)}")


def finish():
    """Print a per-rank summary and exit non-zero if any rank failed."""
    bad = comm.allreduce(len(_failures), op=MPI.SUM)
    if comm.rank == 0:
        print(f"RESULT passed={_passed} failed={bad} ranks={comm.size}", flush=True)
    for rank in range(comm.size):
        comm.Barrier()
        if rank == comm.rank:
            for line in _failures:
                print(f"FAIL[rank {rank}] {line}", flush=True)
    sys.exit(1 if bad else 0)
