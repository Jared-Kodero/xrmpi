"""Shared helpers and fixtures for the mpi_test_* correctness suite.

Every mpi_test_*.py module in this directory imports from here rather
than duplicating fixture setup or the RESULTS/record/report machinery.
`mpi_test.py` is the single entry point: it builds the fixtures once via
`build_fixtures()`, imports each mpi_test_*.py module and calls its
`run(fixtures)`, then calls `report()`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import xarray as xr
from xgeo.xarray.core import MPIXarray

import xgeo as xg
from xgeo import MPIContext
from xrmpi.mpp.mpp_do_update import HaloWidthError
from xrmpi.test.mock_dataset import PATH, PATH2D, create_dataset

mpi = MPIContext()

RESULTS: list[tuple[str, str, bool | None, str]] = []

#: A 1D length that is not divisible by 2, 3, 4, 5, or 6 -- uneven for any
#: rank count in that range except an exact divisor, and (unlike the
#: shared 1D `dist` fixture's time=12, which divides evenly at every rank
#: count the suite normally runs at) genuinely exercises the deficient
#: (some rank(s) hold none/less than a halo width) code path deliberately
#: rather than by incidental luck of a 2D Cartesian factorization. Every
#: module that wants a deterministically-uneven 1D case should use
#: `Fixtures.dist_uneven`/`native_uneven` rather than building its own --
#: this was previously duplicated ad hoc inside mpi_test_halo_ops.py.
UNEVEN_GLOBAL = 21


#: Stream each check to stdout as it completes, in addition to the summary
#: report at the end. Without this the suite is silent from the partition
#: reports until `report()`, so a run that stalls -- or is killed by the
#: scheduler -- leaves a log that cannot distinguish "deadlocked early" from
#: "was nearly finished". Rank 0 only, because interleaved output from every
#: rank is unreadable; a stall on another rank shows up as rank 0 blocking in
#: the next collective, which the watchdog then reports. Set
#: CLIMTOOLS_TEST_QUIET=1 to suppress.
_STREAM = os.environ.get("CLIMTOOLS_TEST_QUIET") != "1"
_STARTED = time.monotonic()

#: Monotonic stamp of the last completed check, on this rank. The phase
#: watchdog measures against this rather than against the phase's own start:
#: at production scale a single halo_ops check can run for minutes and the
#: whole phase for well over an hour, so a timeout measured from the start
#: fires on a perfectly healthy run.
_LAST_PROGRESS = time.monotonic()

#: Seconds without a completed check before the watchdog dumps every rank's
#: stack. Long enough to clear the slowest single check at production scale
#: (assert_allclose over a multi-GB variable is minutes on its own).
_WATCHDOG_TIMEOUT = float(os.environ.get("CLIMTOOLS_TEST_WATCHDOG") or 1800.0)


def record(op: str, case: str, ok: bool | None, msg: str = "") -> None:
    """Record one check's outcome. `ok=None` means a declared
    NotImplementedError under an unsupported partition shape -- reported
    as SKIP, not FAIL."""
    global _LAST_PROGRESS
    RESULTS.append((op, case, ok, msg))
    _LAST_PROGRESS = time.monotonic()
    if _STREAM and mpi.comm.rank == 0:
        status = "SKIP" if ok is None else ("ok  " if ok else "FAIL")
        elapsed = time.monotonic() - _STARTED
        # The reason belongs on the streamed line, not only in the summary
        # report: a run killed by the scheduler never reaches the report, so
        # a bare FAIL leaves nothing to work from.
        reason = f" -- {msg}" if msg and ok is not True else ""
        print(f"  [{elapsed:7.1f}s] {status} {op} :: {case}{reason}", flush=True)


def phase(label: str, timeout: float | None = None):
    """Announce a stage and dump every rank's stack if it stops progressing.

    The watchdog is armed with `record`'s heartbeat, so it reports a phase
    that has genuinely stopped rather than one that is merely long. It also
    does not abort: an earlier version killed a healthy run at 1489 s because
    `halo_ops` legitimately exceeded a fixed 900 s, and losing a
    twenty-five-minute job to a false positive costs far more than letting a
    real deadlock run on to the scheduler's own wall limit. The stack dump --
    the part with the diagnostic value -- happens either way.
    """
    if _STREAM and mpi.comm.rank == 0:
        elapsed = time.monotonic() - _STARTED
        print(f"\n=== [{elapsed:7.1f}s] {label} ===", flush=True)
    return mpi.watchdog(
        label,
        timeout=_WATCHDOG_TIMEOUT if timeout is None else timeout,
        abort=False,
        progress=lambda: _LAST_PROGRESS,
    )


#: Substring of the ValueError mpp_halo_exchange() raises when some rank's
#: local partition along the requested dimension is shorter than the
#: before/after halo width being asked of it (see
#: xgeo.xarray.arithmetic.mpp_halo_exchange's docstring). Every
#: halo-based op -- rolling_reduce, coarsen_reduce, diff, shift,
#: differentiate, ffill, bfill, roll, ... -- funnels through the same
#: mpp_halo_exchange() and so can hit this identical, deliberate refusal
#: whenever a fixture's uneven/undersized partition (see UNEVEN_GLOBAL
#: above) meets a large enough halo width at a given rank count; it is
#: not a bug in the op itself. `is_declared_halo_refusal` is the single
#: place that recognizes this pattern, used by every mpi_test_*.py
#: module's run/except logic so a genuine architectural refusal is
#: always classified as a SKIP (`record(..., None, ...)`), never
#: mischaracterized as a FAIL, regardless of which module raised it.
_DECLARED_HALO_REFUSAL = "shorter than the requested halo"


def is_declared_halo_refusal(exc: Exception) -> bool:
    """True if `exc` is mpp_halo_exchange()'s declared undersized-partition
    refusal rather than a genuine, unexpected failure.

    By type first. The substring fallback only covers a climtools built
    before HaloWidthError existed; matching on message text is what let a
    reworded error turn a whole class of expected skips into failures.
    """
    return isinstance(exc, HaloWidthError) or (
        isinstance(exc, ValueError) and _DECLARED_HALO_REFUSAL in str(exc)
    )


def local_of(value):
    """Materialize this rank's local slice, whether MPIXarray or plain xarray."""
    return value._prepare().load() if isinstance(value, MPIXarray) else value


@dataclass(frozen=True)
class Fixtures:
    native: xr.Dataset
    dist: MPIXarray
    dist2d: MPIXarray
    native_uneven: xr.DataArray
    dist_uneven: MPIXarray

    def gsize(self, dim: str) -> int:
        return self.native.sizes[dim]


def build_fixtures() -> Fixtures:
    """Generate the shared mock dataset once and open it two ways: a 1D
    partition (dim='time') and a 2D Cartesian partition (dims=('lat',
    'lon')) -- every test module compares against the same underlying
    data, just partitioned differently. Also builds a small, separate,
    deliberately-uneven 1D DataArray fixture (see `UNEVEN_GLOBAL`).

    Defaults are the production-scale config (721x1440 grid, 720 steps).
    CLIMTOOLS_TEST_NTIME/_RESOLUTION/_PLEV_STEP override them for local
    iteration on a machine that cannot hold it -- env/env.md previously
    told the reader to edit this call by hand, which is easy to commit by
    accident. The full-size defaults still apply to any run that does not
    set them."""
    create_dataset(
        n_time=int(os.environ.get("CLIMTOOLS_TEST_NTIME") or 24 * 30),
        resolution_deg=float(os.environ.get("CLIMTOOLS_TEST_RESOLUTION") or 0.25),
        plev_step=float(os.environ.get("CLIMTOOLS_TEST_PLEV_STEP") or 100),
    )

    native = xr.open_dataset(PATH).load()
    dist = xg.open_distributed_dataset(
        PATH, mpi, partition_dim="time", log_partitions=True
    )
    dist2d = xg.open_distributed_dataset(
        PATH2D, mpi, partition_dim=("lat", "lon"), log_partitions=True
    )

    def fill_uneven(a, b):
        idx = np.arange(a, b, dtype=np.float64)
        return np.sin(idx) * (idx + 1.0)

    dist_uneven = xg.create_distributed_dataarray(
        mpi,
        fill_uneven,
        dims=("x",),
        shape={"x": UNEVEN_GLOBAL},
        dim="x",
        log_partitions=False,
        name="v",
    )
    idx_global = np.arange(UNEVEN_GLOBAL, dtype=np.float64)
    native_uneven = xr.DataArray(
        np.sin(idx_global) * (idx_global + 1.0), dims=("x",), name="v"
    )
    return Fixtures(
        native=native,
        dist=dist,
        dist2d=dist2d,
        native_uneven=native_uneven,
        dist_uneven=dist_uneven,
    )


def report() -> None:
    gathered = mpi.comm.gather(RESULTS, root=0)
    if mpi.comm.rank != 0:
        return
    combined: dict[tuple[str, str], list[bool | None]] = {}
    msgs: dict[tuple[str, str], str] = {}
    for rank_results in gathered:
        for op, case, ok, msg in rank_results:
            key = (op, case)
            combined.setdefault(key, []).append(ok)
            if msg:
                msgs[key] = msg
    print(f"\n=== mpi_test results ({mpi.comm.size} ranks) ===")
    n_pass = n_skip = n_fail = 0
    for (op, case), oks in combined.items():
        if all(o is None for o in oks):
            status = "SKIP"
            n_skip += 1
        elif all(o for o in oks if o is not None) and all(o is not None for o in oks):
            status = "PASS"
            n_pass += 1
        else:
            status = "FAIL"
            n_fail += 1
        print(f"[{status}] {op:<18} {case:<20} {msgs.get((op, case), '')}")
    print(
        f"--- {n_pass} passed, {n_fail} failed, {n_skip} skipped "
        f"(declared NotImplementedError under an unsupported partition shape) "
        f"of {len(combined)} checks ---"
    )
    if n_fail:
        raise SystemExit(1)
