"""Halo-dependent operation correctness: rolling_reduce, coarsen_reduce,
diff, shift, differentiate, ffill, bfill -- under a single partition
dimension, a two-dimensional (Cartesian) partition, and the shared
deliberately-uneven single-dimension partition (see mpi_test_common).
"""

from __future__ import annotations

import xarray as xr

from xrmpi import MPIContext
from xrmpi.core.core import MPIXarray
from xrmpi.test.mpi_test_common import (
    Fixtures,
    is_declared_halo_refusal,
    local_of,
    record,
)

mpi = MPIContext()


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d
    native_uneven, dist_uneven = fx.native_uneven, fx.dist_uneven

    def run_op(op_name, fn, native_fn, case):
        try:
            result = fn()
        except NotImplementedError as e:
            record(
                op_name, case, None, f"NotImplementedError (declared): {str(e)[:120]}"
            )
            return
        except ValueError as e:
            if is_declared_halo_refusal(e):
                record(
                    op_name, case, None, f"declared refusal (expected): {str(e)[:150]}"
                )
            else:
                record(op_name, case, False, f"unexpected ValueError: {str(e)[:150]}")
            return
        except Exception as e:
            record(
                op_name, case, False, f"unexpected {type(e).__name__}: {str(e)[:150]}"
            )
            return
        try:
            import xarray as xr

            local = local_of(result)
            m = result.meta if isinstance(result, MPIXarray) else None
            if m is None:
                record(op_name, case, False, "result lost its MPI metadata")
                return
            expected_full = native_fn()
            sel = {d: slice(m["starts"][d], m["stops"][d]) for d in m["dims"]}
            expected = expected_full.isel(sel)
            xr.testing.assert_allclose(local, expected, rtol=1e-6)
            record(op_name, case, True)
        except Exception as e:
            record(op_name, case, False, str(e)[:200])

    for case_label, d, dm in [
        ("1d(time)", dist, "time"),
        ("2d(lat,lon)/lat", dist2d, "lat"),
        ("2d(lat,lon)/lon", dist2d, "lon"),
        ("1d(x), uneven", dist_uneven, "x"),
    ]:
        native_here = native_uneven if dm == "x" else native
        # rolling_reduce()'s default reduce="mean" hits the exact same
        # bottleneck-vs-non-bottleneck non-reproducibility documented in
        # detail below for the rolling().{mean,sum} loop: bottleneck's
        # rolling-mean is a streaming, running-sum algorithm whose
        # per-window floating-point error depends on how much data has
        # already passed through its internal accumulator, which is not
        # invariant to how the array is split across ranks. Comparing
        # climtools' independent, short-accumulator per-rank result
        # against native's single long-accumulator run is therefore not a
        # meaningful correctness check while bottleneck is in play, so
        # this pinned comparison -- like the loop below -- forces the
        # stable, non-bottleneck path on both sides.
        with xr.set_options(use_bottleneck=False):
            run_op(
                "rolling_reduce",
                lambda d=d, dm=dm: d.rolling_reduce(dm, window=3, reduce="mean"),
                lambda dm=dm: native_here.rolling({dm: 3}, center=True).mean(),
                case_label,
            )
        run_op(
            "coarsen_reduce",
            lambda d=d, dm=dm: d.coarsen_reduce(
                dm, window=2, reduce="mean", boundary="trim"
            ),
            lambda dm=dm: native_here.coarsen({dm: 2}, boundary="trim").mean(),
            case_label,
        )
        run_op(
            "diff",
            lambda d=d, dm=dm: d.diff(dm, n=1),
            lambda dm=dm: native_here.diff(dm, n=1),
            case_label,
        )
        run_op(
            "shift",
            lambda d=d, dm=dm: d.shift(dm, periods=1),
            lambda dm=dm: native_here.shift({dm: 1}),
            case_label,
        )
        run_op(
            "differentiate",
            lambda d=d, dm=dm: (
                d.differentiate(dm)
                if dm != "time"
                else d.differentiate(dm, datetime_unit="s")
            ),
            lambda dm=dm: (
                native_here.differentiate(dm)
                if dm != "time"
                else native_here.differentiate(dm, datetime_unit="s")
            ),
            case_label,
        )
        run_op(
            "ffill",
            lambda d=d, dm=dm: d.ffill(dm, limit=2),
            lambda dm=dm: native_here.ffill(dm, limit=2),
            case_label,
        )
        run_op(
            "bfill",
            lambda d=d, dm=dm: d.bfill(dm, limit=2),
            lambda dm=dm: native_here.bfill(dm, limit=2),
            case_label,
        )
        # .rolling(dim, window).<reduce>(): the chainable MPIRolling handle
        # (MPIXarray.rolling -> handles.MPIRolling), distinct from and
        # never previously exercised alongside rolling_reduce() above even
        # though it is documented as dispatching straight through it --
        # that dispatch itself (and each of the six reduce names,
        # including count()) was genuinely unverified.
        # bottleneck's rolling reductions (xarray's default fast path when
        # it's installed, which it is here) use streaming, running-sum
        # accumulators (the textbook-unstable E[x^2]-E[x]^2 formula for
        # std; an incremental running total for mean/sum). Their per-window
        # rounding error is a function of how much data has already
        # streamed through that accumulator, which is *not* invariant to
        # how the array is split -- confirmed directly, not assumed, for
        # all three: running each once over the full 24-length native
        # array drifts window-by-window away from a hand-computed ground
        # truth, while climtools runs it independently per rank over each
        # rank's much shorter halo-padded slice, which drifts far less
        # over the same positions. Comparing native's (long-accumulator)
        # bottleneck path against climtools' (short-accumulator) one would
        # therefore fail even for a perfectly correct distributed
        # implementation, for a reason that has nothing to do with either
        # side's correctness -- so mean/sum/std, on both sides, use the
        # stable non-bottleneck path, which is not sensitive to this and
        # gives a meaningful comparison; min/max/count are exact regardless
        # of accumulator strategy and are unaffected either way, so they
        # stay on xarray's normal default.
        for reduce_name in ("mean", "sum", "min", "max", "std", "count"):
            with xr.set_options(
                use_bottleneck=(reduce_name not in ("mean", "sum", "std"))
            ):
                run_op(
                    f"rolling().{reduce_name}",
                    lambda d=d, dm=dm, r=reduce_name: getattr(
                        d.rolling(dm, window=3), r
                    )(),
                    lambda dm=dm, r=reduce_name: getattr(
                        native_here.rolling({dm: 3}, center=True), r
                    )(),
                    case_label,
                )
        mpi.comm.barrier()
