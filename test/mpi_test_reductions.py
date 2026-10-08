"""Reduction correctness: rank-local reductions, cross-rank reduction with
reconstruction (auto-repartition after a reduction removes a partition
dimension), and -- under a multi-dimensional partition -- explicit
no-duplication + exact-coverage verification (see finish()'s dedup fix
in planning.py, and median's own gather-to-root dedup).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xrmpi import MPIContext
from xrmpi.test.mpi_test_common import Fixtures, local_of, record

mpi = MPIContext()


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d
    start, stop = dist.meta["start"], dist.meta["stop"]

    # -- rank-local reduction (dim not the partition dim) -------------------
    try:
        local_mean = local_of(dist.mean(dim="lat"))
        expected = native.isel(time=slice(start, stop)).mean(dim="lat")
        xr.testing.assert_allclose(local_mean, expected, rtol=1e-5)
        record("mean(dim='lat')", "1d(time), rank-local", True)
    except Exception as e:
        record("mean(dim='lat')", "1d(time), rank-local", False, str(e)[:200])

    # -- cross-rank reduction removing the sole partition dim: result
    #    auto-repartitions onto the remaining dimension ---------------------
    try:
        global_mean = dist.mean(dim="time")
        new_dim = global_mean.meta["dim"]
        s2, e2 = global_mean.meta["start"], global_mean.meta["stop"]
        expected = native.mean(dim="time").isel({new_dim: slice(s2, e2)})
        xr.testing.assert_allclose(local_of(global_mean), expected, rtol=1e-5)
        record("mean(dim='time')", "1d(time), reduction+reconstruction", True)
    except Exception as e:
        record(
            "mean(dim='time')",
            "1d(time), reduction+reconstruction",
            False,
            str(e)[:200],
        )
    mpi.comm.barrier()

    # -- multi-dim reductions: correct values AND exactly-once,
    #    non-overlapping coverage of the surviving axis ---------------------
    def check_reduction_2d(op_name, reduce_dim, apply_fn, native_fn):
        try:
            result = apply_fn()
            m = result.meta
            surviving = m["dims"][0]
            s, e = m["starts"][surviving], m["stops"][surviving]
            local = local_of(result)
            shape_ok = local.sizes.get(surviving, 0) == (e - s)
            if e > s:
                expected = native_fn().isel({surviving: slice(s, e)})
                xr.testing.assert_allclose(local, expected, rtol=1e-5)
            bounds = mpi.comm.gather((s, e), root=0)
            ok = shape_ok
            if mpi.comm.rank == 0:
                coverage = np.zeros(fx.gsize(surviving), dtype=int)
                for s_, e_ in bounds:
                    coverage[s_:e_] += 1
                ok = ok and bool(np.all(coverage == 1))
            all_ok = mpi.comm.gather(ok, root=0)
            if mpi.comm.rank == 0:
                record(
                    op_name,
                    f"2d(lat,lon)/{reduce_dim}, reduction+reconstruction",
                    all(all_ok),
                )
        except Exception as e:
            record(
                op_name,
                f"2d(lat,lon)/{reduce_dim}, reduction+reconstruction",
                False,
                str(e)[:200],
            )

    for op_name, apply_fn, native_fn in [
        ("mean", lambda: dist2d.mean(dim="lat"), lambda: native.mean(dim="lat")),
        ("sum", lambda: dist2d.sum(dim="lat"), lambda: native.sum(dim="lat")),
        ("min", lambda: dist2d.min(dim="lat"), lambda: native.min(dim="lat")),
        ("max", lambda: dist2d.max(dim="lat"), lambda: native.max(dim="lat")),
        ("var", lambda: dist2d.var(dim="lat"), lambda: native.var(dim="lat")),
        ("std", lambda: dist2d.std(dim="lat"), lambda: native.std(dim="lat")),
        ("median", lambda: dist2d.median("lat"), lambda: native.median("lat")),
    ]:
        check_reduction_2d(op_name, "lat", apply_fn, native_fn)
        mpi.comm.barrier()

    # -- reductions on the shared *1D* deliberately-uneven fixture
    #    (mpi_test_common.UNEVEN_GLOBAL=21): mean's uneven-1D case is
    #    already covered separately in mpi_test_construction.py (the
    #    weighted-vs-naive-average check), but sum/min/max/var/std/median
    #    were, until now, only ever exercised uneven via dist2d's *2D*
    #    Cartesian split above -- a different code path (reductions.py's
    #    multi-dim branch vs its single-partition-dim branch) that a 1D
    #    partition never reached at an uneven length. -------------------
    native_uneven, dist_uneven = fx.native_uneven, fx.dist_uneven

    for op_name, apply_fn, native_fn in [
        ("sum", lambda: dist_uneven.sum(dim="x"), lambda: native_uneven.sum(dim="x")),
        ("min", lambda: dist_uneven.min(dim="x"), lambda: native_uneven.min(dim="x")),
        ("max", lambda: dist_uneven.max(dim="x"), lambda: native_uneven.max(dim="x")),
        ("var", lambda: dist_uneven.var(dim="x"), lambda: native_uneven.var(dim="x")),
        ("std", lambda: dist_uneven.std(dim="x"), lambda: native_uneven.std(dim="x")),
        ("median", lambda: dist_uneven.median("x"), lambda: native_uneven.median("x")),
    ]:
        try:
            result = apply_fn()
            expected = native_fn()
            local = local_of(result)
            xr.testing.assert_allclose(
                local if isinstance(local, xr.DataArray) else xr.DataArray(local),
                expected,
                rtol=1e-5,
            )
            record(op_name, "1d(x), uneven", True)
        except Exception as e:
            record(op_name, "1d(x), uneven", False, str(e)[:200])
        mpi.comm.barrier()
