"""Scan, redistribution, and dispatch correctness: np.log/isel (rank-local
NumPy dispatch and indexing), cumsum, sortby, reindex, interp, matmul --
single-dim and, where supported, multi-dim, with explicit no-duplication
+ exact-coverage verification for the redistributing operations.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from xgeo.xarray.core import MPIXarray

import xgeo as xg
from xgeo import MPIContext
from xrmpi.test.mpi_test_common import Fixtures, local_of, record

mpi = MPIContext()


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d
    start, stop = dist.meta["start"], dist.meta["stop"]

    # -- rank-local NumPy dispatch and indexing ------------------------------
    try:
        logged = np.log(dist.data["pr"])
        expected = np.log(native["pr"]).isel(time=slice(start, stop))
        xr.testing.assert_allclose(local_of(logged), expected, rtol=1e-5)
        record("np.log", "1d(time), NumPy dispatch", True)
    except Exception as e:
        record("np.log", "1d(time), NumPy dispatch", False, str(e)[:200])

    try:
        sub = local_of(dist.isel(time=slice(0, 3)))
        record("isel", "1d(time), indexing", True, str(dict(sub.sizes)))
    except Exception as e:
        record("isel", "1d(time), indexing", False, str(e)[:200])
    mpi.comm.barrier()

    # sel(): label-based counterpart of isel() above, on the partition
    # dimension -- never previously exercised anywhere in this suite.
    # native.time carries a CF "hours since ..." units attribute, so
    # xr.open_dataset (mpi_test_common.build_fixtures) decodes it to
    # datetime64 on load; slice on the actual (decoded) coordinate
    # values, exactly as a real caller would, not on the raw pre-decode
    # float hours.
    try:
        t0 = native.time.values[2]
        t1 = native.time.values[6]
        result = dist.sel(time=slice(t0, t1))
        local = local_of(result)
        expected_full = native.sel(time=slice(t0, t1))
        m = result.meta if isinstance(result, MPIXarray) else None
        if m is None:
            expected = expected_full
        else:
            d = m["dims"][0]
            s, e = m["starts"][d], m["stops"][d]
            expected = expected_full.isel({d: slice(s, e)})
        xr.testing.assert_allclose(local, expected, rtol=1e-6)
        record("sel", "1d(time), label slice", True)
    except Exception as e:
        record(
            "sel", "1d(time), label slice", False, f"{type(e).__name__}: {str(e)[:200]}"
        )
    mpi.comm.barrier()

    # -- where(): elementwise selection, single-dim and multi-dim, plus
    #    the drop=True guard (unsupported on a distributed object since
    #    it could remove a different number of positions per rank) -------
    try:
        threshold = 0.0002  # fixed threshold; avoids any local-vs-global mean ambiguity
        cond = dist.data["pr"] > threshold
        result = local_of(dist.where(cond, other=-1.0))
        native_cond = native["pr"].isel(time=slice(start, stop)) > threshold
        expected = native.isel(time=slice(start, stop)).where(native_cond, other=-1.0)
        xr.testing.assert_allclose(result, expected, rtol=1e-5)
        record("where", "1d(time)", True)
    except Exception as e:
        record("where", "1d(time)", False, f"{type(e).__name__}: {str(e)[:200]}")

    try:
        dist.where(dist.data["pr"] > 0, drop=True)
        ok = False
        msg = "no exception raised -- silently wrong?"
    except ValueError:
        ok = True
        msg = ""
    except Exception as e:
        ok = False
        msg = f"wrong exception type {type(e).__name__}: {e}"
    record("where", "1d(time), drop=True guard", ok, msg)
    mpi.comm.barrier()

    m2 = dist2d.meta
    lat_s, lat_e = m2["starts"]["lat"], m2["stops"]["lat"]
    lon_s, lon_e = m2["starts"]["lon"], m2["stops"]["lon"]
    try:
        cond2d = dist2d.data["pr"] > threshold
        result2d = local_of(dist2d.where(cond2d, other=-1.0))
        native_cond2d = (
            native["pr"].isel(lat=slice(lat_s, lat_e), lon=slice(lon_s, lon_e))
            > threshold
        )
        expected2d = native.isel(
            lat=slice(lat_s, lat_e), lon=slice(lon_s, lon_e)
        ).where(native_cond2d, other=-1.0)
        xr.testing.assert_allclose(result2d, expected2d, rtol=1e-5)
        record("where", "2d(lat,lon)", True)
    except Exception as e:
        record("where", "2d(lat,lon)", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- scans and redistribution --------------------------------------------
    def check_single_dim(op_name, fn, native_fn, case="1d(time)", *, rtol=1e-6):
        try:
            result = fn()
            local = local_of(result)
            m = result.meta if isinstance(result, MPIXarray) else None
            expected_full = native_fn()
            if m is None:
                xr.testing.assert_allclose(local, expected_full, rtol=rtol)
            else:
                d = m["dims"][0]
                s, e = m["starts"][d], m["stops"][d]
                xr.testing.assert_allclose(
                    local, expected_full.isel({d: slice(s, e)}), rtol=rtol
                )
            record(op_name, case, True)
        except Exception as e:
            record(op_name, case, False, f"{type(e).__name__}: {str(e)[:200]}")

    def check_multidim(op_name, fn, native_fn, *, moved_dim, case="2d(lat,lon)"):
        """No two ranks claim the exact same full region across every
        surviving dimension at once, AND -- grouping ranks by their bounds
        on every surviving dimension other than moved_dim -- each such
        group's own moved_dim ranges exactly, non-overlappingly cover
        [0, global_size). moved_dim's range legitimately repeats across
        different positions of any OTHER surviving dimension (e.g.
        cumsum(lat) redistributes lat independently within each
        lon-group), so a naive global check would wrongly flag that as
        duplication.
        """
        try:
            result = fn()
            m = result.meta
            sel = {d: (m["starts"][d], m["stops"][d]) for d in m["dims"]}
            # The dimension's *post-op* global size, not its pre-op size
            # (fx.gsize(moved_dim), the original native.sizes[moved_dim]):
            # every op this helper checks so far (cumsum, sortby, reindex)
            # happens to leave moved_dim's length unchanged, so those two
            # values were always identical and this distinction was never
            # exercised -- but interp() legitimately changes the length
            # along moved_dim to the interpolation target's length, and a
            # coverage buffer sized from the wrong (stale, pre-op) length
            # can only accidentally agree with the real one. Each rank's
            # own reattached .meta already carries the correct new size
            # for exactly the redistributed range that rank now owns, at
            # no extra communication cost.
            new_moved_size = int(m["global_sizes"][moved_dim])
            local = local_of(result)
            s, e = sel[moved_dim]
            shape_ok = local.sizes.get(moved_dim, 0) == (e - s)
            if e > s:
                expected_full = native_fn()
                expected = expected_full.isel({d: slice(*b) for d, b in sel.items()})
                xr.testing.assert_allclose(local, expected, rtol=1e-5)

            all_sel = mpi.comm.gather(
                (tuple(sorted(sel.items())), new_moved_size), root=0
            )
            ok = shape_ok
            msg = ""
            if mpi.comm.rank == 0:
                sel_only = [entry for entry, _ in all_sel]
                no_full_dup = len(sel_only) == len(set(sel_only))
                groups: dict[tuple, list[tuple[int, int]]] = {}
                group_sizes: dict[tuple, set[int]] = {}
                for entry, moved_size in all_sel:
                    d = dict(entry)
                    other = tuple(
                        sorted((k, v) for k, v in d.items() if k != moved_dim)
                    )
                    groups.setdefault(other, []).append(d[moved_dim])
                    group_sizes.setdefault(other, set()).add(moved_size)
                per_group_ok = True
                for other, ranges in groups.items():
                    sizes_seen = group_sizes[other]
                    if len(sizes_seen) != 1:
                        per_group_ok = False
                        continue
                    coverage = np.zeros(next(iter(sizes_seen)), dtype=int)
                    for s_, e_ in ranges:
                        coverage[s_:e_] += 1
                    if not np.all(coverage == 1):
                        per_group_ok = False
                ok = shape_ok and no_full_dup and per_group_ok
                if not (no_full_dup and per_group_ok):
                    msg = f"no_full_dup={no_full_dup} per_group_coverage_ok={per_group_ok}"
            all_ok = mpi.comm.gather(ok, root=0)
            if mpi.comm.rank == 0:
                record(op_name, case, all(all_ok), msg)
        except NotImplementedError as e:
            record(op_name, case, None, f"NotImplementedError: {str(e)[:150]}")
        except Exception as e:
            record(
                op_name, case, False, f"unexpected {type(e).__name__}: {str(e)[:150]}"
            )

    # cumsum's cross-rank prefix-sum (each rank's local xarray .cumsum(),
    # each rank's own .sum() gathered and turned into an exclusive running
    # total via Exscan-style prefix addition -- see elementwise.py's
    # _cumsum_scan) sums the exact same float32 values as native's single
    # in-order .cumsum() but in a genuinely different addition order
    # (per-rank partial sums combined afterward, vs one long running
    # total). float32 addition is not associative, so this is a real,
    # unavoidable divergence from native's result, not a bug: confirmed
    # directly by rerunning this exact check at n_time=720 (this fixture's
    # production size) and measuring the actual relative error directly,
    # rather than assuming -- it peaks at essentially float32 epsilon
    # accumulated across the interior of a summation, a couple e-7,
    # comfortably under 1e-6 at that specific size but with no analytic
    # guarantee of staying there for a different length or rank count
    # (the textbook worst-case bound for a length-n float32 sum is
    # O(n * eps) =~ 720 * 6e-8 =~ 4e-5). rtol=1e-4 keeps this check
    # sensitive to a genuine off-by-one/duplication bug in the prefix-sum
    # logic (which would corrupt entire trailing segments, not shift the
    # last couple of significant digits) while not being tighter than
    # float32 accumulation itself can honestly promise.
    check_single_dim(
        "cumsum",
        lambda: dist.cumsum("time"),
        lambda: native.cumsum("time"),
        rtol=1e-4,
    )
    check_multidim(
        "cumsum",
        lambda: dist2d.cumsum("lat"),
        lambda: native.cumsum("lat"),
        moved_dim="lat",
    )
    # cumsum carries a running total forward across ranks (each rank's
    # start value depends on an Exscan/prefix-sum over every rank before
    # it), so a rank holding very few -- or, at the extreme, zero --
    # elements is exactly the case most likely to expose an off-by-one in
    # that carry. dist/time=12 and dist2d/lat=19 both happen to give
    # every rank at least a couple of elements at the suite's usual rank
    # counts; the shared uneven fixture (mpi_test_common.UNEVEN_GLOBAL=21)
    # is deliberately not guaranteed to.
    check_single_dim(
        "cumsum",
        lambda: fx.dist_uneven.cumsum("x"),
        lambda: fx.native_uneven.cumsum("x"),
        case="1d(x), uneven",
        rtol=1e-4,
    )
    mpi.comm.barrier()

    check_single_dim(
        "sortby",
        lambda: dist.sortby("time", ascending=False),
        lambda: native.sortby("time", ascending=False),
    )
    check_multidim(
        "sortby",
        lambda: dist2d.sortby("lat", ascending=False),
        lambda: native.sortby("lat", ascending=False),
        moved_dim="lat",
    )
    mpi.comm.barrier()

    new_time = native.time.values[::-1][:8]
    check_single_dim(
        "reindex",
        lambda: dist.reindex(time=new_time),
        lambda: native.reindex(time=new_time),
    )
    new_lat = native.lat.values[::-1]
    check_multidim(
        "reindex",
        lambda: dist2d.reindex(lat=new_lat),
        lambda: native.reindex(lat=new_lat),
        moved_dim="lat",
    )
    mpi.comm.barrier()

    # interp -- Allgather-based; not halo-bounded, checked under the
    # partition dimension it interpolates along.
    from xrmpi.mpp.ext_domains import dim_comm as _dim_comm_check
    from xrmpi.mpp.mpp_domains_define import mpp_compute_extent as _gbb_check

    new_lat_fine = np.linspace(native.lat.values.min(), native.lat.values.max(), 37)
    sub = _dim_comm_check(dist2d.meta, "lat", mpi)
    s, e = _gbb_check(len(new_lat_fine), sub.rank, sub.size)
    check_multidim(
        "interp",
        lambda: dist2d.interp("lat", new_lat_fine[s:e]),
        lambda: native.interp(lat=new_lat_fine),
        moved_dim="lat",
    )
    mpi.comm.barrier()

    # matmul -- 2D DataArrays, partitioned along the shared contraction dim
    # (single-dim only by nature; not attempted under a multi-dim partition).
    GXM, GYM = 12, 5

    def fill_left(a, b):
        return np.arange(a, b, dtype=np.float64)[:, None] * np.ones((1, GYM))

    left_1d = xg.create_distributed_dataarray(
        mpi,
        fill_left,
        dims=("x", "y"),
        shape={"x": GXM, "y": GYM},
        dim="x",
        log_partitions=False,
        name="left",
    )
    right_native = xr.DataArray(
        np.arange(GYM * 3, dtype=np.float64).reshape(GYM, 3), dims=("y", "z")
    )
    native_left = xr.DataArray(
        np.arange(GXM, dtype=np.float64)[:, None] * np.ones((1, GYM)), dims=("x", "y")
    )
    check_single_dim(
        "matmul",
        lambda: left_1d.matmul(right_native),
        lambda: native_left.dot(right_native, dim="y"),
        case="1d(x)",
    )

    # Scalar isel/sel elect a single owning rank. That election used to
    # gather every rank's bounds (or its match flag) and scan the list; it is
    # now a fixed-size reduction, so the outcomes that reduction has to
    # distinguish -- one owner, no owner, and an out-of-range request -- are
    # checked here rather than left to the happy path alone.
    try:
        picked = dist.isel(time=int(native.sizes["time"]) - 1)
        expected = native.isel(time=int(native.sizes["time"]) - 1)
        xr.testing.assert_allclose(local_of(picked), expected, rtol=1e-6)
        record("isel", "scalar, last global index", True)
    except Exception as e:
        record(
            "isel",
            "scalar, last global index",
            False,
            f"{type(e).__name__}: {e!s:.200}",
        )

    try:
        dist.isel(time=int(native.sizes["time"]))
        record("isel", "scalar, out of range raises", False, "no IndexError raised")
    except IndexError:
        record("isel", "scalar, out of range raises", True)
    except Exception as e:
        record(
            "isel",
            "scalar, out of range raises",
            False,
            f"raised {type(e).__name__}, wanted IndexError",
        )

    try:
        # Every rank agrees the label is absent, so no rank claims ownership.
        dist.sel(time=np.datetime64("1600-01-01"))
        record("sel", "scalar, absent label raises", False, "no KeyError raised")
    except KeyError:
        record("sel", "scalar, absent label raises", True)
    except Exception as e:
        record(
            "sel",
            "scalar, absent label raises",
            False,
            f"raised {type(e).__name__}, wanted KeyError",
        )
