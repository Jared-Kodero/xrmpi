"""Correctness checks for the remaining public MPIXarray methods not
covered by any other mpi_test_*.py module: prod, any, all, first, last,
align, evaluate, roll, repartition, apply.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr
from mpi4py import MPI
from xgeo.xarray.core import MPIXarray

from xgeo import MPIContext
from xrmpi.mpp.ext_efp import reproducing_prod
from xrmpi.test.mpi_test_common import (
    Fixtures,
    is_declared_halo_refusal,
    local_of,
    record,
)

mpi = MPIContext()


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d
    start, stop = dist.meta["start"], dist.meta["stop"]

    def check_reduce_1d(op_name, apply_fn, native_fn, case="1d(time)", rtol=1e-5):
        try:
            result = apply_fn()
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

    def check_reduce_2d(op_name, reduce_dim, apply_fn, native_fn, rtol=1e-5):
        # The two gathers below must run on every rank whatever happens above
        # them. They used to sit inside the try, after the comparison, so a
        # rank whose slice failed assert_allclose skipped both while its peers
        # posted them -- and from that point every collective in the job was
        # paired with the wrong one. That is how a single mismatched value in
        # this check turned into a deadlock two phases later.
        s = e = 0
        ok = False
        detail = ""
        surviving = None
        try:
            result = apply_fn()
            m = result.meta
            surviving = m["dims"][0]
            s, e = m["starts"][surviving], m["stops"][surviving]
            local = local_of(result)
            ok = local.sizes.get(surviving, 0) == (e - s)
            if ok and e > s:
                expected = native_fn().isel({surviving: slice(s, e)})
                try:
                    xr.testing.assert_allclose(local, expected, rtol=rtol)
                except AssertionError as exc:
                    ok, detail = False, str(exc)[:200]
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:200]}"

        bounds = mpi.comm.gather((surviving, s, e), root=0)
        states = mpi.comm.gather((ok, detail), root=0)
        if mpi.comm.rank == 0:
            named = [d for d, _, _ in bounds if d is not None]
            covered = False
            if named:
                coverage = np.zeros(fx.gsize(named[0]), dtype=int)
                for _, s_, e_ in bounds:
                    coverage[s_:e_] += 1
                covered = bool(np.all(coverage == 1))
            failures = [d for good, d in states if not good]
            record(
                op_name,
                f"2d(lat,lon)/{reduce_dim}",
                covered and not failures,
                "; ".join(d for d in failures if d)[:200]
                or ("" if covered else "partition coverage is not exactly 1"),
            )

    # -- prod, any, all: standard reductions, same shape as sum/mean.
    #    dim="lat" is rank-local for `dist` (partitioned along "time"),
    #    exactly like roll()'s pre-existing gap above -- so the genuine
    #    cross-rank case (reducing over the dimension `dist` is actually
    #    distributed along, the same "reduction+reconstruction" case
    #    mpi_test_reductions.py's mean(dim='time') deliberately tests)
    #    had never been exercised for any of these three. -------------
    #    Every prod check below runs on the rescaled field. "pr" is U(0, 50)
    #    in float32, whose geometric mean is 50/e, so an unscaled product over
    #    hundreds of points leaves the float32 range entirely -- and once a
    #    partial saturates, whether an exact zero is met before or after it
    #    decides between 0.0 and inf * 0 = NaN, which depends on which rank
    #    held the zero. Dividing by the geometric mean centres the product on
    #    1 regardless of the reduction length, so native is a valid reference.
    #    (Dividing by any other constant is not enough: at 721 points, /25
    #    underflows every column to 0.0 and the comparison passes vacuously.)
    prod_scale = 50.0 / math.e
    scaled_1d = dist / prod_scale
    scaled_2d = dist2d / prod_scale
    native_scaled = native / prod_scale
    #    The tolerance is looser than the 1e-5 the other reductions use, and
    #    has to be: a product of n float32 values accumulates a relative error
    #    of order n * 2**-24, which is 4e-5 at n = 721, and native's sequential
    #    order and the distributed tree order do not make the same roundings.
    #    Demanding 1e-5 of a 721-term float32 product asks for agreement finer
    #    than the arithmetic supports.
    prod_rtol = 1e-3
    check_reduce_1d(
        "prod",
        lambda: scaled_1d.prod(dim="lat"),
        lambda: native_scaled.prod(dim="lat"),
        rtol=prod_rtol,
    )
    check_reduce_2d(
        "prod",
        "lat",
        lambda: scaled_2d.prod(dim="lat"),
        lambda: native_scaled.prod(dim="lat"),
        rtol=prod_rtol,
    )
    # A product along "time" is the one reduction whose native counterpart
    # cannot serve as a reference on the raw fixture. "pr" is U(0, 50) in
    # float32, so a 720-step product is of order 1e911 and every partial
    # saturates to +inf long before the reduction ends. Overflow alone would
    # compare equal, but rng.random(dtype=float32) draws from a 24-bit space,
    # so exact zeros do occur (~10 of them at production size) -- and once a
    # partial is inf, whether a zero is seen before or after it decides
    # between 0.0 and inf * 0 = NaN. That made the old check depend on where
    # a zero happened to land, hence on rank count, and it is what the
    # unseeded RNG intermittently exposed.
    #
    # Two separate properties are checked instead. First, agreement with
    # native on a field whose product is actually representable, which is the
    # only regime where native is a valid reference at all.
    check_reduce_1d(
        "prod",
        lambda: scaled_1d.prod(dim="time"),
        lambda: native_scaled.prod(dim="time"),
        case="1d(time), reduction+reconstruction",
        rtol=prod_rtol,
    )
    mpi.comm.barrier()

    # Second, that the raw overflowing field still reduces to the same answer
    # whatever the rank count -- the reproducibility guarantee FMS states for
    # its own reductions, and the property the distributed implementation is
    # responsible for even where native xarray itself is unreliable.
    try:
        result = dist.prod(dim="time")
        raw = local_of(result)["pr"]
        expected = xr.DataArray(
            reproducing_prod(
                np.asarray(native["pr"].values),
                MPI.COMM_SELF,
                axis=native["pr"].dims.index("time"),
                dtype=np.dtype(np.float32),
            ),
            dims=[d for d in native["pr"].dims if d != "time"],
        )
        # The reduction result is repartitioned, so compare this rank's own
        # slice of it -- same convention as check_reduce_1d above.
        m = result.meta if isinstance(result, MPIXarray) else None
        if m is not None:
            d = m["dims"][0]
            expected = expected.isel({d: slice(m["starts"][d], m["stops"][d])})
        matches = np.array_equal(
            np.asarray(raw.values), np.asarray(expected.values), equal_nan=True
        )
        agreed = mpi.comm.gather(matches, root=0)
        if mpi.comm.rank == 0:
            record("prod", "1d(time), overflowing, rank-count invariant", all(agreed))
    except Exception as e:
        if mpi.comm.rank == 0:
            record(
                "prod",
                "1d(time), overflowing, rank-count invariant",
                False,
                f"{type(e).__name__}: {str(e)[:200]}",
            )
    mpi.comm.barrier()

    def check_bool_reduce_1d(op_name, method, reduce_dim="lat", case="1d(time)"):
        try:
            cond = dist.apply(lambda d: d["pr"] > 0.0002, dist._prepare())
            result = getattr(cond, method)(dim=reduce_dim)
            local = local_of(result)
            m = result.meta if isinstance(result, MPIXarray) else None
            native_cond = native["pr"] > 0.0002
            expected_full = getattr(native_cond, method)(dim=reduce_dim)
            if m is None:
                xr.testing.assert_allclose(local, expected_full, rtol=1e-5)
            else:
                d = m["dims"][0]
                s, e = m["starts"][d], m["stops"][d]
                xr.testing.assert_allclose(
                    local, expected_full.isel({d: slice(s, e)}), rtol=1e-5
                )
            record(op_name, case, True)
        except Exception as e:
            record(op_name, case, False, f"{type(e).__name__}: {str(e)[:200]}")

    check_bool_reduce_1d("any", "any")
    check_bool_reduce_1d("all", "all")
    check_bool_reduce_1d(
        "any", "any", reduce_dim="time", case="1d(time), reduction+reconstruction"
    )
    check_bool_reduce_1d(
        "all", "all", reduce_dim="time", case="1d(time), reduction+reconstruction"
    )
    mpi.comm.barrier()

    # -- prod on the shared deliberately-uneven 1D fixture --------------
    try:
        result = fx.dist_uneven.prod(dim="x")
        local = local_of(result)
        expected = fx.native_uneven.prod(dim="x")
        xr.testing.assert_allclose(
            local if isinstance(local, xr.DataArray) else xr.DataArray(local),
            expected,
            rtol=1e-5,
        )
        record("prod", "1d(x), uneven", True)
    except Exception as e:
        record("prod", "1d(x), uneven", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- first, last: pick a position along one dimension. Compare data
    #    values only (dropping any 'lat' coordinate from both sides):
    #    first()/last() legitimately retain 'lat' as a coordinate
    #    broadcast across the other dims (showing which position was
    #    picked per element), an xarray-inherited quirk of
    #    isel(dim=<vectorized index>, drop=True) not fully dropping the
    #    coordinate the way a scalar-index isel(dim=i, drop=True) does
    #    -- confirmed directly against plain xarray, not a climtools
    #    behavior. -------------------------------------------------------
    def check_first_last(op_name, apply_fn, native_fn, expected_lat_value):
        try:
            result = apply_fn()
            local = local_of(result)
            lat_coord_ok = True
            if "lat" in getattr(local, "coords", {}):
                lat_coord_ok = bool(
                    np.all(local.coords["lat"].values == expected_lat_value)
                )
                local = local.drop_vars("lat")
            m = result.meta if isinstance(result, MPIXarray) else None
            expected_full = native_fn()
            if "lat" in getattr(expected_full, "coords", {}):
                expected_full = expected_full.drop_vars("lat")
            if m is None:
                xr.testing.assert_allclose(local, expected_full, rtol=1e-5)
            else:
                d = m["dims"][0]
                s, e = m["starts"][d], m["stops"][d]
                xr.testing.assert_allclose(
                    local, expected_full.isel({d: slice(s, e)}), rtol=1e-5
                )
            record(
                op_name,
                "1d(time)",
                lat_coord_ok,
                "" if lat_coord_ok else "retained lat coordinate has wrong value",
            )
        except Exception as e:
            record(op_name, "1d(time)", False, f"{type(e).__name__}: {str(e)[:200]}")

    check_first_last(
        "first",
        lambda: dist.first("lat"),
        lambda: native.isel(lat=0, drop=True),
        float(native.lat.values[0]),
    )
    check_first_last(
        "last",
        lambda: dist.last("lat"),
        lambda: native.isel(lat=-1, drop=True),
        float(native.lat.values[-1]),
    )
    mpi.comm.barrier()

    # -- roll: circular shift, same shape/distribution as input. Rolling
    #    along "lat" while `dist` is partitioned along "time" is
    #    rank-local by construction (meta["dims"] doesn't contain "lat"),
    #    so it only ever exercises roll()'s trivial early-return branch
    #    -- never the actual mpp_halo_exchange(periodic=True)-based cross-
    #    rank shift roll() uses when the roll dimension IS the
    #    distributed one (confirmed directly from elementwise.py: the
    #    early return is `if meta is None or dim not in meta["dims"]`).
    #    Added below: roll along "time" itself, the genuinely-distributed
    #    case, at three shift magnitudes -- a small shift (well within
    #    any single rank's local length), a negative shift (exercises
    #    the "after" halo side instead of "before"), and a shift near
    #    global_size//2 (exercises the smallest-magnitude-equivalent
    #    normalization in roll()'s own docstring/comments). -------------
    try:
        result_lat = local_of(dist.roll("lat", shift_by=2))
        expected_lat = native.isel(time=slice(start, stop)).roll(
            lat=2, roll_coords=False
        )
        xr.testing.assert_allclose(result_lat, expected_lat, rtol=1e-6)
        record("roll", "1d(time)/lat", True)
    except Exception as e:
        record("roll", "1d(time)/lat", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    global_time = fx.gsize("time")
    for shift_by, case_label in [
        (2, "1d(time)/time, +shift"),
        (-3, "1d(time)/time, -shift"),
        (global_time // 2, "1d(time)/time, near-half shift"),
    ]:
        try:
            result_time = local_of(dist.roll("time", shift_by=shift_by))
            m = dist.meta
            s, e = m["start"], m["stop"]
            expected_full = native.roll(time=shift_by, roll_coords=False)
            expected_time = expected_full.isel(time=slice(s, e))
            xr.testing.assert_allclose(result_time, expected_time, rtol=1e-6)
            record("roll", case_label, True)
        except ValueError as e:
            # mpp_halo_exchange() correctly refuses a halo request wider than
            # some rank's own local length rather than silently returning
            # wrong data -- a genuine architectural limit of a single-hop
            # point-to-point exchange (this rank only ever talks to its
            # immediate neighbor, which itself only holds its own local
            # share), not a defect in roll() or mpp_halo_exchange(). At this
            # suite's usual rank counts, "near-half shift" on the small
            # (time=12) fixture is expected to land beyond what any single
            # rank locally holds -- recorded as a declared, expected
            # refusal (SKIP), the same convention used elsewhere in this
            # suite for a declared NotImplementedError, rather than
            # mischaracterized as a failure of roll() itself. See
            # mpi_test_common.is_declared_halo_refusal, shared with every
            # other halo-based op's identical refusal path.
            if is_declared_halo_refusal(e):
                record(
                    "roll",
                    case_label,
                    None,
                    f"declared refusal (expected): {str(e)[:150]}",
                )
            else:
                record(
                    "roll", case_label, False, f"unexpected ValueError: {str(e)[:200]}"
                )
        except Exception as e:
            record("roll", case_label, False, f"{type(e).__name__}: {str(e)[:200]}")
        mpi.comm.barrier()

    # -- evaluate: string-expression evaluation, rank-local ------------------
    try:
        result = local_of(
            dist.evaluate("a + b * 2", a=dist.data["pr"], b=dist.data["t2m"])
        )
        expected = (
            native["pr"].isel(time=slice(start, stop))
            + native["t2m"].isel(time=slice(start, stop)) * 2
        )
        xr.testing.assert_allclose(result, expected, rtol=1e-5)
        record("evaluate", "1d(time), rank-local", True)
    except Exception as e:
        record(
            "evaluate",
            "1d(time), rank-local",
            False,
            f"{type(e).__name__}: {str(e)[:200]}",
        )
    mpi.comm.barrier()

    # -- apply: call a rank-local callable, propagating MPI metadata --------
    try:
        result = dist.apply(lambda d: d * 2.0, dist._prepare())
        local = local_of(result)
        expected = native.isel(time=slice(start, stop)) * 2.0
        xr.testing.assert_allclose(local, expected, rtol=1e-6)
        meta_ok = isinstance(result, MPIXarray) and result.meta is not None
        record(
            "apply",
            "1d(time), rank-local",
            meta_ok,
            "" if meta_ok else "meta did not propagate",
        )
    except Exception as e:
        record(
            "apply",
            "1d(time), rank-local",
            False,
            f"{type(e).__name__}: {str(e)[:200]}",
        )
    mpi.comm.barrier()

    # -- repartition: distribute an object every rank already holds fully ---
    def fill_replicated(n):
        idx = np.arange(n, dtype=np.float64)
        return xr.DataArray(np.sin(idx), dims=("z",), name="v")

    GZ = 21  # uneven for several rank counts, exercising the same path as
    #          get_balanced_bounds's own uneven-split checks
    replicated = fill_replicated(GZ)
    try:
        # repartition() is the MPIXarray method for an object every rank
        # ALREADY holds fully -- unlike distribute_data's root-scatter
        # contract, no data movement is actually required here since
        # every rank starts with the same full array; it just
        # relabels/slices locally.
        result2 = MPIXarray(replicated, mpi, auto_partition=False).repartition(dim="z")
        local2 = local_of(result2)
        m2 = result2.meta
        s2, e2 = m2["start"], m2["stop"]
        expected2 = replicated.isel(z=slice(s2, e2))
        xr.testing.assert_allclose(local2, expected2, rtol=1e-10)
        record("repartition", "1d(z), uneven", True)
    except Exception as e:
        record(
            "repartition", "1d(z), uneven", False, f"{type(e).__name__}: {str(e)[:200]}"
        )
    mpi.comm.barrier()

    # -- align: partition two operands identically ---------------------------
    try:
        other_full = fill_replicated(GZ) * 3.0
        left, right = MPIXarray(replicated, mpi, auto_partition=False).align(
            other_full, dim="z"
        )
        m_left, m_right = left.meta, right.meta
        same_partition = (
            m_left["dims"] == m_right["dims"]
            and m_left["starts"] == m_right["starts"]
            and m_left["stops"] == m_right["stops"]
        )
        s3, e3 = m_left["start"], m_left["stop"]
        expected_left = replicated.isel(z=slice(s3, e3))
        expected_right = (fill_replicated(GZ) * 3.0).isel(z=slice(s3, e3))
        xr.testing.assert_allclose(local_of(left), expected_left, rtol=1e-10)
        xr.testing.assert_allclose(local_of(right), expected_right, rtol=1e-10)
        record("align", "1d(z), matching partitions", same_partition)
    except Exception as e:
        record(
            "align",
            "1d(z), matching partitions",
            False,
            f"{type(e).__name__}: {str(e)[:200]}",
        )
