"""Construction correctness: open_distributed_dataset, create_distributed_dataarray,
create_distributed_dataset -- single- and multi-dimensional, even and uneven
partitions, with explicit reconstruction (exact, non-overlapping global
coverage) checks.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import xarray as xr
from xgeo.xarray.core import MPIXarray

import xgeo as xg
from xgeo import MPIContext
from xrmpi.test.mock_dataset import PATH
from xrmpi.test.mpi_test_common import Fixtures, local_of, record

mpi = MPIContext()


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d

    # -- open_distributed_dataset, one distributed dimension -----------------------
    start, stop = dist.meta["start"], dist.meta["stop"]
    try:
        xr.testing.assert_allclose(
            local_of(dist), native.isel(time=slice(start, stop)), rtol=1e-6
        )
        record("open_distributed_dataset", "1d(time)", True)
    except Exception as e:
        record("open_distributed_dataset", "1d(time)", False, str(e)[:200])

    # -- balanced-bounds boundary cases: even, remainder>0, length<ranks,
    #    length 0, length 1 -------------------------------------------------
    def check_bounds_case(global_size: int, label: str) -> None:
        def fill(a, b):
            idx = np.arange(a, b)
            return (idx[:, None] * 100 + np.arange(3)[None, :]).astype(np.float64)

        da = xg.create_distributed_dataarray(
            mpi,
            fill,
            dims=("x", "y"),
            shape={"x": global_size, "y": 3},
            dim="x",
            log_partitions=False,
        )
        s, e = da.meta["start"], da.meta["stop"]
        local = local_of(da)
        idx_global = np.arange(global_size)
        expected_global = (idx_global[:, None] * 100 + np.arange(3)[None, :]).astype(
            np.float64
        )
        shape_ok = local.shape[0] == (e - s)
        val_ok = (
            np.array_equal(local.values, expected_global[s:e])
            if local.shape[0]
            else True
        )

        bounds = mpi.comm.gather((s, e), root=0)
        coverage_ok = None
        if mpi.comm.rank == 0:
            bounds_sorted = sorted(bounds)
            coverage_ok = (
                bounds_sorted[0][0] == 0 and bounds_sorted[-1][1] == global_size
            )
            for i in range(1, len(bounds_sorted)):
                if bounds_sorted[i][0] != bounds_sorted[i - 1][1]:
                    coverage_ok = False

        all_ok = mpi.comm.gather(shape_ok and val_ok, root=0)
        if mpi.comm.rank == 0:
            record("get_balanced_bounds", label, all(all_ok) and bool(coverage_ok))

    check_bounds_case(21, "uneven, remainder>0")
    check_bounds_case(20, "even, control")
    check_bounds_case(2, "length < ranks")
    check_bounds_case(1, "length == 1")
    check_bounds_case(0, "length == 0")
    mpi.comm.barrier()

    # -- multi-dim analog of the above: both partition dimensions
    #    deliberately shorter than the rank count, forcing compute_layout
    #    to give at least one axis more divisions than its own global
    #    length can fill without an empty rank (e.g. length 1 split
    #    across 2+ divisions). Checks the same properties as the 1D
    #    cases -- correct values on owning ranks, genuinely empty (not
    #    corrupted) data elsewhere, exact non-overlapping coverage --
    #    but for a degenerate Cartesian grid rather than a degenerate
    #    single dimension. -----------------------------------------------
    def fill_degenerate(x0, x1, y0, y1):
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        return (xs[:, None] * 10 + ys[None, :]).astype(np.float64)

    DGX, DGY = 1, 1
    da_deg = xg.create_distributed_dataarray(
        mpi,
        fill_degenerate,
        dims=("x", "y"),
        shape={"x": DGX, "y": DGY},
        dim=("x", "y"),
        log_partitions=False,
        name="v",
    )
    m_deg = da_deg.meta
    dxs, dxe = m_deg["starts"]["x"], m_deg["stops"]["x"]
    dys, dye = m_deg["starts"]["y"], m_deg["stops"]["y"]
    local_deg = local_of(da_deg)
    expected_deg_global = (
        np.arange(DGX)[:, None] * 10 + np.arange(DGY)[None, :]
    ).astype(np.float64)
    shape_ok = local_deg.shape == (dxe - dxs, dye - dys)
    val_ok = (
        np.array_equal(local_deg.values, expected_deg_global[dxs:dxe, dys:dye])
        if (dxe > dxs and dye > dys)
        else local_deg.size == 0
    )
    bounds_deg = mpi.comm.gather((dxs, dxe, dys, dye, shape_ok and val_ok), root=0)
    if mpi.comm.rank == 0:
        grid = np.zeros((DGX, DGY), dtype=int)
        all_ok = True
        any_empty = False
        for b in bounds_deg:
            if b[1] > b[0] and b[3] > b[2]:
                grid[b[0] : b[1], b[2] : b[3]] += 1
            else:
                any_empty = True
            all_ok = all_ok and b[4]
        # At rank counts > 1 this global 1x1 array cannot fill every
        # rank without overlap, so at least one rank MUST come back
        # empty -- if none did, the empty-rank path was never actually
        # exercised and this check would be vacuous.
        degenerate_exercised = any_empty or mpi.comm.size == 1
        record(
            "create_distributed_dataarray",
            "2d(x,y), degenerate (both dims < ranks)",
            all_ok and bool(np.all(grid == 1)) and degenerate_exercised,
        )
    mpi.comm.barrier()

    # -- weighted (Allreduce sum/count) mean under a genuinely uneven
    #    split, vs a naive-average ground truth ----------------------------
    def fill_uneven(a, b):
        idx = np.arange(a, b)
        return (
            np.sin(idx.astype(np.float64))[:, None] * (idx[:, None] + 1)
            + np.arange(3)[None, :]
        ).astype(np.float64)

    GLOBAL = 21  # uneven for any rank count in {2,3,4,5,6} except divisors
    da_uneven = xg.create_distributed_dataarray(
        mpi,
        fill_uneven,
        dims=("x", "y"),
        shape={"x": GLOBAL, "y": 3},
        dim="x",
        log_partitions=False,
        name="v",
    )
    gmean = da_uneven.mean(dim="x")
    idx_global = np.arange(GLOBAL)
    full = (
        np.sin(idx_global.astype(np.float64))[:, None] * (idx_global[:, None] + 1)
        + np.arange(3)[None, :]
    ).astype(np.float64)
    expected_mean = full.mean(axis=0)
    try:
        if isinstance(gmean, MPIXarray) and gmean.meta is not None:
            s3, e3 = gmean.meta["start"], gmean.meta["stop"]
            ok = np.allclose(local_of(gmean).values, expected_mean[s3:e3], rtol=1e-10)
        else:
            ok = np.allclose(np.asarray(local_of(gmean)), expected_mean, rtol=1e-10)
        all_ok = mpi.comm.gather(ok, root=0)
        if mpi.comm.rank == 0:
            record("mean", "uneven partition, weighted vs naive-average", all(all_ok))
    except Exception as e:
        record(
            "mean", "uneven partition, weighted vs naive-average", False, str(e)[:200]
        )
    mpi.comm.barrier()

    # -- multiple distributed dimensions -------------------------------
    m2 = dist2d.meta
    lat_s, lat_e = m2["starts"]["lat"], m2["stops"]["lat"]
    lon_s, lon_e = m2["starts"]["lon"], m2["stops"]["lon"]
    try:
        expected = native.isel(lat=slice(lat_s, lat_e), lon=slice(lon_s, lon_e))
        xr.testing.assert_allclose(local_of(dist2d), expected, rtol=1e-6)
        ok = True
    except Exception:
        ok = False
    bounds2d = mpi.comm.gather((lat_s, lat_e, lon_s, lon_e, ok), root=0)
    if mpi.comm.rank == 0:
        lat_n, lon_n = native.sizes["lat"], native.sizes["lon"]
        grid = np.zeros((lat_n, lon_n), dtype=int)
        all_ok = True
        for b in bounds2d:
            grid[b[0] : b[1], b[2] : b[3]] += 1
            all_ok = all_ok and b[4]
        record(
            "open_distributed_dataset",
            "2d(lat,lon), reconstruction",
            all_ok and bool(np.all(grid == 1)),
        )
    mpi.comm.barrier()

    GX, GY = 19, 36

    def fill2d(x_start, x_stop, y_start, y_stop):
        xs = np.arange(x_start, x_stop)
        ys = np.arange(y_start, y_stop)
        return (xs[:, None] * 1000 + ys[None, :]).astype(np.float64)

    da2d = xg.create_distributed_dataarray(
        mpi,
        fill2d,
        dims=("x", "y"),
        shape={"x": GX, "y": GY},
        dim=("x", "y"),
        log_partitions=False,
        name="v",
    )
    m3 = da2d.meta
    xs, xe = m3["starts"]["x"], m3["stops"]["x"]
    ys, ye = m3["starts"]["y"], m3["stops"]["y"]
    expected_global = (np.arange(GX)[:, None] * 1000 + np.arange(GY)[None, :]).astype(
        np.float64
    )
    local2d = local_of(da2d)
    ok_val = np.array_equal(local2d.values, expected_global[xs:xe, ys:ye])
    ok_shape = local2d.shape == (xe - xs, ye - ys)
    bounds2 = mpi.comm.gather((xs, xe, ys, ye, ok_val and ok_shape), root=0)
    if mpi.comm.rank == 0:
        grid = np.zeros((GX, GY), dtype=int)
        all_ok = True
        for b in bounds2:
            grid[b[0] : b[1], b[2] : b[3]] += 1
            all_ok = all_ok and b[4]
        record(
            "create_distributed_dataarray",
            "2d(x,y), reconstruction",
            all_ok and bool(np.all(grid == 1)),
        )
    mpi.comm.barrier()

    def fill_x_only(a, b):
        return np.arange(a, b, dtype=np.float64) * 7.0

    def fill_const():
        return np.full((3,), 42.0)

    ds2d = xg.create_distributed_dataset(
        mpi,
        data_vars={
            "full2d": (("x", "y"), fill2d),
            "x_only": (("x",), fill_x_only),
            "const": (("z",), fill_const),
        },
        sizes={"x": GX, "y": GY, "z": 3},
        dim=("x", "y"),
        log_partitions=False,
    )
    local_ds = local_of(ds2d)
    ok = (
        np.array_equal(local_ds["full2d"].values, expected_global[xs:xe, ys:ye])
        and np.array_equal(
            local_ds["x_only"].values, np.arange(xs, xe, dtype=np.float64) * 7.0
        )
        and np.array_equal(local_ds["const"].values, np.full((3,), 42.0))
        and local_ds.sizes["z"] == 3
    )
    all_ok = mpi.comm.gather(ok, root=0)
    if mpi.comm.rank == 0:
        record(
            "create_distributed_dataset",
            "2d(x,y), mixed both/one/no-partition-dim vars",
            all(all_ok),
        )
    mpi.comm.barrier()

    correct_da = xr.DataArray(np.zeros((xe - xs, ye - ys)), dims=("x", "y"))
    try:
        check_ds = xg.create_distributed_dataset(
            mpi,
            data_vars={"pre_built": correct_da, "other": (("x", "y"), fill2d)},
            sizes={"x": GX, "y": GY},
            dim=("x", "y"),
            log_partitions=False,
        )
        local_of(check_ds)
        ok_a = True
    except Exception:
        ok_a = False

    wrong_y = xr.DataArray(np.zeros((xe - xs, (ye - ys) + 1)), dims=("x", "y"))
    try:
        xg.create_distributed_dataset(
            mpi,
            data_vars={"wrong_y": wrong_y, "other": (("x", "y"), fill2d)},
            sizes={"x": GX, "y": GY},
            dim=("x", "y"),
            log_partitions=False,
        )
        ok_b = False  # should have raised
    except ValueError:
        ok_b = True
    except Exception:
        ok_b = False

    wrong_x = xr.DataArray(np.zeros(((xe - xs) + 1, ye - ys)), dims=("x", "y"))
    try:
        xg.create_distributed_dataset(
            mpi,
            data_vars={"wrong_x": wrong_x, "other": (("x", "y"), fill2d)},
            sizes={"x": GX, "y": GY},
            dim=("x", "y"),
            log_partitions=False,
        )
        ok_c = False  # should have raised
    except ValueError:
        ok_c = True
    except Exception:
        ok_c = False

    all_ok = mpi.comm.gather((ok_a, ok_b, ok_c), root=0)
    if mpi.comm.rank == 0:
        record(
            "create_distributed_dataset",
            "2d(x,y), DataArray shape validation",
            all(all(t) for t in all_ok),
        )
    mpi.comm.barrier()

    # -- to_netcdf(parallel=True): real write-reread-compare round trip,
    #    single-dim (auto-chunked) and multi-dim (explicit chunks
    #    required -- auto-chunk inference isn't multi-dim generalized
    #    yet, and says so rather than guessing) -------------------------
    parallel_dir = PATH.parent / f"{PATH.stem}_parallel"
    if mpi.comm.rank == 0:
        parallel_dir.mkdir(parents=True, exist_ok=False)
    mpi.comm.barrier()

    def check_parallel_write(case: str, path, write_fn) -> None:
        ok = True
        msg = ""
        try:
            write_fn()
        except Exception as e:
            ok = False
            msg = f"{type(e).__name__}: {str(e)[:180]}"

        statuses = mpi.comm.gather((ok, msg), root=0)
        if mpi.comm.rank == 0:
            all_write_ok = all(status[0] for status in statuses)
            if not all_write_ok:
                msg = next(status[1] for status in statuses if not status[0])
        else:
            all_write_ok = None
        all_write_ok = mpi.comm.bcast(all_write_ok, root=0)

        read_ok = True
        if all_write_ok and mpi.comm.rank == 0:
            try:
                written = xr.open_dataset(path).load()
                xr.testing.assert_allclose(written, native, rtol=1e-6)
            except Exception as e:
                read_ok = False
                msg = f"{type(e).__name__}: {str(e)[:180]}"
        read_ok = mpi.comm.bcast(read_ok, root=0)
        if mpi.comm.rank == 0:
            record(
                "to_netcdf(parallel=True)",
                case,
                bool(all_write_ok and read_ok),
                "" if all_write_ok and read_ok else msg,
            )
        mpi.comm.barrier()

    try:
        out_1d = parallel_dir / "parallel_1d.nc"
        check_parallel_write(
            "1d(time)",
            out_1d,
            lambda: xg.to_netcdf(
                dist._prepare(),
                out_1d,
                mpi,
                parallel=True,
                allow_serial=(mpi.comm.size == 1),
            ),
        )

        out_2d = parallel_dir / "parallel_2d.nc"
        # One chunk per rank's compute domain, not one chunk spanning the
        # whole global array. The old global-shape version asked for a
        # 2.8 GiB chunk for "pr" at production resolution, which exceeds
        # what a parallel NetCDF-4 write can safely emit and is correctly
        # rejected by chunks._validate_explicit_chunk_bytes -- the failure
        # was in the fixture, not the writer. Following the decomposition
        # is also what fms2_io does when it registers a domain-decomposed
        # variable.
        meta2d = dist2d.meta
        grid_shape = dict(
            zip(meta2d["dims"], meta2d["cart"]["grid_shape"], strict=True)
        )
        chunk_length = {
            d: math.ceil(int(meta2d["global_sizes"][d]) / grid_shape[d])
            for d in meta2d["dims"]
        }
        var_chunks = {
            name: tuple(
                chunk_length.get(d, int(native.sizes[d])) for d in native[name].dims
            )
            for name in native.data_vars
        }
        check_parallel_write(
            "2d(lat,lon), explicit chunks",
            out_2d,
            lambda: xg.to_netcdf(
                dist2d._prepare(),
                out_2d,
                mpi,
                parallel=True,
                chunks=var_chunks,
                allow_serial=(mpi.comm.size == 1),
            ),
        )

        out_2d_auto = parallel_dir / "parallel_2d_auto.nc"
        check_parallel_write(
            "2d(lat,lon), auto chunks",
            out_2d_auto,
            lambda: xg.to_netcdf(
                dist2d._prepare(),
                out_2d_auto,
                mpi,
                parallel=True,
                allow_serial=(mpi.comm.size == 1),
            ),
        )
    finally:
        mpi.comm.barrier()
        if mpi.comm.rank == 0:
            shutil.rmtree(parallel_dir, ignore_errors=True)
        mpi.comm.barrier()
