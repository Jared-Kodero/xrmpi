"""Parallel NetCDF output compared with files written by native xarray."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import netCDF4
import pandas as pd
from mpi_helpers import *
from xrmpi.mpi.diagnostics import MPIError

tmp = comm.bcast(tempfile.mkdtemp() if comm.rank == 0 else None)
root = comm.rank == 0
nt = 23
time = pd.date_range("2001-03-04 06:00", periods=nt, freq="6h")
noleap = xr.date_range(
    "2001-01-01", periods=nt, freq="D", calendar="noleap", use_cftime=True
)
a = gdata((nt, 6, 5), 3, nan=True)
yy, xx = np.meshgrid(np.arange(6.0), np.arange(5.0), indexing="ij")
ds = xr.Dataset(
    {
        "pr": (("time", "y", "x"), a, {"units": "mm"}),
        "mask": (("time", "y", "x"), a > 0),
        "count": (("time", "y", "x"), (np.nan_to_num(np.abs(a)) * 10).astype("int32")),
        "clim": (("y", "x"), gdata((6, 5), 4)),
        "level": ((), 5.0),
        "dt": (("time",), (np.arange(nt) * 3600e9).astype("timedelta64[ns]")),
    },
    coords={
        "time": time,
        "y": np.arange(6.0),
        "x": np.arange(5.0),
        "lat": (("y", "x"), yy * 2),
        "lon": (("y", "x"), xx * 3),
    },
    attrs={"title": "case"},
)
ds_cf = ds[["pr"]].assign_coords(time=("time", noleap))
packed = {
    "pr": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 1.0,
        "_FillValue": -999,
    }
}
ENCODED_KEYS = (
    "units",
    "calendar",
    "dtype",
    "_FillValue",
    "scale_factor",
    "add_offset",
)

# Collective writes of uncompressed variables fail on some MPI-IO/HDF5 builds, so the
# coordinates, which xarray leaves uncompressed, are compressed as well.
ENC = {name: {"zlib": True} for name in ("time", "y", "x", "lat", "lon", "k")}


def same(label: str, got, want) -> None:
    """Record whether two arbitrary Python values are equal."""
    equal = repr(got) == repr(want)  # NaN fill values compare unequal otherwise
    check(
        label if equal else f"{label}: {got!r} vs {want!r}",
        np.array([equal]),
        np.array([True]),
    )


def compare(label: str, out: str, data: xr.Dataset, **native_kwargs) -> None:
    """Assert that ``out`` decodes like, and stores like, a native xarray file."""
    reference = os.path.join(tmp, f"native_{label}.nc")
    data.to_netcdf(reference, **native_kwargs)
    with xr.open_dataset(out) as got, xr.open_dataset(reference) as want:
        xr.testing.assert_identical(got, want)
        for name in want.variables:
            same(
                f"{label} {name} encoding",
                {k: got[name].encoding.get(k) for k in ENCODED_KEYS},
                {k: want[name].encoding.get(k) for k in ENCODED_KEYS},
            )
    with netCDF4.Dataset(out) as mine, netCDF4.Dataset(reference) as theirs:
        for name, variable in theirs.variables.items():
            same(
                f"{label} {name} raw dtype and dims",
                (mine[name].dtype, mine[name].dimensions),
                (variable.dtype, variable.dimensions),
            )


def run(label: str, source, *, data=ds, native_kwargs=None, **kwargs) -> None:
    """Write ``source`` in parallel and compare on rank 0."""
    out = os.path.join(tmp, f"{label}.nc")
    kwargs.setdefault("encoding", ENC)
    if isinstance(source, xr.Dataset):
        xm.to_netcdf(source, out, mpi, parallel=True, allow_serial=True, **kwargs)
    else:
        source.to_netcdf(out, parallel=True, allow_serial=True, **kwargs)
    comm.Barrier()
    if root:
        compare(label, out, data, **(native_kwargs or {}))
    comm.Barrier()


for part in ("time", "y", "x"):
    case(
        f"root-owned {part}",
        lambda part=part: run(f"root_{part}", ds, partition_dim=part),
    )
for dims in ("time", "y", ["y", "x"], ["time", "y"], ["time", "y", "x"]):
    label = "_".join(dims) if isinstance(dims, list) else dims
    case(
        f"distributed {label}",
        lambda dims=dims, label=label: run(
            f"d_{label}", xm.partition(ds if root else None, mpi, dim=dims)
        ),
    )
case(
    "root-owned dask",
    lambda: run("root_dask", ds.chunk({"time": 5}), data=ds, partition_dim="time"),
)
case(
    "dask local",
    lambda: run(
        "d_dask", xm.partition(ds.chunk({"time": 4}) if root else None, mpi, dim="time")
    ),
)
case(
    "unlimited time",
    lambda: run(
        "d_unl",
        xm.partition(ds if root else None, mpi, dim="time"),
        unlimited_dims="time",
        native_kwargs={"unlimited_dims": ["time"]},
    ),
)
case(
    "cftime",
    lambda: run(
        "d_cf", xm.partition(ds_cf if root else None, mpi, dim="time"), data=ds_cf
    ),
)
case(
    "packed integers with missing values",
    lambda: run(
        "d_packed",
        xm.partition(ds if root else None, mpi, dim="time"),
        encoding={**ENC, **packed},
        native_kwargs={"encoding": packed},
    ),
)
case(
    "explicit chunks",
    lambda: run(
        "d_chunks",
        xm.partition(ds if root else None, mpi, dim="time"),
        encoding={**ENC, "pr": {"zlib": True, "chunksizes": (1, 3, 5)}},
    ),
)


def chunks_stored() -> None:
    if root:
        with netCDF4.Dataset(os.path.join(tmp, "d_chunks.nc")) as nc:
            check("explicit chunks stored", np.array(nc["pr"].chunking()), [1, 3, 5])


case("chunks stored", chunks_stored)


def zero_length() -> None:
    """More ranks than elements leaves some partitions empty."""
    small = xr.Dataset(
        {"v": (("k", "x"), gdata((2, 3), 8)), "w": (("x",), np.arange(3.0))},
        coords={"k": [10.0, 20.0], "x": np.arange(3.0)},
    )
    run("d_small", xm.partition(small if root else None, mpi, dim="k"), data=small)


case("zero-length partitions", zero_length)


def fails_on_every_rank() -> None:
    """A distributed string variable is rejected on all ranks, without hanging."""
    names = xr.Dataset(
        {"s": (("k",), np.array(["a", "bb", "ccc", "d", "e"]))},
        coords={"k": np.arange(5)},
    )
    distributed = xm.partition(names if root else None, mpi, dim="k")
    try:
        distributed.to_netcdf(
            os.path.join(tmp, "bad.nc"), parallel=True, allow_serial=True
        )
    except MPIError as exc:
        raised = "cannot write distributed variable 's'" in str(exc)
    else:
        raised = False
    check("string variable rejected on all ranks", np.array([raised]), np.array([True]))


case("collective failure", fails_on_every_rank)
finish()
