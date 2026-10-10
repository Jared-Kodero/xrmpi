"""open_dataset, partition, new_dataset and serial NetCDF output."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from mpi_helpers import *

tmp = comm.bcast(tempfile.mkdtemp() if comm.rank == 0 else None)
nt = 40
times = pd.date_range("2000-01-01", periods=nt, freq="6h")
a = gdata((nt, 6, 5), 77)
ds = xr.Dataset(
    {"pr": (("time", "lat", "lon"), a), "tas": (("time", "lat", "lon"), a * 2 + 1)},
    coords={"time": times, "lat": np.arange(6.0), "lon": np.arange(5.0)},
)
path = os.path.join(tmp, "in.nc")
if comm.rank == 0:
    ds.to_netcdf(path)
comm.Barrier()

for part in ("time", "lat", "auto"):

    def opened(part=part):
        d = xm.open_dataset(path, mpi, partition_dim=part, log_partitions=False)
        check(f"open[{part}] values", gather_global(d["pr"]), a)
        check(
            f"open[{part}] sum",
            gather_global(d["pr"].sum(dim="time")),
            ds["pr"].sum("time").values,
        )
        check(
            f"open[{part}] mean",
            gather_global(d["tas"].mean()),
            ds["tas"].mean().values,
        )
        check(
            f"open[{part}] dataset mean",
            gather_global(d.mean(dim="time")["pr"]),
            ds["pr"].mean("time").values,
        )

    case(f"open {part}", opened)


def chunked():
    d = xm.open_dataset(
        path, mpi, partition_dim="time", chunks={"time": 7}, log_partitions=False
    )
    check(
        "open chunks",
        gather_global(d["pr"].sum(dim="time")),
        ds["pr"].sum("time").values,
    )


case("open chunks", chunked)


def partitioned():
    p = xm.partition(ds if comm.rank == 0 else None, mpi, dim="time")
    check("partition dataset", gather_global(p["pr"]), a)
    q = xm.partition(ds["pr"] if comm.rank == 0 else None, mpi, dim="lat")
    check("partition dataarray", gather_global(q), a)


case("partition", partitioned)


def new_ds():
    u, v = gdata((9, 4), 1), gdata((9, 4), 2)
    d = xm.new_dataset(
        mpi,
        {
            "u": (("t", "x"), Filler(u, ("t", "x"), ["t"])),
            "v": (("t", "x"), Filler(v, ("t", "x"), ["t"])),
        },
        sizes={"t": 9, "x": 4},
        dim="t",
        log_partitions=False,
    )
    check("new_dataset u", gather_global(d["u"]), u)
    check("new_dataset v", gather_global(d["v"]), v)


case("new_dataset", new_ds)


def serial_write():
    if comm.rank != 0:
        return
    out = os.path.join(tmp, "app.nc")
    xm.to_netcdf(
        ds.isel(time=slice(0, 30)),
        out,
        unlimited_dim="time",
        show_progress=False,
        batch_size=7,
    )
    xm.append_to_netcdf(ds.isel(time=slice(30, 40)), out, dim="time")
    got = xr.open_dataset(out)
    check("append roundtrip", got["pr"].values, a)
    check(
        "append time",
        got["time"].values.astype("datetime64[ns]").astype("int64"),
        times.values.astype("datetime64[ns]").astype("int64"),
    )
    # The progress bar writes to the stream it is given.
    import io

    sink = io.StringIO()
    xm.to_netcdf(
        ds,
        os.path.join(tmp, "bar.nc"),
        unlimited_dim="time",
        batch_size=5,
        show_progress=True,
        stdout=sink,
    )
    check("progress written", np.array([len(sink.getvalue()) > 0]), np.array([True]))


case("serial write", serial_write)
finish()
