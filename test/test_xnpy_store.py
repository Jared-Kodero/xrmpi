"""XNpyStore round trips and refusals.\n\nSerial, not MPI: ``python test/test_xnpy_store.py``\n"""

import shutil

import numpy as np
import xarray as xr

import xgeo as xg
from xgeo.xarray.utils import XNpyStore

base = "/tmp/xnpy_probe"
shutil.rmtree(base, ignore_errors=True)
out = []


def chk(name, fn):
    try:
        out.append((name, fn()))
    except Exception as e:
        out.append((name, f"{type(e).__name__}: {str(e)[:60]}"))


arr = np.arange(24, dtype=np.float64).reshape(4, 6)
da = xr.DataArray(
    arr,
    dims=("t", "x"),
    coords={"t": np.arange(4.0), "x": np.arange(6.0)},
    name="v",
    attrs={"units": "m"},
)
ds = xr.Dataset({"v": da, "w": da * 2}, attrs={"title": "probe"})

chk(
    "ndarray round trip",
    lambda: np.array_equal(
        XNpyStore(f"{base}/a").save(arr) and XNpyStore(f"{base}/a").load(), arr
    ),
)
chk(
    "dataarray round trip",
    lambda: bool(
        XNpyStore(f"{base}/b").save(da) and XNpyStore(f"{base}/b").load().equals(da)
    ),
)
chk(
    "dataset round trip",
    lambda: bool(
        XNpyStore(f"{base}/c").save(ds) and XNpyStore(f"{base}/c").load().equals(ds)
    ),
)
chk("variables() no payload", lambda: XNpyStore(f"{base}/c").variables() == ("v", "w"))
chk(
    "single variable load",
    lambda: bool(XNpyStore(f"{base}/c").load("w").equals(ds["w"])),
)
chk("suffix appended", lambda: XNpyStore(f"{base}/a").path.name.endswith(".xnpy"))
chk("mmap by default", lambda: isinstance(XNpyStore(f"{base}/a").load(), np.memmap))
chk(
    "mmap_mode=None in memory",
    lambda: not isinstance(XNpyStore(f"{base}/a").load(mmap_mode=None), np.memmap),
)
chk("attrs survive", lambda: XNpyStore(f"{base}/b").load().attrs["units"] == "m")


def existing():
    try:
        XNpyStore(f"{base}/a").save(arr)
        return "NOT RAISED"
    except FileExistsError:
        return True


chk("refuses overwrite", existing)
chk(
    "overwrite=True works",
    lambda: bool(XNpyStore(f"{base}/a").save(arr, overwrite=True)),
)


def objdtype():
    try:
        XNpyStore(f"{base}/d").save(np.array([{"a": 1}], dtype=object))
        return "NOT RAISED"
    except TypeError:
        return True


chk("refuses object dtype", objdtype)


def nonfinite():
    try:
        XNpyStore(f"{base}/e").save(xr.DataArray(arr, attrs={"bad": float("nan")}))
        return "NOT RAISED"
    except TypeError:
        return True


chk("refuses non-finite attr", nonfinite)
chk("exported on xgeo", lambda: xg.XNpyStore is XNpyStore)

for n, r in out:
    print(f"{'OK  ' if r is True else 'BUG '} {n}: {'' if r is True else r}")
