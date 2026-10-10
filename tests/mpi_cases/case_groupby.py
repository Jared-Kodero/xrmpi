"""Grouped and resampled reductions, weighted means, and small inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from mpi_helpers import *
from xrmpi.mpp.mpp_domains_define import mpp_compute_extent

D = ("t", "y")
nt = 30
times = pd.date_range("2001-01-01", periods=nt, freq="12h")
a = gdata((nt, 7), 11)
an = gdata((nt, 7), 12, nan=True)
x, xn = make(a, D, "t", {"t": times}), make(an, D, "t", {"t": times})
ref = xr.DataArray(a, dims=D, coords={"t": times}, name="v")
refn = xr.DataArray(an, dims=D, coords={"t": times}, name="v")
labels = np.arange(nt) % 4
lo, hi = mpp_compute_extent(nt, comm.rank, comm.size)
group = xr.DataArray(labels, dims="t", name="g")


def T(label, function, want, tol=1e-9):
    case(label, lambda: check(label, gather_global(function()), np.asarray(want), tol))


for red in ("sum", "mean", "min", "max", "count"):
    T(
        f"groupby {red}",
        lambda r=red: getattr(x.groupby("t", labels[lo:hi]), r)(),
        getattr(ref.groupby(group), red)().transpose("g", "y").values,
    )
    # Groups that are all NaN on some ranks (or everywhere) must not poison
    # MPI MIN/MAX, whose result with a NaN operand is undefined.
    T(
        f"groupby {red} nan",
        lambda r=red: getattr(xn.groupby("t", labels[lo:hi]), r)(),
        getattr(refn.groupby(group), red)().transpose("g", "y").values,
    )
    T(
        f"resample 2D {red}",
        lambda r=red: getattr(x.resample("t", "2D"), r)(),
        getattr(ref.resample(t="2D"), red)().values,
    )
    T(
        f"resample 2D {red} nan",
        lambda r=red: getattr(xn.resample("t", "2D"), r)(),
        getattr(refn.resample(t="2D"), red)().values,
    )
# Calendar offsets are labelled by pandas at the right bin edge.
T(
    "resample 1W mean",
    lambda: x.resample("t", "1W").mean(),
    ref.resample(t="1W").mean().values,
)
T(
    "resample 5D sum",
    lambda: x.resample("t", "5D").sum(),
    ref.resample(t="5D").sum().values,
)
T(
    "resample 2W sum",
    lambda: x.resample("t", "2W").sum(),
    ref.resample(t="2W").sum().values,
)

allnan = an.copy()
allnan[labels == 2, 3] = np.nan  # one group entirely missing in one column
xa = make(allnan, D, "t", {"t": times})
ra = xr.DataArray(allnan, dims=D, coords={"t": times}, name="v")
for red in ("min", "max"):
    T(
        f"groupby {red} all-NaN group",
        lambda r=red: getattr(xa.groupby("t", labels[lo:hi]), r)(),
        getattr(ra.groupby(group), red)().transpose("g", "y").values,
    )

w = xr.DataArray(np.abs(gdata((7,), 21)) + 0.1, dims=("y",))
T("weighted mean y", lambda: x.weighted(w).mean("y"), ref.weighted(w).mean("y").values)
tw = xr.DataArray(np.abs(gdata((nt,), 22)) + 0.1, dims=("t",), coords={"t": times})
T(
    "weighted mean t",
    lambda: x.weighted(tw.isel(t=slice(lo, hi))).mean("t"),
    ref.weighted(tw).mean("t").values,
)

# Inputs smaller than the rank count leave some ranks empty.
if comm.size <= 3:
    for size in (3, 4, 5):
        at = gdata((size, 4), 31)
        xt = make(at, D, "t", {"t": times[:size]})
        rt = xr.DataArray(at, dims=D, coords={"t": times[:size]})
        for op in ("sum", "mean", "min", "max", "var"):
            T(
                f"small{size} {op}",
                lambda op=op: getattr(xt, op)(dim="t"),
                getattr(rt, op)(dim="t").values,
            )
        T(f"small{size} cumsum", lambda: xt.cumsum("t"), rt.cumsum("t").values)
        T(f"small{size} ffill", lambda: xt.ffill("t"), rt.ffill("t").values)
finish()
