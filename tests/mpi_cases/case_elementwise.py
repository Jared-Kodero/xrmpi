"""Scans, shifts, stencils, indexing and interpolation along a partitioned axis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi_helpers import *
from xrmpi.mpp.mpp_domains_define import mpp_compute_extent

D = ("t", "y", "x")
a = gdata((17, 5, 6), 1)
an = gdata((17, 5, 6), 2, nan=True)
tc = np.arange(17.0) ** 1.3
x, xn = make(a, D, "t", {"t": tc}), make(an, D, "t", {"t": tc})
ref = xr.DataArray(a, dims=D, coords={"t": tc})
refn = xr.DataArray(an, dims=D, coords={"t": tc})


def T(label, function, want, tol=1e-9):
    case(label, lambda: check(label, gather_global(function()), np.asarray(want), tol))


T("cumsum", lambda: x.cumsum("t"), ref.cumsum("t").values)
T("cumsum nan", lambda: xn.cumsum("t"), refn.cumsum("t").values)
T("cumprod", lambda: x.cumprod("t"), ref.cumprod("t").values)
T("cumsum y", lambda: x.cumsum("y"), ref.cumsum("y").values)
T("median", lambda: x.median("t"), ref.median("t").values)
T(
    "quantile",
    lambda: x.quantile([0.1, 0.5, 0.9], "t"),
    ref.quantile([0.1, 0.5, 0.9], "t").transpose("quantile", "y", "x").values,
)
for n in (1, 2, -1, -3):
    T(f"shift {n}", lambda n=n: x.shift("t", n), ref.shift(t=n).values)
T(
    "shift fill",
    lambda: x.shift("t", 2, fill_value=0.0),
    ref.shift(t=2, fill_value=0.0).values,
)
for n in (1, 2):
    T(f"diff {n}", lambda n=n: x.diff("t", n), ref.diff("t", n).values)
    T(
        f"diff lower {n}",
        lambda n=n: x.diff("t", n, label="lower"),
        ref.diff("t", n, label="lower").values,
    )
for k in (1, 4, -2, 17, 20):
    T(f"roll {k}", lambda k=k: x.roll("t", k), np.roll(a, k, axis=0))
T("pad", lambda: x.pad("t", (2, 3)), ref.pad(t=(2, 3), mode="constant").values)
T(
    "pad value",
    lambda: x.pad("t", (2, 3), constant_values=7.0),
    ref.pad(t=(2, 3), constant_values=7.0).values,
)

holes = an.copy()
holes[:, 0, 0] = np.nan  # permanently missing column
holes[:3, 1, 1] = np.nan  # missing until the first valid value
xh = make(holes, D, "t", {"t": tc})
rh = xr.DataArray(holes, dims=D, coords={"t": tc})
T("ffill", lambda: xn.ffill("t"), refn.ffill("t").values)
T("bfill", lambda: xn.bfill("t"), refn.bfill("t").values)
T("ffill holes", lambda: xh.ffill("t"), rh.ffill("t").values)
T("bfill holes", lambda: xh.bfill("t"), rh.bfill("t").values)
T("ffill limit", lambda: xn.ffill("t", limit=2), refn.ffill("t", limit=2).values)
T("bfill limit", lambda: xn.bfill("t", limit=2), refn.bfill("t", limit=2).values)

for window in (3, 4, 5):
    for center in (True, False):
        for red in ("mean", "sum", "min", "max"):
            T(
                f"rolling {red} w{window} c{center}",
                lambda w=window, c=center, r=red: x.rolling_reduce("t", w, r, center=c),
                getattr(ref.rolling(t=window, center=center), red)().values,
            )
T(
    "rolling min_periods",
    lambda: x.rolling_reduce("t", 5, "mean", min_periods=1),
    ref.rolling(t=5, center=True, min_periods=1).mean().values,
)
for red in ("mean", "sum", "max"):
    T(
        f"coarsen {red}",
        lambda r=red: x.coarsen_reduce("t", 4, r, boundary="trim"),
        getattr(ref.coarsen(t=4, boundary="trim"), red)().values,
    )

T("where", lambda: x.where(x > 0), ref.where(ref > 0).values)
T("where other", lambda: x.where(x > 0, -1.0), ref.where(ref > 0, -1.0).values)
T("binop", lambda: x + x * 2 - 1, a + a * 2 - 1)
T("ufunc", lambda: np.exp(x), np.exp(a))
T("compare", lambda: x > 0.2, a > 0.2)
T("isel slice", lambda: x.isel(t=slice(3, 12)), a[3:12])
T("isel scalar", lambda: x.isel(y=2), a[:, 2])
T("sel", lambda: x.sel(t=slice(tc[2], tc[9])), ref.sel(t=slice(tc[2], tc[9])).values)
T(
    "isel then sum",
    lambda: x.isel(t=slice(2, 11)).sum(dim="y"),
    ref.isel(t=slice(2, 11)).sum("y").values,
)


# Redistribution paths pickle rank-local data; a closure-backed lazy array
# must be evaluated first rather than pickled as a task graph.
def closure_array():
    return xm.new_dataarray(
        mpi,
        lambda s, e: a[s:e],
        D,
        shape=dict(zip(D, a.shape)),
        dim="t",
        dtype=a.dtype,
        coords={"t": tc},
        name="v",
    )


T(
    "sortby closure",
    lambda: closure_array().sortby("t", ascending=False),
    ref.sortby("t", ascending=False).values,
)
T(
    "reindex closure",
    lambda: closure_array().reindex(t=tc[::-1][:10]),
    ref.reindex(t=tc[::-1][:10]).values,
)

# interp: each rank passes its own slice of the target grid.
new = np.linspace(0.0, 40.0, 13)
lo, hi = mpp_compute_extent(len(new), comm.rank, comm.size)
T("interp", lambda: x.interp("t", new[lo:hi]), ref.interp(t=new).values)
T("differentiate", lambda: x.differentiate("t"), ref.differentiate("t").values, 1e-8)
finish()
