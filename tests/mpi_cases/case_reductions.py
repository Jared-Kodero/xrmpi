"""Distributed reductions against serial xarray, with and without NaN."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi_helpers import *

D = ("t", "y", "x")
a = gdata((17, 5, 6), 1)
an = gdata((17, 5, 6), 2, nan=True)
x, xn = make(a, D, "t"), make(an, D, "t")
ref = xr.DataArray(a, dims=D)
refn = xr.DataArray(an, dims=D)

for op in ("sum", "mean", "min", "max", "var", "std", "prod", "any", "all"):
    for dim in ("t", "y", "x"):

        def run(op=op, dim=dim):
            check(
                f"{op} {dim}",
                gather_global(getattr(x, op)(dim=dim)),
                getattr(ref, op)(dim=dim).values,
            )
            if op not in ("any", "all"):
                check(
                    f"{op} {dim} nan",
                    gather_global(getattr(xn, op)(dim=dim)),
                    getattr(refn, op)(dim=dim).values,
                )

        case(f"{op} {dim}", run)


def first_last():
    check("first", gather_global(x.first(dim="t")), ref.isel(t=0).values)
    check("last", gather_global(x.last(dim="t")), ref.isel(t=-1).values)


case("first/last", first_last)
case("sum all", lambda: check("sum all", gather_global(x.sum()), ref.sum().values))
case(
    "sum two dims",
    lambda: check(
        "sum t,y", gather_global(x.sum(dim=("t", "y"))), ref.sum(("t", "y")).values
    ),
)
case(
    "prod skipna=False",
    lambda: check(
        "prod nan propagate",
        gather_global(xn.prod(dim="t", skipna=False)),
        refn.prod(dim="t", skipna=False).values,
    ),
)


# Unnamed arrays: xarray names a dask-backed DataArray after its local graph, which
# differs between ranks and must not enter the collective agreement.
def unnamed():
    u = make(a, D, "t", name=None)
    check("unnamed sum", gather_global(u.sum(dim="y")), ref.sum("y").values)


case("unnamed", unnamed)


def dtypes():
    ai = (gdata((17, 5), 5) * 10).astype(np.int32)
    xi, ri = make(ai, ("t", "y"), "t"), xr.DataArray(ai, dims=("t", "y"))
    for op in ("sum", "mean", "min", "max", "prod", "var"):
        check(
            f"int32 {op}",
            gather_global(getattr(xi, op)(dim="t")),
            getattr(ri, op)(dim="t").values,
        )
    ab = gdata((17, 5), 6) > 0
    xb, rb = make(ab, ("t", "y"), "t"), xr.DataArray(ab, dims=("t", "y"))
    for op in ("sum", "any", "all", "min", "max"):
        check(
            f"bool {op}",
            gather_global(getattr(xb, op)(dim="t")),
            getattr(rb, op)(dim="t").values,
        )
    ac = gdata((17, 5), 7) + 1j * gdata((17, 5), 8)
    xc, rc = make(ac, ("t", "y"), "t"), xr.DataArray(ac, dims=("t", "y"))
    for op in ("sum", "mean", "prod"):
        check(
            f"complex {op}",
            gather_global(getattr(xc, op)(dim="t")),
            getattr(rc, op)(dim="t").values,
        )
    a32 = gdata((17, 5), 9).astype(np.float32)
    x32, r32 = make(a32, ("t", "y"), "t"), xr.DataArray(a32, dims=("t", "y"))
    for op in ("sum", "mean", "var"):
        check(
            f"float32 {op}",
            gather_global(getattr(x32, op)(dim="t")),
            getattr(r32, op)(dim="t").values,
            1e-4,
        )


case("dtypes", dtypes)
finish()
