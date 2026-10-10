"""Reductions and stencils on two- and three-axis process grids."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi_helpers import *

D = ("t", "y", "x")
a = gdata((13, 10, 9), 3)
an = gdata((13, 10, 9), 4, nan=True)
tc = np.arange(13.0)
ref = xr.DataArray(a, dims=D, coords={"t": tc})
refn = xr.DataArray(an, dims=D, coords={"t": tc})


def T(label, function, want, tol=1e-9):
    case(label, lambda: check(label, gather_global(function()), np.asarray(want), tol))


for parts in (("y", "x"), ("t", "y"), ("t", "y", "x")):
    x, xn = make(a, D, parts, {"t": tc}), make(an, D, parts, {"t": tc})
    tag = "".join(parts)
    for dim in D:
        for op in ("sum", "mean", "min", "max", "var", "prod"):
            T(
                f"[{tag}] {op} {dim}",
                lambda op=op, dim=dim: getattr(x, op)(dim=dim),
                getattr(ref, op)(dim=dim).values,
            )
        T(
            f"[{tag}] sum nan {dim}",
            lambda dim=dim: xn.sum(dim=dim),
            refn.sum(dim=dim).values,
        )
        T(
            f"[{tag}] prod nan {dim}",
            lambda dim=dim: xn.prod(dim=dim),
            refn.prod(dim=dim).values,
        )
    T(f"[{tag}] sum all", lambda: x.sum(), ref.sum().values)
    T(f"[{tag}] sum t,y", lambda: x.sum(dim=("t", "y")), ref.sum(("t", "y")).values)
    T(f"[{tag}] cumsum y", lambda: x.cumsum("y"), ref.cumsum("y").values)
    T(
        f"[{tag}] rolling y",
        lambda: x.rolling_reduce("y", 3, "mean"),
        ref.rolling(y=3, center=True).mean().values,
    )
    T(f"[{tag}] diff x", lambda: x.diff("x"), ref.diff("x").values)
    T(f"[{tag}] shift y", lambda: x.shift("y", 2), ref.shift(y=2).values)
    T(f"[{tag}] roll x", lambda: x.roll("x", 3), np.roll(a, 3, axis=2))
    T(f"[{tag}] ffill y", lambda: xn.ffill("y"), refn.ffill("y").values)
    T(f"[{tag}] median y", lambda: x.median("y"), ref.median("y").values)
    T(f"[{tag}] where", lambda: x.where(x > 0), ref.where(ref > 0).values)
    T(f"[{tag}] binop", lambda: x * x + 1, a * a + 1)
    # An indexing step changes the global sizes; later collectives must still
    # use the process grid the data actually sits on.
    T(
        f"[{tag}] isel t then sum y",
        lambda: x.isel(t=slice(2, 11)).sum(dim="y"),
        ref.isel(t=slice(2, 11)).sum("y").values,
    )
    T(
        f"[{tag}] isel y then sum t",
        lambda: x.isel(y=slice(1, 8)).sum(dim="t"),
        ref.isel(y=slice(1, 8)).sum("t").values,
    )
finish()
