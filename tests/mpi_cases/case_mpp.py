"""The FMS-style layer: EFP sums, domains, halo exchange, global fields."""

import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi_helpers import *
from xrmpi.mpp.ext_efp import reproducing_prod
from xrmpi.mpp.mpp_do_update import (
    mpp_complete_update_domains,
    mpp_do_update_fold,
    mpp_get_boundary,
    mpp_start_update_domains,
    mpp_update_domains,
)
from xrmpi.mpp.mpp_domains_define import mpp_define_domains
from xrmpi.mpp.mpp_domains_util import (
    mpp_get_domain_components,
    mpp_get_layout,
)
from xrmpi.mpp.mpp_efp import mpp_reproducing_sum
from xrmpi.mpp.mpp_global_field import mpp_global_field
from xrmpi.mpp.mpp_group_update import (
    mpp_add_to_group_update,
    mpp_create_group_update,
    mpp_do_group_update,
)
from xrmpi.mpp.mpp_parameter import FOLD_NORTH_EDGE

rng = np.random.default_rng(5)


def agree(label, ok):
    """Record ``ok`` only if every rank holds it."""
    check(label, np.array([all(comm.allgather(bool(ok)))]), np.array([True]))


def slab(n):
    return slice((n * comm.rank) // comm.size, (n * (comm.rank + 1)) // comm.size)


# --- extended fixed point -------------------------------------------------
g = rng.normal(size=1000) * 10.0 ** rng.integers(-8, 8, size=1000)
got = float(np.asarray(mpp_reproducing_sum(g[slab(1000)], comm)).ravel()[0])
check("efp matches fsum", got, math.fsum(g), 1e-13)
g2 = rng.normal(size=(40, 3))
check(
    "efp axis=0",
    mpp_reproducing_sum(g2[slab(40)], comm, axis=0),
    [math.fsum(g2[:, j]) for j in range(3)],
    1e-14,
)
p = rng.uniform(0.5, 1.5, size=30)
check("reproducing_prod", reproducing_prod(p[slab(30)], comm), np.prod(p), 1e-12)
check(
    "reproducing_prod int",
    reproducing_prod(np.arange(1, 13)[slab(12)], comm),
    np.prod(np.arange(1, 13)),
    0,
)

# --- halo buffers must not alias the pool ---------------------------------
dom1 = mpp_define_domains(mpi, {"t": 12 * comm.size}, "t")
first = np.full((12, 3), float(comm.rank))
second = np.full((12, 3), 100.0 + comm.rank)
before, _after, _, _ = mpp_complete_update_domains(
    mpp_start_update_domains(first, dom1, "t", 0, before=2, after=2)
)
snapshot = {k: v.copy() for k, v in before.items()}
mpp_complete_update_domains(
    mpp_start_update_domains(second, dom1, "t", 0, before=2, after=2)
)
agree(
    "halo buffers survive the next exchange",
    all(np.array_equal(before[k], snapshot[k]) for k in before),
)

# --- two-axis domains -----------------------------------------------------
NY, NX = 11, 13
A = np.arange(NY * NX, dtype=float).reshape(NY, NX)
dom = mpp_define_domains(mpi, {"y": NY, "x": NX}, ("y", "x"))
ys, ye, xs, xe = dom.starts["y"], dom.stops["y"], dom.starts["x"], dom.stops["x"]
loc = A[ys:ye, xs:xe].copy()

check("layout", np.array(mpp_get_layout(dom)), np.array(dom.cart["grid_shape"]))
check("layout 1-D", np.array(mpp_get_layout(dom1)), np.array([comm.size]))
check("global_field y", mpp_global_field(loc, dom, "y", 0), A[:, xs:xe])
check("global_field x", mpp_global_field(loc, dom, "x", 1), A[ys:ye, :])
parts = mpp_get_domain_components(dom)
agree(
    "components use the per-axis communicator",
    parts["y"].comm.size == dom.cart["grid_shape"][0]
    and parts["x"].comm.size == dom.cart["grid_shape"][1],
)


def reference_pad(halo, cyclic):
    out = A
    for axis, (low, high) in enumerate(halo):
        width = [(0, 0), (0, 0)]
        width[axis] = (low, high)
        out = (
            np.pad(out, width, mode="wrap")
            if cyclic[axis]
            else np.pad(out, width, mode="constant", constant_values=np.nan)
        )
    return out


for halo in (((1, 1), (1, 1)), ((2, 1), (0, 2)), ((1, 0), (0, 0))):
    for cyclic in ((False, False), (True, True), (True, False)):

        def update(halo=halo, cyclic=cyclic):
            d = dataclasses.replace(dom, cyclic={"y": cyclic[0], "x": cyclic[1]})
            padded, received = mpp_update_domains(
                loc, d, ("y", "x"), halo={"y": halo[0], "x": halo[1]}
            )
            expect = reference_pad(halo, cyclic)[
                ys : ye + sum(halo[0]), xs : xe + sum(halo[1])
            ]
            y0 = halo[0][0] - received["y"][0]
            x0 = halo[1][0] - received["x"][0]
            expect = expect[y0 : y0 + padded.shape[0], x0 : x0 + padded.shape[1]]
            keep = ~np.isnan(expect)
            agree(
                f"update2d {halo} {cyclic}",
                padded.shape == expect.shape
                and np.array_equal(padded[keep], expect[keep]),
            )

        case(f"update2d {halo} {cyclic}", update)

padded, _, _ = mpp_update_domains(loc, dom, "y", 0, before=2, after=2, periodic=True)
agree(
    "1-D periodic update",
    np.array_equal(
        padded, np.pad(A, ((2, 2), (0, 0)), mode="wrap")[ys : ye + 4, xs:xe]
    ),
)

group = mpp_create_group_update(dom, "y", 0, before=1, after=1)
mpp_add_to_group_update(group, "a", loc)
mpp_add_to_group_update(group, "b", loc * 2)
out, low, high = mpp_do_group_update(group)
agree(
    "group update",
    out["a"].shape[0] == loc.shape[0] + low + high
    and np.array_equal(out["b"], out["a"] * 2),
)

lower, upper = mpp_get_boundary(loc, dom, "y", 0)
agree("get_boundary", (lower is None) == (ys == 0) and (upper is None) == (ye == NY))

fold = dataclasses.replace(dom, fold=FOLD_NORTH_EDGE)
folded = mpp_do_update_fold(loc, fold, "y", 0, "x", 1, width=1)
agree(
    "north fold",
    np.array_equal(folded[-1], A[NY - 1, ::-1][xs:xe])
    if ye == NY
    else np.array_equal(folded, loc),
)
finish()
