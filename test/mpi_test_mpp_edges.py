"""Edge cases the main suite does not reach: empty compute domains and
oversized halo requests.

Run standalone: ``mpirun -n 4 python test/mpi_test_mpp_edges.py``
"""

import numpy as np

from xgeo import MPIContext
from xrmpi.mpp.mpp_do_update import HaloWidthError, mpp_update_domains
from xrmpi.mpp.mpp_domains import Domain
from xrmpi.mpp.mpp_domains_define import mpp_compute_extent
from xrmpi.mpp.mpp_global_reduce import (
    mpp_global_max,
    mpp_global_min,
    mpp_global_sum,
)

mpi = MPIContext()
comm = mpi.comm
out = []

# Bug 1: a rank owning nothing must not deadlock a global extremum.
N = max(1, comm.size - 1)
lo, hi = mpp_compute_extent(N, comm.rank, comm.size)
d = Domain(
    dims=("x",), global_sizes={"x": N}, starts={"x": lo}, stops={"x": hi}, comm=comm
)
a = np.arange(N, dtype=np.float64)[lo:hi]
out.append(
    ("empty rank: global_max", float(mpp_global_max(a, d, ("x",))) == float(N - 1))
)
out.append(("empty rank: global_min", float(mpp_global_min(a, d, ("x",))) == 0.0))
out.append(
    (
        "empty rank: global_sum",
        float(mpp_global_sum(a, d, ("x",))) == float(np.arange(N).sum()),
    )
)

# Bug 2: an oversized halo must raise, not return uninitialised memory.
M = 8
l2, h2 = mpp_compute_extent(M, comm.rank, comm.size)
d2 = Domain(
    dims=("x",), global_sizes={"x": M}, starts={"x": l2}, stops={"x": h2}, comm=comm
)
b = np.arange(M, dtype=np.float64)[l2:h2]
if comm.size > 1:
    try:
        mpp_update_domains(b, d2, "x", 0, before=M, after=M)
        out.append(("oversized halo raises", False))
    except HaloWidthError:
        out.append(("oversized halo raises", True))
# A halo that fits must still be exact.
padded, low, high = mpp_update_domains(b, d2, "x", 0, before=1, after=1)
out.append(
    (
        "valid halo exact",
        np.array_equal(padded, np.arange(M, dtype=np.float64)[l2 - low : h2 + high]),
    )
)

res = comm.gather(out, root=0)
if comm.rank == 0:
    for i, (name, _) in enumerate(out):
        print(f"{'OK  ' if all(r[i][1] for r in res) else 'FAIL'} {name}")
