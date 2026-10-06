"""MPI-vs-native correctness suite: single entry point.

The one script test.sh invokes via srun; run locally with:
    mpirun --oversubscribe -n <N> python mpi_test.py

Builds the shared fixtures once (mpi_test_common.build_fixtures), then
imports and runs each mpi_test_*.py module in turn -- each owns one
coherent area rather than everything living in a single file:

  mpi_test_construction.py  construction: open_distributed_dataset,
                             create_distributed_dataarray, create_distributed_dataset;
                             even/uneven/multi-dim partitioning
  mpi_test_reductions.py    mean/sum/min/max/var/std/median; rank-local,
                             reconstruction, and multi-dim dedup+coverage
  mpi_test_halo_ops.py      rolling_reduce, coarsen_reduce, diff, shift,
                             differentiate, ffill, bfill
  mpi_test_scans.py         NumPy dispatch, isel, cumsum, sortby,
                             reindex, interp, matmul, where
  mpi_test_groupby.py       groupby, resample
  mpi_test_misc.py          prod, any, all, first, last, align,
                             evaluate, roll, repartition, apply
  mpi_test_mpp.py           the FMS-adapted primitives in mpp/
                             directly: domain bookkeeping, reproducing
                             sum/product, checksums, halo start/complete

Standalone checks, run separately because they need their own rank counts
or no MPI at all:

  mpi_test_mpp_edges.py      degenerate decompositions: ranks owning no
                             points, halos wider than a neighbour
  mpi_test_interp_memory.py  interpolation must not hold the global field
  test_xnpy_store.py         XNpyStore round trips and refusals (serial)

Every numeric check compares this rank's local slice against the
matching slice of a plain, non-distributed xarray/numpy computation --
never against a "rank 0 holds everything" assumption. Multi-dimensional
checks additionally verify reconstruction correctness (exact,
non-overlapping global coverage) and rank consistency (no two ranks
claim the same range).
"""

from __future__ import annotations

from xrmpi.test import (
    mpi_test_construction,
    mpi_test_groupby,
    mpi_test_halo_ops,
    mpi_test_misc,
    mpi_test_mpp,
    mpi_test_reductions,
    mpi_test_scans,
)
from xrmpi.test.mpi_test_common import build_fixtures, phase, report

_MODULES = (
    ("construction", mpi_test_construction),
    ("reductions", mpi_test_reductions),
    ("halo_ops", mpi_test_halo_ops),
    ("scans", mpi_test_scans),
    ("groupby", mpi_test_groupby),
    ("misc", mpi_test_misc),
    ("mpp", mpi_test_mpp),
)

with phase("build_fixtures"):
    fixtures = build_fixtures()

for _label, _module in _MODULES:
    with phase(_label):
        _module.run(fixtures)

report()
