"""Isolate the effect of packed-buffer halo exchange: message count and
wall time for mpp_halo_exchange() as a function of haloed variable count, at
a fixed small per-variable size so the result reflects per-call MPI
posting overhead (what buffer packing targets) rather than payload
bandwidth.

Run with: mpirun --oversubscribe -n <N> python bench_halo_packing.py
"""

from __future__ import annotations

import time

import numpy as np
import xarray as xr
from mpi4py import MPI

import xgeo as xg
from xgeo import MPIContext
from xgeo.xarray.halo import mpp_halo_exchange

mpi = MPIContext()


N_VARS = 12
GLOBAL_LEN = 40 * mpi.comm.size
BEFORE = AFTER = 2
REPS = 100

rank = mpi.comm.rank

if rank == 0:
    data_vars = {}
    for i in range(N_VARS):
        dtype = [np.float32, np.float64, np.int32][i % 3]
        arr = (np.arange(GLOBAL_LEN, dtype=np.float64) + i).astype(dtype)
        data_vars[f"v{i}"] = xr.DataArray(arr, dims=("x",), name=f"v{i}")
    full = xr.Dataset(data_vars)
else:
    full = None

ds = xg.distribute_data(full, mpi, dim="x", log_partitions=False)

mpi.comm.barrier()
mpp_halo_exchange(mpi, ds._prepare(), "x", before=BEFORE, after=AFTER)  # warm up

mpi.comm.barrier()
t0 = time.perf_counter()
for _ in range(REPS):
    mpp_halo_exchange(mpi, ds._prepare(), "x", before=BEFORE, after=AFTER)
mpi.comm.barrier()
t1 = time.perf_counter()

local_time = (t1 - t0) / REPS
slowest = mpi.comm.reduce(local_time, op=MPI.MAX, root=0)

if rank == 0:
    print(
        f"ranks={mpi.comm.size}  n_vars={N_VARS} (3 dtypes: float32/float64/int32)  reps={REPS}"
    )
    print(
        f"wall time per mpp_halo_exchange() call (slowest rank): {slowest * 1000:.4f} ms"
    )
