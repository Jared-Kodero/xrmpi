"""Interpolation must not make every rank hold the global field.\n\nRun standalone: ``mpirun -n 8 python test/mpi_test_interp_memory.py``\n"""

import tracemalloc

import numpy as np

import xgeo as xg
import xgeo.xarray.elementwise as ew
from xgeo import MPIContext

mpi = MPIContext()
comm = mpi.comm

NT, NX = 8000, 400


def fill(a, b):
    return np.random.default_rng(0).random((b - a, NX))


d = xg.create_distributed_dataarray(
    mpi,
    fill,
    dims=("t", "x"),
    shape={"t": NT, "x": NX},
    dim="t",
    name="v",
    coords={"t": np.arange(NT, dtype=float)},
).load()
meta = d.meta
start, stop = meta["starts"]["t"], meta["stops"]["t"]
targets = np.linspace(start, stop - 1, stop - start)


def peak_mb():
    tracemalloc.start()
    b = tracemalloc.get_traced_memory()[0]
    d.interp("t", targets)
    p = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return (p - b) / 1e6


halo = comm.allgather(peak_mb())
# Force the old behaviour by disabling the width estimate.
saved = ew._interp_halo_width
ew._interp_halo_width = lambda *a, **k: None
gathered = comm.allgather(peak_mb())
ew._interp_halo_width = saved

if comm.rank == 0:
    g = NT * NX * 8 / 1e6
    print(
        f"ranks={comm.size}  global field={g:.0f} MB  local slice={g / comm.size:.0f} MB"
    )
    print(f"  halo path   : max per-rank peak {max(halo):7.1f} MB")
    print(f"  gather path : max per-rank peak {max(gathered):7.1f} MB")
    print(f"  reduction   : {max(gathered) / max(halo):.1f}x")
