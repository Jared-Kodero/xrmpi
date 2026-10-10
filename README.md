# xrmpi

MPI-distributed Xarray operations, domain decomposition, halo exchange, collective reductions, and parallel NetCDF I/O.

`xrmpi` contains the distributed-array layer formerly included in `xgeo`. Geospatial analysis, plotting, colormaps, and storage remain in [`xgeo`](../xgeo/README.md). The packages are independent: xrmpi maintains its own locking, progress, and temporary-file utilities.

## Installation

Python **3.14 or newer** is required. From the `xrmpi` clone root, create the environment, build the parallel I/O stack, and install the package:

```bash
conda env create -f env/environment.yml
conda activate xmpi
bash env/build_libs.sh
conda deactivate && conda activate xmpi      # load the hooks written by build_libs.sh
python -m pip install --no-deps -e .
python env/config_stack.py --verify
```

`build_libs.sh` removes only the Python bindings `mpi4py`, `h5py`, `netCDF4`, and `xarray`; Conda's serial HDF5 and NetCDF-C stay installed for GDAL, CDO, NCO, and ESMF. It then selects one parallel MPI/HDF5/NetCDF-C stack:

1. a compatible HPC module pair (`hpcx-mpi` + `netcdf-mpi`), checked by a parallel compile probe;
2. otherwise, the latest stable HDF5 and NetCDF-C releases, cloned with `git` and built with one MPI compiler (an MPI module, `mpicc` on `PATH`, or a source-built Open MPI).

`mpi4py`, `h5py`, and `netCDF4` are compiled against that stack, and activation hooks restore its runtime paths. See [`env/env.md`](env/env.md) for options, virtualenv use, and the full workflow. To check the whole environment, including the Conda packages that share HDF5 and NetCDF:

```bash
python env/test_stack.py                                   # mpiexec -n 2
python env/test_stack.py --launcher "srun --mpi=pmix"      # inside a Slurm allocation
```

An ordinary `python -m pip install .` also works, but it does not guarantee HDF5/NetCDF libraries with collective parallel I/O. Use `build_libs.sh` when parallel output is required.

## Quick start

Every rank runs the same script. Launch with the MPI used to build the environment, for example `mpirun -n 4 python quickstart.py`, or `srun --mpi=pmix -n 4 python quickstart.py` inside a Slurm allocation:

```python
import numpy as np

import xrmpi as xm

mpi = xm.MPIContext()
dist = xm.open_dataset("data.nc", mpi, partition_dim="time")
logged = np.log(dist["pr"])
rolled = dist.rolling_reduce("time", window=5, reduce="mean")
time_mean = dist.mean(dim="time")
time_mean.to_netcdf("time_mean.nc", parallel=True)
```

All participating ranks must execute collective operations and collective I/O calls.

| Function | Purpose |
| --- | --- |
| `xrmpi.open_dataset` | Open a NetCDF file lazily; each rank keeps its slice of the partition dimension(s) |
| `xrmpi.new_dataarray` | Build a distributed DataArray from a fill function and global sizes |
| `xrmpi.new_dataset` | Build a distributed Dataset from `(dims, fill)` variable specifications |
| `xrmpi.partition` | Split an object held on one rank (`root`) across all ranks |
| `xrmpi.to_netcdf` | Write serially or with collective parallel NetCDF-4 I/O |
| `xrmpi.append_to_netcdf` | Append to an existing NetCDF file |
| `xrmpi.MPIContext` | Communicator, rank information, and MPI diagnostics |
| `xrmpi.MPIXarray` | Xarray-like wrapper around a rank-local distributed object |

Supported operations include NumPy ufuncs, halo-aware rolling and differences, collective reductions, grouped/resampled reductions, cumulative scans, interpolation, reindexing, sorting, redistribution, and NetCDF output. See [`core/core.py`](core/core.py) and [`core/io.py`](core/io.py) for signatures and options.

## Detailed example: ERA5 dewpoint to vapor pressure

This example processes an hourly ERA5 pressure-level file, `dewpoint.nc`, holding `dpt(time, level, lat, lon)` in °C (122 976 hours × 19 levels × 29 × 33 points, about 9 GB as float32). Each rank reads and processes only its block of hours.

Actual vapor pressure follows from dewpoint with the Bolton (1980) form of the Magnus equation:

$$
e = 6.112 \exp\left(\frac{17.67\, T_d}{T_d + 243.5}\right)
$$

where $e$ is the vapor pressure (hPa) and $T_d$ the dewpoint temperature (°C). The fit is accurate to about 0.1 % for $-30\,^\circ\mathrm{C} \le T_d \le 35\,^\circ\mathrm{C}$ over liquid water; colder upper-tropospheric dewpoints carry larger relative errors.

Save as `vapor_pressure.py`:

```python
"""Hourly vapor pressure, 24-h running mean, and climatology from ERA5 dewpoint."""

import numpy as np

import xrmpi as xm

PATH = "dewpoint.nc"


def vapor_pressure(dewpoint):
    """Return vapor pressure in hPa from dewpoint in degC (Bolton, 1980)."""
    return 6.112 * np.exp(17.67 * dewpoint / (dewpoint + 243.5))


mpi = xm.MPIContext()

# 1. Open lazily; rank 0 reads the header and every rank keeps a block of hours.
data = xm.open_dataset(PATH, mpi, partition_dim="time", chunks={"time": 24})
dpt = data["dpt"]

# 2. Rank-local arithmetic and ufuncs: no communication is needed.
e = vapor_pressure(dpt)

# 3. Rolling windows cross rank boundaries; halo exchange supplies the
#    23 neighbouring hours each rank needs at the edges of its block.
e_daily = e.rolling_reduce("time", window=24, reduce="mean")

# 4. A reduction over the partition dimension is collective: every rank
#    contributes its partial result to the global climatology.
e_clim = e.mean(dim="time")

# 5. Collective NetCDF-4 output: each rank writes its own block of hours.
e.to_netcdf("vapor_pressure.nc", parallel=True)
e_daily.to_netcdf("vapor_pressure_24h.nc", parallel=True)
e_clim.to_netcdf("vapor_pressure_climatology.nc", parallel=True)

if mpi.is_root():
    print(f"{mpi.comm.size} ranks processed {PATH}")
```

Run it on 8 ranks inside an allocation:

```bash
salloc -n 8 -t 01:00:00
conda activate xmpi
srun --mpi=pmix -n 8 python vapor_pressure.py      # or: mpirun -n 8 python vapor_pressure.py
```

What happens at each step:

| Step | Communication | Notes |
| --- | --- | --- |
| `open_dataset` | Rank 0 reads the header and broadcasts the layout | Ranks open the file only after the plan is agreed; data stay lazy until needed |
| `vapor_pressure` | None | Element-wise, so each rank works on its own hours |
| `rolling_reduce` | Neighbour halo exchange | Without halos, windows near each block edge would be truncated |
| `mean(dim="time")` | Collective reduction | Every rank must call it, including ranks with few hours |
| `to_netcdf(parallel=True)` | Collective MPI-IO | Distributed results are written by all ranks; the reduced climatology has no partition dimension, so rank 0 writes it while the others pass a placeholder |

Parallel reads and writes need the NetCDF-4/HDF5 format; check an input with `ncdump -k dewpoint.nc` (expect `netCDF-4`). Classic or 64-bit-offset files open serially but not with `parallel=True`.

## Building data without a file

`new_dataarray` evaluates a fill function only on each rank's slice, so large synthetic fields never exist on a single rank. The function receives the global `(start, stop)` bounds of the rank's slice of each partition dimension, in the order given by `dim`, and returns an array of the local shape. `partition` instead starts from an object held on one rank:

```python
import numpy as np
import xarray as xr

import xrmpi as xm

mpi = xm.MPIContext()


def fill(start, stop):
    """Return hourly air temperature in K for global hours start to stop."""
    t = np.arange(start, stop)[:, None, None]
    y = np.arange(29)[None, :, None]
    x = np.arange(33)[None, None, :]
    lat = 40.5 + 0.25 * y
    diurnal = 5.0 * np.sin(2.0 * np.pi * (t % 24) / 24.0)
    return 300.0 - 0.6 * lat + diurnal + 0.0 * x


tas = xm.new_dataarray(
    mpi,
    fill,
    ("time", "lat", "lon"),
    shape={"time": 8760, "lat": 29, "lon": 33},
    dim="time",
    dtype=np.float32,
    name="tas",
    attrs={"units": "K"},
)

small = xr.open_dataset("station.nc") if mpi.is_root() else None
local = xm.partition(small, mpi, dim="time")
```

## Package overview

| Namespace | Purpose |
| --- | --- |
| `xrmpi` | Distributed-array constructors and public MPI entry points |
| `xrmpi.core` | Xarray operations, planning, partitioning, and I/O |
| `xrmpi.mpi` | MPI context, initialization, and diagnostics |
| `xrmpi.mpp` | FMS-derived communication and domain-decomposition layer |

Importing `xrmpi` does not import mpi4py or initialize MPI; public objects load on first access. The package sets `HDF5_USE_FILE_LOCKING=FALSE` unless it is already set.

The communication layer is adapted from GFDL's [FMS](https://github.com/NOAA-GFDL/FMS). `mpp_*` modules implement the FMS-style operations; `ext_*` modules provide additional collectives and domain helpers.

## Testing

The suite is in [`tests/`](tests/) and runs with `python -m pytest tests` from the clone root. `tests/test_local.py` needs no launcher: it covers domain decomposition, the extended-fixed-point digit kernel (against an executable copy of the original algorithm), the reproducible product, the buffer pool, and the progress bar. `tests/test_mpi.py` launches each script in `tests/mpi_cases/` with `mpiexec` on 1, 2, 3 and 4 ranks and compares reductions, scans, stencils, indexing, grouped and resampled reductions, two- and three-axis process grids, halo exchange, and file I/O with serial xarray references. The scripts exit non-zero if any rank records a mismatch. They skip when no launcher is found, honour `XRMPI_TEST_TIMEOUT` (seconds per script, default 600), and relax the parallel-NetCDF probe when the installed netCDF4 lacks parallel support (collective parallel output itself is not exercised in that case).

## References

* Bolton, D. (1980). The computation of equivalent potential temperature. *Monthly Weather Review*, 108(7), 1046–1053. https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2

## Links

* [Environment setup](env/env.md)
* [Parallel stack build](env/build_libs.sh)
* [Stack test](env/test_stack.py)
* [Distributed-array implementation](core/core.py)
* [I/O implementation](core/io.py)
* [Communication layer](mpp/)
* [License](LICENSE)

Distributed under the [MIT License](LICENSE).