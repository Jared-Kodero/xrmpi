# xrmpi

MPI-distributed Xarray operations, domain decomposition, halo exchange, collective reductions, and parallel NetCDF I/O.

`xrmpi` contains the distributed-array layer formerly included in `xgeo`. Geospatial analysis, plotting, colormaps, and storage remain in [`xgeo`](../xgeo/README.md). The current MPI implementation also uses shared logging, progress, and temporary-file utilities from `xgeo`, so its package metadata declares that dependency.

## Installation

Python **3.14 or newer** is required. For the package's own environment, run from the `xrmpi` clone root:

```bash
python3 env/setup_env.py xmpi
conda activate xmpi
```

The installer uses [`env/environment.yml`](env/environment.yml), completes dependency installation, removes existing `xarray`, `mpi4py`, `h5py`, and `netCDF4`, then rebuilds the MPI extensions against one parallel MPI/HDF5/NetCDF-C stack and reinstalls Xarray. Local `xgeo` and `xrmpi` are installed editable with dependency resolution disabled after the stack build. See [`env/env.md`](env/env.md) for active-environment use and verification.

An ordinary package install is also possible:

```bash
python -m pip install .
```

This does not ensure that the resulting HDF5/NetCDF libraries support collective parallel output. Use the environment installer when parallel I/O is required.

## Distributed arrays

Save this example as `distributed.py` and launch it with the MPI implementation used to build the environment, for example `mpiexec -n 4 python distributed.py`:

```python
import numpy as np

import xrmpi as xm


mpi = xm.MPIContext()
dist = xm.open_distributed_dataset("data.nc", mpi, partition_dim="time")
logged = np.log(dist["pr"])
rolled = dist.rolling_reduce("time", window=5, reduce="mean")
global_mean = dist.mean(dim="time")
dist.to_netcdf("output.nc", parallel=True)
```

All participating ranks must execute collective operations and collective I/O calls.

| Constructor | Purpose |
| --- | --- |
| `xrmpi.open_distributed_dataset` | Open a NetCDF dataset using rank-local partitions |
| `xrmpi.create_distributed_dataarray` | Construct a distributed DataArray |
| `xrmpi.create_distributed_dataset` | Construct a distributed Dataset |
| `xrmpi.distribute_data` | Partition an existing Xarray object |
| `xrmpi.empty_distributed_dataset` | Represent an empty rank-local partition |
| `xrmpi.is_distributed_empty` | Check whether a partition is empty |

Supported operations include NumPy ufuncs, halo-aware rolling and differences, collective reductions, grouped/resampled reductions, cumulative scans, interpolation, reindexing, sorting, redistribution, and NetCDF output. See [`core/core.py`](core/core.py) and [`core/io.py`](core/io.py) for signatures and options.

## Package overview

| Namespace | Purpose |
| --- | --- |
| `xrmpi` | Distributed-array constructors and public MPI entry points |
| `xrmpi.core` | Xarray operations, planning, partitioning, and I/O |
| `xrmpi.mpi` | MPI context, initialization, and diagnostics |
| `xrmpi.mpp` | FMS-derived communication and domain-decomposition layer |

The communication layer is adapted from GFDL's [FMS](https://github.com/NOAA-GFDL/FMS). `mpp_*` modules implement the FMS-style operations; `ext_*` modules provide additional collectives and domain helpers.

## Testing

The MPI scripts are in [`test/`](test/). They require a matching MPI launcher and sufficient memory for their fixture sizes. The current test sources also retain imports from before the package split; migrate those imports before running the suite against the new package names.

## Links

* [Environment setup](env/env.md)
* [Environment installer](env/setup_env.py)
* [Distributed-array implementation](core/core.py)
* [I/O implementation](core/io.py)
* [Communication layer](mpp/)
* [License](LICENSE)

Distributed under the [MIT License](LICENSE).
