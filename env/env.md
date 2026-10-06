# xrmpi environment

This directory manages the environment for users who clone `xrmpi`. Run the commands below inside that clone, beside its `pyproject.toml`. `xgeo/env/` and the author's `.workspace/setup/` are separate setups with their own specifications and helpers.

## Run from the xrmpi clone root

```bash
python3 env/setup_env.py xmpi
conda activate xmpi
```

Passing a name creates or updates that Conda environment. Without a name, the installer uses the active non-base Conda environment or active virtualenv; otherwise it creates `xmpi`:

```bash
conda activate xmpi
python3 env/setup_env.py
```

The automated installer targets Linux and requires internet access for package/source downloads. Native source builds require a C compiler, CMake, and Make; HPC module builds require an initialized module system.

The Python used to run this installer must be 3.10 or newer; the target environment Python must be 3.14 or newer. A virtualenv must already provide that version; the Conda specification selects it automatically. Conda is installed under `~/miniconda3` only if needed and not already available.

## Installation order

1. Solve `environment.yml` for a Conda target, then install ordinary Python dependencies and build tools. For a virtualenv, install those dependencies with its Python.
2. Locate a compatible MPI/parallel-NetCDF module pair. If unavailable, build the native stack under the target environment's `opt/parallel-io/`; an MPI implementation is also built when `mpicc` is absent.
3. Remove existing `xarray`, `mpi4py`, `h5py`, and `netCDF4`. Conda-owned copies are removed with Conda without removing their dependents; pip removes any remaining distributions.
4. Build `mpi4py`, MPI-enabled `h5py`, and `netCDF4` from source using the selected MPI compiler and parallel HDF5/NetCDF-C libraries. Disable cached wheels and dependency resolution for these builds.
5. Reinstall Xarray and verify MPI-enabled HDF5 and parallel NetCDF-4 support.
6. Install local `xgeo` and `xrmpi` editable with `--no-deps`, then check Python dependency metadata. When no sibling `xgeo` source tree is present, its declared dependency is installed during the ordinary dependency phase.

There is no Conda solve after the parallel build. If later dependency changes replace the MPI extensions, rerun the installer.

`HPC_STACK_BUILD_N_JOBS` controls native build concurrency (default: 4). `XRMPI_STACK_PREFIX` overrides the native build location.

## Activation

The build writes `etc/conda/activate.d/mpi-netcdf.sh` and the corresponding deactivation hook. They restore the selected modules, compiler path, HDF5/NetCDF roots, and shared-library paths when activating Conda.

For a virtualenv, source its normal activation script and the stack hook. Before deactivating it, source the stack deactivation hook:

```bash
source /path/to/venv/bin/activate
source "$VIRTUAL_ENV/etc/conda/activate.d/mpi-netcdf.sh"
# Run xrmpi work here.
source "$VIRTUAL_ENV/etc/conda/deactivate.d/mpi-netcdf.sh"
deactivate
```

Module-backed hooks require an initialized `module` command. Source-built hooks also work without a module system.

## Verification

After activating the target environment and stack hooks:

```bash
python - <<'PY'
import mpi4py
mpi4py.rc.initialize = False
import h5py
import netCDF4
import xarray

assert h5py.get_config().mpi
assert netCDF4.__has_parallel4_support__
print("xarray:", xarray.__version__)
print("h5py MPI:", h5py.get_config().mpi)
print("netCDF4 parallel4:", netCDF4.__has_parallel4_support__)
PY
```

These checks confirm the extensions' build capabilities. Verify collective reads and writes with the target MPI launcher and filesystem before running production workloads.

Build references: [mpi4py source installation](https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources), [h5py parallel HDF5](https://docs.h5py.org/en/stable/mpi.html), and [netCDF4 parallel I/O](https://unidata.github.io/netcdf4-python/#parallel-io).
