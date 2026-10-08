# xrmpi parallel I/O environment

Solve the ordinary Conda dependencies before replacing the native I/O stack:

```bash
conda env create -f env/environment.yml
conda activate xmpi
bash env/build_libs.sh
conda deactivate
conda activate xmpi
```

For an existing environment, complete any Conda update first. The build also accepts an explicit environment name or prefix:

```bash
bash env/build_libs.sh xmpi
bash env/build_libs.sh "$CONDA_PREFIX"
```

The scripts resolve their helper files beside `build_libs.sh`; no checkout location is hard-coded. The target must be a non-base Conda environment or an existing virtualenv. Source builds require internet access, `git`, a C compiler, CMake, Make, and the Python build dependencies in `environment.yml`.

## Installation locations

The build uses native HPC module libraries when a compatible pair exists. Only a source build installs HDF5 and NetCDF-C, under `~/.local/hpc_parallel_io`, matching the old source-build location. Open MPI also installs there when no usable MPI compiler is found. `XRMPI_STACK_PREFIX` overrides this location. The native prefix must be separate from the target environment's `lib/` and `lib64/` directories.

Python packages install into the selected environment through its `bin/python`. Activation and deactivation hooks install into that environment's `etc/conda/activate.d/mpi-netcdf.sh` and `etc/conda/deactivate.d/mpi-netcdf.sh`.

The default native prefix is shared by environments belonging to the same user. Rebuilding it replaces the stack used by those environments. Use a separate `XRMPI_STACK_PREFIX` when environments need different native versions or MPI implementations.

## Replacement workflow

1. `config_stack.py --clean` uninstalls `mpi4py`, `h5py`, `netCDF4`, and `xarray` with pip and removes only their Conda records with `conda remove --force-remove`. The Conda base environment is refused.
2. Conda's serial `hdf5` and `libnetcdf` stay installed, as in the old workflow, so GDAL, CDO, NCO, wgrib2, and ESMF keep their own libraries. The parallel bindings never use them: see steps 5 and 7.
3. `config_stack.py --find` probes HPC MPI/NetCDF module pairs, newest first, and uses the first pair whose NetCDF-C and linked HDF5 pass the parallel header and compile checks. The selected MPI and NetCDF modules are loaded for the Python builds and by the activation hook.
4. Only when no compatible pair exists, or with `config_stack.py --build`, download and build the latest stable HDF5 and NetCDF-C releases together as parallel shared libraries in the native prefix. The newest stable release tag is read with `git ls-remote` (pre-release tags are ignored) and shallow-cloned with `git clone --depth 1`; no GitHub API or tarball is used. Open MPI, when needed, is the latest tagged release downloaded as its release tarball, because its git tree requires autotools and submodules. HDF5 is configured with MPI and zlib support, and NetCDF-C against that HDF5. The MPI compiler is the newest MPI module providing `mpicc`, then `mpicc` from `PATH`; Open MPI is built only when neither exists. Existing HDF5/NetCDF libraries in that prefix are removed before replacement. Conda prefixes are excluded from the CMake searches, so the shared native libraries link only system dependencies; HDF5 requires system zlib headers. Check parallel headers and compile a C probe referencing both parallel APIs.
5. Write environment hooks and set matching compiler, native roots, and runtime library paths. Build `mpi4py`, MPI-enabled `h5py`, and `netCDF4` from source with pip, without dependency resolution, build isolation, cached wheels, or binary distributions. Conda Python's link flags place `-L$CONDA_PREFIX/lib` first, which would link Conda's serial libraries, so `LDSHARED` is overridden to link and rpath the parallel library directories first.
6. Install unpinned `xarray` with pip `--no-deps --force-reinstall`.
7. Import the Python stack, load `mpi4py.MPI`, require MPI-enabled h5py and NetCDF-4 parallel support, and require that every loaded `libhdf5`/`libnetcdf` comes from outside the Conda environment.

Rerunning the script repeats the replacement. Later Conda changes can reinstall the serial `h5py` or `netcdf4` packages over the parallel builds, so repeat the build after such changes.

The serial and parallel `libnetcdf` share the soname `libnetcdf.so.22`, and a process loads only one. In parallel scripts, import `netCDF4` and `h5py` (xarray imports them lazily) before packages that load GDAL or ESMF (for example `rioxarray`, `rasterio`, or `xesmf`); otherwise netCDF4 uses the serial copy that those packages loaded first.

`HPC_STACK_BUILD_N_JOBS` controls native build concurrency (default: 4). By default the latest stable releases are built. `XRMPI_HDF5_VERSION` and `XRMPI_NETCDF_VERSION` pin a release; `XRMPI_HDF5_TAG` and `XRMPI_NETCDF_TAG` select any tag or branch, for example `develop` (HDF5) or `main` (NetCDF-C) for unreleased code. Source builds require `git`; HDF5 2.x requires CMake 3.26 or newer.

## Activation and verification

The hooks load the selected MPI module when required, prepend its compiler directory and native library directories, and export `HDF5_DIR`, `NETCDF4_DIR`, and `MPI4PY_BUILD_MPICC`. HDF5 file locking is disabled as in the old hook. HDF5 discovery uses `HDF5_DIR` alone; conflicting include, library, and pkg-config overrides are unset. Deactivation restores the saved variables and unloads only modules loaded by the hook.

Running the build in a child shell does not update the caller's runtime paths. Reactivate the environment after the build, then verify:

```bash
python env/config_stack.py --verify
```

For a virtualenv, source the stack activation hook after its normal activation script and source the stack deactivation hook before `deactivate`:

```bash
source /path/to/venv/bin/activate
source "$VIRTUAL_ENV/etc/conda/activate.d/mpi-netcdf.sh"
python env/config_stack.py --verify
source "$VIRTUAL_ENV/etc/conda/deactivate.d/mpi-netcdf.sh"
deactivate
```

Module-backed hooks require an initialized `module` command; source-built hooks do not. Import and compile checks establish build capabilities, not collective I/O behavior. Test multi-rank reads and writes on the target cluster and filesystem before production use.

Build references: [mpi4py source installation](https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources), [h5py installation](https://docs.h5py.org/en/stable/build.html), and [NetCDF-C CMake instructions](https://docs.unidata.ucar.edu/netcdf-c/current/netCDF-CMake.html).
