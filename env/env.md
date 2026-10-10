# xrmpi parallel I/O environment

Configure MPI-enabled Python bindings for parallel HDF5 and NetCDF I/O. The automated workflow uses compatible HPC modules when available and builds the native libraries from source otherwise. Manual virtualenv recipes cover Ubuntu packages and source-built NetCDF-C.

## Quick start

Create or update the Conda environment before replacing its Python I/O packages:

```bash
conda env create -f env/environment.yml
conda activate xmpi
bash env/build_libs.sh
conda deactivate
conda activate xmpi
python env/config_stack.py --verify
```

For an existing environment, complete any Conda update first. The build accepts an explicit environment name or prefix:

```bash
bash env/build_libs.sh xmpi
# Alternatively:
bash env/build_libs.sh "$CONDA_PREFIX"
```

The target must be a non-base Conda environment or an existing virtualenv. Helper files are resolved relative to `build_libs.sh`; no checkout location is hard-coded. Source builds require internet access, Git, a C compiler, CMake, Make, system zlib headers, and the Python build dependencies in `environment.yml`. HDF5 2.x requires CMake 3.26 or newer.

## Automated build

### Library selection and installation

The build probes HPC MPI/NetCDF module pairs, newest first, and selects the first pair whose NetCDF-C and linked HDF5 pass parallel header and compile checks. If no compatible pair exists, it builds parallel shared HDF5 and NetCDF-C libraries in `~/.local/hpc_parallel_io`. Open MPI installs there only when neither an MPI module nor `PATH` provides a usable `mpicc`.

`XRMPI_STACK_PREFIX` overrides the native installation directory, which must be separate from the target environment's `lib/` and `lib64/` directories. The default prefix is shared across environments belonging to the same user; rebuilding it replaces their native stack. Use separate prefixes for environments that require different native versions or MPI implementations.

Python packages install through the selected environment's `bin/python`. Activation and deactivation hooks install into:

```text
etc/conda/activate.d/mpi-netcdf.sh
etc/conda/deactivate.d/mpi-netcdf.sh
```

### Replacement sequence

1. `config_stack.py --clean` uninstalls `mpi4py`, `h5py`, `netCDF4`, and `xarray` with pip and removes only their Conda records using `conda remove --force-remove`. It refuses the Conda base environment.
2. `config_stack.py --find` selects compatible HPC modules. If none qualify, or when `config_stack.py --build` is requested, the source build replaces existing HDF5/NetCDF libraries in the native prefix. It configures HDF5 with MPI and zlib support, builds NetCDF-C against that HDF5, and excludes Conda prefixes from CMake searches so native libraries link to system dependencies. Parallel headers and a C probe referencing both parallel APIs are checked.
3. The build writes environment hooks and sets matching compilers, native roots, and runtime library paths. It builds `mpi4py`, MPI-enabled `h5py`, and `netCDF4` from source with pip, without dependency resolution, build isolation, cached wheels, or binary distributions. `LDSHARED` places the parallel library directories first in the link and rpath flags, ahead of Conda Python's default `-L$CONDA_PREFIX/lib`.
4. It reinstalls unpinned `xarray` with `--no-deps --force-reinstall`.
5. Verification imports the Python stack, loads `mpi4py.MPI`, checks MPI-enabled h5py and NetCDF-4 parallel support, and requires every loaded `libhdf5`/`libnetcdf` to come from outside the Conda environment.

Conda's serial `hdf5` and `libnetcdf` remain installed for GDAL, CDO, NCO, wgrib2, and ESMF. The parallel bindings use the selected external stack.

Rerunning the script repeats the replacement. Later Conda changes can overwrite the parallel Python bindings with serial packages; rerun the build after such changes.

### Source versions and build options

By default, source builds use the latest stable HDF5 and NetCDF-C release tags, discovered with `git ls-remote` and shallow-cloned with `git clone --depth 1`. Pre-release tags are ignored; no GitHub API or tarball is used for these libraries. Open MPI uses its latest tagged release tarball because its Git tree requires autotools and submodules.

| Variable | Purpose |
| --- | --- |
| `XRMPI_STACK_PREFIX` | Native installation prefix; default: `~/.local/hpc_parallel_io`. |
| `HPC_STACK_BUILD_N_JOBS` | Native build concurrency; default: `4`. |
| `XRMPI_HDF5_VERSION` | Pin an HDF5 release. |
| `XRMPI_NETCDF_VERSION` | Pin a NetCDF-C release. |
| `XRMPI_HDF5_TAG` | Select an HDF5 tag or branch, such as `develop`. |
| `XRMPI_NETCDF_TAG` | Select a NetCDF-C tag or branch, such as `main`. |

## Activation and verification

The hooks load the selected MPI and NetCDF modules when required, prepend compiler and native library directories, and export `HDF5_DIR`, `NETCDF4_DIR`, and `MPI4PY_BUILD_MPICC`. HDF5 discovery uses `HDF5_DIR` alone; conflicting include, library, and pkg-config overrides are unset. HDF5 file locking is disabled. Deactivation restores saved variables and unloads only modules loaded by the hook.

Running the build in a child shell does not update the caller's runtime paths. Reactivate Conda after the build, as shown in the quick start. For a virtualenv, source the stack hook after normal activation and its deactivation hook before `deactivate`:

```bash
source /path/to/venv/bin/activate
source "$VIRTUAL_ENV/etc/conda/activate.d/mpi-netcdf.sh"
python env/config_stack.py --verify
source "$VIRTUAL_ENV/etc/conda/deactivate.d/mpi-netcdf.sh"
deactivate
```

Module-backed hooks require an initialized `module` command; source-built hooks do not.

The serial and parallel NetCDF libraries share the soname `libnetcdf.so.22`, and a process loads only one. In parallel scripts, import `netCDF4` and `h5py` before packages that load GDAL or ESMF, such as `rioxarray`, `rasterio`, or `xesmf`. Xarray imports its backends lazily, so importing xarray first does not ensure the parallel libraries are loaded.

Import and compile checks establish build capabilities. Test multi-rank reads and writes on the target cluster and filesystem before production use.

## Manual virtualenv setup

These recipes provide alternatives to the automated build. They use Debian-style Open MPI HDF5 paths; the paths shown assume x86-64 Linux.

### Ubuntu 24.04 packaged parallel NetCDF-C

Use the packaged parallel stack to avoid building NetCDF-C from source:

```bash
apt-get install -y --no-install-recommends \
    openmpi-bin libopenmpi-dev libhdf5-openmpi-dev libnetcdf-mpi-dev

# Confirm that NC-4 Parallel Support is yes.
grep "NC-4 Parallel Support" \
    /usr/lib/x86_64-linux-gnu/netcdf/mpi/libnetcdf.settings

python3 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel "cython>=3,<4" numpy

CC=mpicc HDF5_MPI=ON \
NETCDF4_DIR=/usr/lib/x86_64-linux-gnu/netcdf/mpi \
HDF5_INCDIR=/usr/include/hdf5/openmpi \
HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
    venv/bin/pip install --no-binary netCDF4 --no-build-isolation "netCDF4==1.6.5"
```

For this split HDF5 layout, netCDF4's build script requires `HDF5_INCDIR` and `HDF5_LIBDIR`; `HDF5_DIR` and `CPPFLAGS` alone are insufficient. The recorded setup pins netCDF4 to 1.6.5 because builds from 1.7 encountered static declaration conflicts in `netcdf-compat.h` against packaged NetCDF-C 4.9.0, involving the bzip2 and Blosc APIs.

### NetCDF-C from source

Use this recipe when a packaged parallel NetCDF-C build is unavailable. HDF5 and Open MPI come from system packages; NetCDF-C is built against the Open MPI HDF5 installation.

```bash
apt-get install -y libhdf5-openmpi-dev openmpi-bin libopenmpi-dev \
    m4 zlib1g-dev libcurl4-openssl-dev libjpeg-dev automake libtool \
    bison flex cmake python3-venv python3-dev

git clone --branch v4.9.3 https://github.com/Unidata/netcdf-c.git
cd netcdf-c
mkdir build-autotools
cd build-autotools

CC=mpicc \
CPPFLAGS="-I/usr/include/hdf5/openmpi" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi" \
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/openmpi:$LD_LIBRARY_PATH" \
    ../configure --prefix=/usr/local --enable-parallel-tests --disable-dap --disable-byterange

make -j"$(nproc)"
make install
ldconfig

# Expected: yes, and libhdf5_openmpi.so rather than libhdf5_serial.so.
nc-config --has-parallel
ldd /usr/local/lib/libnetcdf.so | grep hdf5

python3 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel cython numpy

CC=mpicc HDF5_MPI=ON \
CPPFLAGS="-I/usr/include/hdf5/openmpi -I/usr/local/include" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/usr/local/lib" \
    venv/bin/pip install mpi4py

CC=mpicc HDF5_MPI=ON \
CPPFLAGS="-I/usr/include/hdf5/openmpi -I/usr/local/include" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/usr/local/lib" \
    venv/bin/pip install --no-binary netCDF4 --no-build-isolation netCDF4

venv/bin/pip install pandas scipy xarray dask cf_xarray rich bottleneck

# climtools imports the plotting stack eagerly, including for MPI-only use.
venv/bin/pip install matplotlib cartopy seaborn cmocean ipython scikit-image
```

In the recorded setup, CMake's `find_package(HDF5)` selected serial HDF5; the autotools configuration above selected the parallel installation through explicit paths.

### Manual runtime paths and checks

Set runtime paths before verification and execution to prevent serial libraries from being loaded. For the packaged stack:

```bash
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu/netcdf/mpi/lib:$LD_LIBRARY_PATH"
```

For the source-built NetCDF-C stack:

```bash
export PATH="/usr/local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/local/lib:$LD_LIBRARY_PATH"
```

Check Python NetCDF-4 parallel support; the result must be `1`:

```bash
venv/bin/python -c "import netCDF4; print(netCDF4.__has_parallel4_support__)"
```

If `apt-get update` fails with a NodeSource `403` on its `InRelease` file, disable or remove that repository before retrying. This issue occurred in the original container setup.

## Build references

- [mpi4py source installation](https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources)
- [h5py installation](https://docs.h5py.org/en/stable/build.html)
- [NetCDF-C CMake instructions](https://docs.unidata.ucar.edu/netcdf-c/current/netCDF-CMake.html)
