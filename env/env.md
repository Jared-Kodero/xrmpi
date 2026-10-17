# xrmpi parallel I/O environment

`pip install xrmpi` installs the Python dependencies, but collective parallel I/O requires `mpi4py`, `h5py`, and `netCDF4` to be compiled against an MPI-enabled HDF5 and NetCDF-C stack. Prebuilt PyPI wheels bundle serial libraries and do not provide this. The standard build below uses distribution packages; HPC systems modify it to build against the cluster's modules.

## Standard build (Ubuntu 24.04)

The packaged parallel stack avoids building NetCDF-C:

```bash
apt-get install -y --no-install-recommends \
    openmpi-bin libopenmpi-dev libhdf5-openmpi-dev libnetcdf-mpi-dev

python3 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel "cython>=3,<4" numpy

CC=mpicc HDF5_MPI=ON \
NETCDF4_DIR=/usr/lib/x86_64-linux-gnu/netcdf/mpi \
HDF5_INCDIR=/usr/include/hdf5/openmpi \
HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
    venv/bin/pip install --no-binary mpi4py,h5py,netCDF4 --no-build-isolation \
    mpi4py h5py "netCDF4==1.6.5"

venv/bin/pip install xrmpi

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu/netcdf/mpi/lib:$LD_LIBRARY_PATH"
```

For this split HDF5 layout, the netCDF4 build requires `HDF5_INCDIR` and `HDF5_LIBDIR`; `HDF5_DIR` alone is insufficient. netCDF4 is pinned to 1.6.5 because 1.7 builds conflict with the bzip2 and Blosc declarations in `netcdf-compat.h` against the packaged NetCDF-C 4.9.0.

## HPC systems

On HPC systems, apply the following changes to the standard build so that the bindings are compiled against the cluster's MPI, parallel HDF5, and parallel NetCDF-C modules, which are tuned for the site interconnect and filesystem:

1. Skip the `apt-get` step and load the site modules instead (names are site specific):

   ```bash
   module load <mpi> <hdf5-parallel> <netcdf-c-parallel>
   nc-config --has-parallel    # must print yes
   ```

2. Replace the distribution paths with the module prefixes:

   ```bash
   export CC=mpicc HDF5_MPI=ON HDF5_DIR=<hdf5-prefix> NETCDF4_DIR="$(nc-config --prefix)"
   ```

   Do not set `HDF5_INCDIR` or `HDF5_LIBDIR` unless the module uses a split layout. Drop the `netCDF4==1.6.5` pin unless the build fails with the declaration conflict described above.

3. Run the same `pip install --no-binary mpi4py,h5py,netCDF4 --no-build-isolation mpi4py h5py netCDF4` and `pip install xrmpi` commands in the virtualenv.

4. Load the same modules at run time, including in batch scripts, in place of the `LD_LIBRARY_PATH` export.

Do not install `hdf5`, `libnetcdf`, `h5py`, or `netCDF4` from Conda into the same environment. The serial libraries share the soname of the parallel ones (`libnetcdf.so.22`), and a process loads only one.

## Verification

```bash
python -c "import mpi4py.MPI, h5py, netCDF4; print(h5py.get_config().mpi, netCDF4.__has_parallel4_support__)"
```

The output must be `True 1`. Confirm that the loaded libraries come from the intended stack:

```bash
ldd "$(python -c 'import netCDF4._netCDF4 as m; print(m.__file__)')" | grep -E "hdf5|netcdf"
```

In parallel scripts, import `netCDF4` and `h5py` before packages that load GDAL or ESMF, such as `rioxarray`, `rasterio`, or `xesmf`. Xarray imports its backends lazily, so importing xarray first does not ensure that the parallel libraries are loaded.

Import checks establish build capabilities only. Test multi-rank reads and writes on the target cluster and filesystem before production use.

## References

- [mpi4py source installation](https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources)
- [h5py installation](https://docs.h5py.org/en/stable/build.html)
- [netCDF4-python installation](https://unidata.github.io/netcdf4-python/#installation)
