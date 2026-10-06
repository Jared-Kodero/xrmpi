#!/bin/bash -l
# Build the target environment's MPI stack using this setup's own helpers.
# Usage: bash build_libs.sh <environment-prefix-or-conda-name>

set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/utils.sh"
target="${1:-${CONDA_PREFIX:-${VIRTUAL_ENV:-}}}"
if [[ -z "$target" || "$target" == "base" ]]; then
    echo "Specify a non-base environment prefix or Conda name." >&2
    exit 1
fi
if [[ -x "$target/bin/python" ]]; then
    env_prefix="$(cd "$target" && pwd -P)"
else
    eval "$(conda shell.bash hook)"
    conda activate "$target"
    env_prefix="$CONDA_PREFIX"
fi
python="$env_prefix/bin/python"
export XRMPI_STACK_PREFIX="${XRMPI_STACK_PREFIX:-$env_prefix/opt/parallel-io}"

stack="$("$python" "$script_dir/find_stack.py")"
eval "$stack"
if [[ -n "$MPI_MODULE" || -n "$NETCDF_MODULE" ]]; then
    module purge
    [[ -z "$MPI_MODULE" ]] || module load "$MPI_MODULE"
    [[ -z "$NETCDF_MODULE" ]] || module load "$NETCDF_MODULE"
fi

export NETCDF4_DIR HDF5_DIR HDF5_LIBDIR
export HDF5_INCDIR="$HDF5_DIR/include"
export PATH="$(dirname "$MPICC"):$env_prefix/bin:$PATH"
stack_library_path="$HDF5_LIBDIR:$NETCDF4_DIR/lib:$NETCDF4_DIR/lib64"
export LD_LIBRARY_PATH="$stack_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

phase "Parallel I/O build roots"
echo "Python:      $python"
echo "mpicc:       $MPICC"
echo "HDF5 root:   $HDF5_DIR"
echo "NetCDF root: $NETCDF4_DIR"

phase "Removing existing xarray/mpi4py/h5py/netCDF4"
"$python" "$script_dir/clean_stack.py"

phase "Installing stack activation hooks"
"$python" "$script_dir/conda_hooks.py" \
    --prefix "$env_prefix" \
    --env-name "$(basename "$env_prefix")" \
    --mpi-module "$MPI_MODULE" \
    --netcdf-module "$NETCDF_MODULE" \
    --hdf5-dir "$HDF5_DIR" \
    --netcdf4-dir "$NETCDF4_DIR" \
    --hdf5-libdir "$HDF5_LIBDIR" \
    --mpicc "$MPICC" \
    --library-path "$stack_library_path"

phase "Building mpi4py"
export MPI4PY_BUILD_MPICC="$MPICC"
export CC="$MPICC"
"$python" -m pip install \
    --no-cache-dir --no-binary=mpi4py --no-build-isolation --no-deps mpi4py

phase "Building h5py with parallel HDF5"
export HDF5_MPI="ON"
"$python" -m pip install \
    --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py

phase "Building netCDF4"
"$python" -m pip install \
    --no-cache-dir --no-binary=netCDF4 --no-build-isolation --no-deps netCDF4
unset CC HDF5_MPI

phase "Reinstalling xarray"
"$python" -m pip install --no-cache-dir --no-deps xarray

phase "Verifying parallel I/O support"
check_parallel "$python"
"$python" -c 'import xarray; print("xarray", xarray.__version__)'

phase "Parallel extension linkage"
for extension in \
    "$("$python" -c 'import h5py.h5f; print(h5py.h5f.__file__)')" \
    "$("$python" -c 'import netCDF4._netCDF4; print(netCDF4._netCDF4.__file__)')"; do
    echo "$extension"
    ldd "$extension" | grep -Ei 'hdf5|netcdf|mpi' || true
done
phase "Parallel MPI/HDF5/NetCDF stack installed successfully"
