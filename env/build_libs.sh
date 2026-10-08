#!/bin/bash -l
# Usage: bash env/build_libs.sh <environment-prefix-or-conda-name>

target="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"

# Restart as a login shell when invoked with `bash env/build_libs.sh`.
if ! shopt -q login_shell; then
    if (( $# == 0 )) && [[ -n "$target" ]]; then
        set -- "$target"
    fi
    exec bash -l "$0" "$@"
fi

set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ $# -gt 0 && "$1" != --* ]]; then
    target="$1"
    shift
fi

if (( $# )); then
    echo "Usage: bash env/build_libs.sh [environment-prefix-or-conda-name]" >&2
    exit 1
fi

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

export XRMPI_STACK_PREFIX="${XRMPI_STACK_PREFIX:-$HOME/.local/hpc_parallel_io}"
python="$env_prefix/bin/python"
export PATH="$env_prefix/bin:$PATH"

"$python" "$script_dir/config_stack.py" --clean

# Use the system module stack; build HDF5/NetCDF-C only when none is usable.
stack="$("$python" "$script_dir/config_stack.py" --find)"
eval "$stack"

if [[ -n "$MPI_MODULE" ]]; then
    module purge
    module load "$MPI_MODULE"
    if [[ -n "$NETCDF_MODULE" ]]; then
        module load "$NETCDF_MODULE"
    fi
fi

mpi_prefix="$(dirname "$(dirname "$MPICC")")"
library_path="$HDF5_DIR/lib:$HDF5_DIR/lib64:$NETCDF4_DIR/lib:$NETCDF4_DIR/lib64:$mpi_prefix/lib:$mpi_prefix/lib64"

export HDF5_DIR NETCDF4_DIR
export HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$(dirname "$MPICC"):$env_prefix/bin:$PATH"
export MPI4PY_BUILD_MPICC="$MPICC"

hdf5_libdir="$HDF5_LIBDIR"
unset HDF5_INCDIR HDF5_INCLUDEDIR HDF5_LIBDIR HDF5_PKGCONFIG_NAME

printf '\nParallel I/O build roots\n'
printf 'Python: %s\nMPI module: %s\nmpicc: %s\nHDF5: %s\nNetCDF-C: %s\n' \
    "$python" "$MPI_MODULE" "$MPICC" "$HDF5_DIR" "$NETCDF4_DIR"

"$python" "$script_dir/conda_hooks.py" \
    --prefix "$env_prefix" \
    --env-name "${env_prefix##*/}" \
    --mpi-module "$MPI_MODULE" \
    --netcdf-module "$NETCDF_MODULE" \
    --hdf5-dir "$HDF5_DIR" \
    --netcdf4-dir "$NETCDF4_DIR" \
    --hdf5-libdir "$hdf5_libdir" \
    --mpicc "$MPICC" \
    --library-path "$library_path"

export CC="$MPICC"
export HDF5_MPI=ON
export H5PY_SETUP_REQUIRES=0

# Conda Python's link flags put -L$CONDA_PREFIX/lib first, where Conda's serial
# HDF5/NetCDF remain for other packages; link and rpath the parallel libraries.
LDSHARED="$MPICC -shared"
IFS=: read -ra library_dirs <<< "$library_path"
for library_dir in "${library_dirs[@]}"; do
    if [[ -d "$library_dir" ]]; then
        LDSHARED+=" -L$library_dir -Wl,-rpath,$library_dir"
    fi
done
export LDSHARED

for package in mpi4py h5py netCDF4; do
    printf '\nBuilding %s with pip --no-deps\n' "$package"
    "$python" -m pip install \
        --no-cache-dir \
        --no-deps \
        --no-build-isolation \
        "--no-binary=$package" \
        --force-reinstall \
        "$package"
done

unset CC HDF5_MPI H5PY_SETUP_REQUIRES LDSHARED

"$python" -m pip install \
    --no-cache-dir \
    --no-deps \
    --no-build-isolation \
    --force-reinstall \
    xarray

"$python" "$script_dir/config_stack.py" --verify
