

PHASE_COUNT=0

phase() {
    local text="$*"
    local n_cols

    PHASE_COUNT=$((PHASE_COUNT + 1))
    n_cols="$(tput cols 2>/dev/null || printf '80')"

    printf '\n[%d] %s\n' "$PHASE_COUNT" "$text"
    printf '%*s\n' "$n_cols" '' | tr ' ' '-'
}

check_parallel() {
    "$1" -c '
import mpi4py
mpi4py.rc.initialize = False
import h5py
import netCDF4
import sys

sys.exit(0 if h5py.get_config().mpi and netCDF4.__has_parallel4_support__ else 1)
'
}

check_pnetcdf() {
    local env_name="$1"
    local script="$2"
    conda activate "$env_name"
    if ! check_parallel "$(command -v python)"; then
        bash "$script" "$env_name"
        check_parallel "$(command -v python)"
    fi
    conda deactivate
}
