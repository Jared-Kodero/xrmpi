from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MPI_QUERY = "hpcx-mpi"
NETCDF_QUERY = "netcdf-mpi"
BUILD_PREFIX = Path(
    os.environ.get(
        "XRMPI_STACK_PREFIX", str(Path.home() / ".local" / "hpc_parallel_io")
    )
)
BUILD_JOBS = int(os.getenv("HPC_STACK_BUILD_N_JOBS", "4"))

PROBE_C = """#include <mpi.h>
#include <netcdf.h>
#include <netcdf_meta.h>
#include <netcdf_par.h>
#if !defined(NC_HAS_PARALLEL4) || !NC_HAS_PARALLEL4
#error "no parallel netcdf4"
#endif
int main(void) {
    int ncid;
    return nc_create_par("probe.nc", NC_CLOBBER, MPI_COMM_SELF, MPI_INFO_NULL, &ncid);
}
"""


@dataclass(frozen=True)
class Stack:
    mpi_module: str
    netcdf_module: str
    mpicc: Path
    netcdf_root: Path
    hdf5_root: Path


def bash(script: str) -> subprocess.CompletedProcess[str]:
    """Run a script in a login shell, where the HPC `module` function exists."""
    return subprocess.run(
        ["bash", "-lc", script],
        capture_output=True,
        text=True,
        check=False,
    )


def module_names(query: str) -> list[str]:
    """Return matching module versions in descending natural-version order."""
    result = bash(f"module -t spider {shlex.quote(query)} 2>&1")
    names = set(re.findall(rf"{re.escape(query)}/[^\s,:;)]+", result.stdout))
    return sorted(
        names,
        key=lambda name: [
            (0, int(part), "") if part.isdigit() else (1, 0, part)
            for part in re.split(r"(\d+)", name)
            if part
        ],
        reverse=True,
    )


def probe_module_stack(mpi_module: str, netcdf_module: str) -> Stack | None:
    """Validate one loaded MPI/NetCDF pair and identify its linked HDF5."""
    module_setup = f"""
module purge >/dev/null 2>&1 || exit 1
module load {shlex.quote(mpi_module)} >/dev/null 2>&1 || exit 1
module load {shlex.quote(netcdf_module)} >/dev/null 2>&1 || exit 1
command -v mpicc >/dev/null 2>&1 || exit 1
command -v nc-config >/dev/null 2>&1 || exit 1
"""

    compile_probe = f"""
probe_exe=$(mktemp) || exit 1
trap 'rm -f "$probe_exe"' EXIT
printf '%s' {shlex.quote(PROBE_C)} \\
    | mpicc $(nc-config --cflags) -x c - $(nc-config --libs) -o "$probe_exe" \\
        >/dev/null 2>&1 || exit 1
"""

    resolve_netcdf = """
mpicc_path=$(command -v mpicc) || exit 1
netcdf_root=$(nc-config --prefix) || exit 1

netcdf_lib=''
for libdir in "$netcdf_root/lib" "$netcdf_root/lib64"; do
    if [ -e "$libdir/libnetcdf.so" ]; then
        netcdf_lib="$libdir/libnetcdf.so"
        break
    fi
done
[ -n "$netcdf_lib" ] || exit 1
"""

    resolve_hdf5 = r"""
hdf5_lib=$(ldd "$netcdf_lib" 2>/dev/null \
    | sed -nE 's|^[[:space:]]*libhdf5[^[:space:]]*\.so[^[:space:]]*[[:space:]]+=>[[:space:]]+(/[^[:space:]]+).*|\1|p' \
    | head -n 1)
[ -n "$hdf5_lib" ] || exit 1
hdf5_root=$(dirname "$(dirname "$hdf5_lib")")
"""

    emit_stack = """
printf '__STACK_MPICC__=%s\\n' "$mpicc_path"
printf '__STACK_NETCDF__=%s\\n' "$netcdf_root"
printf '__STACK_HDF5__=%s\\n' "$hdf5_root"
"""

    script = f"{module_setup}\n{compile_probe}\n{resolve_netcdf}\n{resolve_hdf5}\n{emit_stack}"

    result = bash(script)
    if result.returncode != 0:
        return None

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.startswith("__STACK_") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key] = value

    required = {
        "__STACK_MPICC__",
        "__STACK_NETCDF__",
        "__STACK_HDF5__",
    }
    if not required <= values.keys():
        return None

    return Stack(
        mpi_module=mpi_module,
        netcdf_module=netcdf_module,
        mpicc=Path(values["__STACK_MPICC__"]),
        netcdf_root=Path(values["__STACK_NETCDF__"]),
        hdf5_root=Path(values["__STACK_HDF5__"]),
    )


def find_module_stack() -> Stack | None:
    """Return the newest compatible NetCDF/MPI module combination."""
    mpi_modules = module_names(MPI_QUERY)
    netcdf_modules = module_names(NETCDF_QUERY)

    for netcdf_module in netcdf_modules:
        for mpi_module in mpi_modules:
            print(f"Trying {mpi_module} + {netcdf_module}", file=sys.stderr)
            stack = probe_module_stack(mpi_module, netcdf_module)
            if stack is not None:
                return stack
    return None


def latest_release(repo: str, fallback_tag: str) -> tuple[str, str]:
    """Return the latest release version and exact GitHub tag."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/latest",
        headers={"User-Agent": "HPC-Stack-Resolver"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        payload = {}

    tag = str(payload.get("tag_name", "")) if isinstance(payload, dict) else ""
    pattern = r"(?:^v?|[_-])(\d+(?:[._]\d+)+)"
    match = re.search(pattern, tag)
    if match is not None:
        return match.group(1).replace("_", "."), tag

    fallback_match = re.search(pattern, fallback_tag)
    if fallback_match is None:
        raise ValueError(f"fallback tag has no version: {fallback_tag}")
    return fallback_match.group(1).replace("_", "."), fallback_tag


def download_and_extract(url: str, archive: Path, source_dir: Path) -> None:
    """Download a gzip tarball and extract its top-level directory into source_dir."""
    source_dir.mkdir(parents=True)
    request = urllib.request.Request(url, headers={"User-Agent": "HPC-Stack-Resolver"})
    with (
        urllib.request.urlopen(request, timeout=30) as response,
        archive.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    subprocess.run(
        [
            "tar",
            "-xzf",
            str(archive),
            "-C",
            str(source_dir),
            "--strip-components=1",
        ],
        check=True,
        stdout=sys.stderr,
    )


def build_openmpi(work_dir: Path, prefix: Path) -> Path:
    """Build Open MPI and return its mpicc path."""
    version, _ = latest_release("open-mpi/ompi", "v5.0.3")
    archive = work_dir / f"openmpi-{version}.tar.gz"
    source_dir = work_dir / "openmpi"
    release_series = ".".join(version.split(".")[:2])
    url = f"https://download.open-mpi.org/release/open-mpi/v{release_series}/openmpi-{version}.tar.gz"

    print(f"Downloading Open MPI {version}...", file=sys.stderr)
    download_and_extract(url, archive, source_dir)
    subprocess.run(
        ["./configure", f"--prefix={prefix}"],
        cwd=source_dir,
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["make", f"-j{BUILD_JOBS}"], cwd=source_dir, check=True, stdout=sys.stderr
    )
    subprocess.run(["make", "install"], cwd=source_dir, check=True, stdout=sys.stderr)

    mpicc = prefix / "bin" / "mpicc"
    if not mpicc.is_file():
        raise RuntimeError("Open MPI built successfully, but mpicc was not installed")
    return mpicc


def build_parallel_hdf5(work_dir: Path, prefix: Path, mpicc: Path) -> None:
    """Build shared HDF5 with MPI support."""
    version, tag = latest_release("HDFGroup/hdf5", "hdf5_1.14.6")
    archive = work_dir / f"hdf5-{version}.tar.gz"
    source_dir = work_dir / "hdf5"
    build_dir = work_dir / "hdf5-build"
    url = f"https://github.com/HDFGroup/hdf5/archive/refs/tags/{tag}.tar.gz"

    print(f"Downloading HDF5 {version}...", file=sys.stderr)
    download_and_extract(url, archive, source_dir)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DCMAKE_C_COMPILER={mpicc}",
            "-DHDF5_ENABLE_PARALLEL=ON",
            "-DBUILD_SHARED_LIBS=ON",
            "-DHDF5_BUILD_CPP_LIB=OFF",
            "-DHDF5_BUILD_FORTRAN=OFF",
            "-DHDF5_BUILD_JAVA=OFF",
            "-DHDF5_BUILD_EXAMPLES=OFF",
            "-DHDF5_BUILD_TOOLS=OFF",
            "-DBUILD_TESTING=OFF",
        ],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", str(BUILD_JOBS)],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["cmake", "--install", str(build_dir)], check=True, stdout=sys.stderr
    )


def build_parallel_netcdf(work_dir: Path, prefix: Path, mpicc: Path) -> None:
    """Build shared NetCDF-C against the parallel HDF5 installation."""
    version, tag = latest_release("Unidata/netcdf-c", "v4.10.1")
    archive = work_dir / f"netcdf-c-{version}.tar.gz"
    source_dir = work_dir / "netcdf-c"
    build_dir = work_dir / "netcdf-c-build"
    url = f"https://github.com/Unidata/netcdf-c/archive/refs/tags/{tag}.tar.gz"

    print(f"Downloading NetCDF-C {version}...", file=sys.stderr)
    download_and_extract(url, archive, source_dir)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DCMAKE_C_COMPILER={mpicc}",
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DHDF5_ROOT={prefix}",
            "-DHDF5_PREFER_PARALLEL=ON",
            "-DNETCDF_ENABLE_HDF5=ON",
            "-DNETCDF_ENABLE_HDF4=OFF",
            "-DNETCDF_ENABLE_DAP=OFF",
            "-DNETCDF_ENABLE_NCZARR=OFF",
            "-DNETCDF_ENABLE_TESTS=OFF",
            "-DBUILD_SHARED_LIBS=ON",
        ],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", str(BUILD_JOBS)],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["cmake", "--install", str(build_dir)], check=True, stdout=sys.stderr
    )


def build_source_stack() -> Stack:
    """Construct an MPI + parallel HDF5 + NetCDF-C stack under BUILD_PREFIX."""
    BUILD_PREFIX.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        existing_mpicc = shutil.which("mpicc")
        if existing_mpicc is None:
            print("No mpicc found in PATH; building Open MPI...", file=sys.stderr)
            mpicc = build_openmpi(work_dir, BUILD_PREFIX)
        else:
            mpicc = Path(existing_mpicc)

        build_parallel_hdf5(work_dir, BUILD_PREFIX, mpicc)
        build_parallel_netcdf(work_dir, BUILD_PREFIX, mpicc)

    return Stack(
        mpi_module="source-build",
        netcdf_module="source-build",
        mpicc=mpicc,
        netcdf_root=BUILD_PREFIX,
        hdf5_root=BUILD_PREFIX,
    )


def macro_is_one(header: Path, macro: str) -> bool:
    """Return whether a generated C header defines macro as integer 1."""
    if not header.is_file():
        return False
    pattern = rf"^#define\s+{re.escape(macro)}\s+1\b"
    return (
        re.search(pattern, header.read_text(errors="replace"), re.MULTILINE) is not None
    )


def validate_stack(stack: Stack) -> Path:
    """Validate NetCDF-4/HDF5 parallel support and return the HDF5 lib directory."""
    netcdf_header = stack.netcdf_root / "include" / "netcdf_meta.h"
    if not macro_is_one(netcdf_header, "NC_HAS_PARALLEL4"):
        raise RuntimeError(f"{stack.netcdf_root} is not a parallel NetCDF-4 build")

    netcdf_libdirs = (stack.netcdf_root / "lib", stack.netcdf_root / "lib64")
    if not any((libdir / "libnetcdf.so").exists() for libdir in netcdf_libdirs):
        raise RuntimeError(f"could not locate libnetcdf.so below {stack.netcdf_root}")

    hdf5_header = stack.hdf5_root / "include" / "H5pubconf.h"
    if not macro_is_one(hdf5_header, "H5_HAVE_PARALLEL"):
        raise RuntimeError(f"HDF5 at {stack.hdf5_root} is not MPI-enabled")

    for libdir in (stack.hdf5_root / "lib", stack.hdf5_root / "lib64"):
        if any(libdir.glob("libhdf5*.so*")):
            return libdir
    raise RuntimeError(
        f"could not locate an HDF5 shared library below {stack.hdf5_root}"
    )


def print_stack(stack: Stack, hdf5_libdir: Path) -> None:
    """Emit shell-safe assignments for downstream build scripts."""
    values = {
        "MPI_MODULE": "" if stack.mpi_module == "source-build" else stack.mpi_module,
        "NETCDF_MODULE": ""
        if stack.netcdf_module == "source-build"
        else stack.netcdf_module,
        "MPICC": str(stack.mpicc),
        "NETCDF4_DIR": str(stack.netcdf_root),
        "HDF5_DIR": str(stack.hdf5_root),
        "HDF5_LIBDIR": str(hdf5_libdir),
    }
    for name, value in values.items():
        print(f"{name}={shlex.quote(value)}")


def main() -> int:
    stack = find_module_stack()
    if stack is None:
        print(
            "No compatible HPC module stack found; falling back to source build...",
            file=sys.stderr,
        )
        stack = build_source_stack()

    hdf5_libdir = validate_stack(stack)
    print_stack(stack, hdf5_libdir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
