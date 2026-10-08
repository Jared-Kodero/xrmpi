"""Configure the MPI, HDF5, NetCDF, and Python I/O stack.

Build actions emit Bash assignments; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from build_stack import BUILD_PREFIX, Stack, build_source_stack

PIP_PACKAGES = ("mpi4py", "h5py", "netCDF4", "xarray")
LIBRARY = re.compile(r"^lib(?:hdf5(?:[_\-.]|$)|netcdf(?:\.|$))")

MPI_QUERY = "hpcx-mpi"
NETCDF_QUERY = "netcdf-mpi"
PROBE_C = """#include <mpi.h>
#include <hdf5.h>
#include <netcdf.h>
#include <netcdf_meta.h>
#include <netcdf_par.h>
#if !defined(NC_HAS_PARALLEL4) || !NC_HAS_PARALLEL4
#error "no parallel netcdf4"
#endif
#if !defined(H5_HAVE_PARALLEL) || !H5_HAVE_PARALLEL
#error "no parallel hdf5"
#endif
int main(void) {
    int ncid;
    hid_t access = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(access, MPI_COMM_SELF, MPI_INFO_NULL);
    H5Pclose(access);
    return nc_create_par("probe.nc", NC_NETCDF4 | NC_CLOBBER,
                         MPI_COMM_SELF, MPI_INFO_NULL, &ncid);
}
"""


def conda_records(prefix: Path) -> dict[str, dict]:
    """Read package ownership metadata without traversing dependency graphs."""
    records = {}
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        record = json.loads(path.read_text())
        records[record["name"]] = record
    return records


def remove_libraries(prefix: Path) -> None:
    """Remove HDF5 and NetCDF files from lib and lib64."""
    paths = []
    visited = set()
    for directory in (prefix / "lib", prefix / "lib64"):
        if not directory.exists():
            continue
        if not directory.resolve().is_relative_to(prefix):
            raise RuntimeError(
                f"Refusing to clean an external library directory: {directory}"
            )
        if directory.resolve() in visited:
            continue
        visited.add(directory.resolve())
        paths.extend(path for path in directory.iterdir() if LIBRARY.match(path.name))
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            raise RuntimeError(f"Unexpected directory in library cleanup: {path}")
    for path in sorted(paths):
        print(f"Removing residual library: {path}", file=sys.stderr, flush=True)
        path.unlink()


def clean_stack(prefix: Path) -> None:
    """Remove the Python I/O bindings; keep Conda native libraries for consumers."""
    prefix = prefix.resolve()
    metadata = prefix / "conda-meta"
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if metadata.is_dir():
        if not conda:
            raise RuntimeError("Conda is required to remove Conda-owned packages")
        info = json.loads(
            subprocess.run(
                [conda, "info", "--json"], check=True, text=True, capture_output=True
            ).stdout
        )
        if prefix == Path(info["root_prefix"]).resolve():
            raise RuntimeError("The base Conda environment cannot be cleaned")
    elif sys.prefix == sys.base_prefix:
        raise RuntimeError("Use a non-base Conda environment or virtualenv Python")

    # Conda hdf5/libnetcdf stay for gdal, cdo, nco, and esmf; only the bindings go.
    bindings = {name.lower() for name in PIP_PACKAGES}
    removed = sorted(name for name in conda_records(prefix) if name.lower() in bindings)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", *PIP_PACKAGES],
        check=True,
        stdout=sys.stderr,
    )
    if removed:
        subprocess.run(
            [
                conda,
                "remove",
                "--prefix",
                str(prefix),
                "--force-remove",
                "--yes",
                *removed,
            ],
            check=True,
            stdout=sys.stderr,
        )


def bash(script: str) -> subprocess.CompletedProcess[str]:
    """Run module commands in a login shell."""
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
    """Resolve and compile-check one MPI/NetCDF module pair."""
    try:
        env = module_environment(mpi_module, netcdf_module)
        mpicc = shutil.which("mpicc", path=env.get("PATH"))
        nc_config = shutil.which("nc-config", path=env.get("PATH"))
        if not mpicc or not nc_config:
            return None
        result = subprocess.run(
            [nc_config, "--prefix"], env=env, check=True, capture_output=True, text=True
        )
        netcdf_root = Path(result.stdout.strip())
        library = next(
            (
                netcdf_root / directory / "libnetcdf.so"
                for directory in ("lib", "lib64")
                if (netcdf_root / directory / "libnetcdf.so").is_file()
            ),
            None,
        )
        if library is None:
            return None
        result = subprocess.run(
            ["ldd", str(library)], env=env, capture_output=True, text=True, check=False
        )
        match = re.search(r"^\s*libhdf5\S*\s+=>\s+(/\S+)", result.stdout, re.MULTILINE)
        if match is None:
            return None
        stack = Stack(
            mpi_module,
            netcdf_module,
            Path(mpicc),
            netcdf_root,
            Path(match[1]).parent.parent,
        )
        compile_probe(stack, stack_environment(stack, env))
        return stack
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Rejecting {mpi_module} + {netcdf_module}: {exc}", file=sys.stderr)
        return None


def module_environment(*names: str) -> dict[str, str]:
    """Capture a login-shell environment after loading modules."""
    names = tuple(name for name in names if name and name != "source-build")
    commands = ["set -e"]
    if names:
        commands.append("module purge >&2")
        commands.extend(f"module load {shlex.quote(name)} >&2" for name in names)
    commands.append("printf '\\0__ENV__\\0'; env -0")
    result = bash("\n".join(commands))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Could not load the HPC modules")
    values = result.stdout.partition("\0__ENV__\0")[2]
    return dict(item.split("=", 1) for item in values.split("\0") if item)


def stack_environment(
    stack: Stack, env: dict[str, str] | None = None
) -> dict[str, str]:
    """Load the selected modules and capture their environment."""
    env = (
        dict(env)
        if env is not None
        else module_environment(stack.mpi_module, stack.netcdf_module)
    )
    libdirs = [
        str(root / directory)
        for root in (stack.hdf5_root, stack.netcdf_root, stack.mpicc.parent.parent)
        for directory in ("lib", "lib64")
        if (root / directory).is_dir()
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*dict.fromkeys(libdirs), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    env["PATH"] = os.pathsep.join(
        [str(stack.mpicc.parent), str(Path(sys.executable).parent), env.get("PATH", "")]
    )
    env["HDF5_DIR"] = str(stack.hdf5_root)
    env["NETCDF4_DIR"] = str(stack.netcdf_root)
    env["MPI4PY_BUILD_MPICC"] = str(stack.mpicc)
    for name in (
        "HDF5_INCDIR",
        "HDF5_INCLUDEDIR",
        "HDF5_LIBDIR",
        "HDF5_PKGCONFIG_NAME",
    ):
        env.pop(name, None)
    return env


def compile_probe(stack: Stack, env: dict[str, str]) -> Path:
    """Compile the parallel APIs using the selected compiler and library roots."""
    hdf5_libdir = validate_stack(stack)
    netcdf_libdir = next(
        directory
        for directory in (stack.netcdf_root / "lib", stack.netcdf_root / "lib64")
        if (directory / "libnetcdf.so").exists()
    )
    with tempfile.TemporaryDirectory(prefix="xrmpi-probe-") as temporary:
        source = Path(temporary) / "probe.c"
        source.write_text(PROBE_C)
        result = subprocess.run(
            [
                str(stack.mpicc),
                str(source),
                f"-I{stack.hdf5_root / 'include'}",
                f"-I{stack.netcdf_root / 'include'}",
                f"-L{hdf5_libdir}",
                f"-L{netcdf_libdir}",
                "-lnetcdf",
                "-lhdf5",
                "-o",
                str(Path(temporary) / "probe"),
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Parallel stack compile probe failed: {result.stderr.strip()}"
            )

    return hdf5_libdir


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


def validate_stack(stack: Stack) -> Path:
    """Validate NetCDF-4/HDF5 parallel support and return the HDF5 lib directory."""
    for header, macro in (
        (stack.netcdf_root / "include/netcdf_meta.h", "NC_HAS_PARALLEL4"),
        (stack.hdf5_root / "include/H5pubconf.h", "H5_HAVE_PARALLEL"),
    ):
        if not re.search(
            rf"^#define\s+{macro}\s+1\b", header.read_text(), re.MULTILINE
        ):
            raise RuntimeError(f"Parallel support missing: {macro} in {header}")
    if not any(
        (stack.netcdf_root / directory / "libnetcdf.so").is_file()
        for directory in ("lib", "lib64")
    ):
        raise RuntimeError(f"libnetcdf.so missing from {stack.netcdf_root}")

    for libdir in (stack.hdf5_root / "lib", stack.hdf5_root / "lib64"):
        if any(libdir.glob("libhdf5*.so*")):
            return libdir
    raise RuntimeError(
        f"could not locate an HDF5 shared library below {stack.hdf5_root}"
    )


def emit_stack(stack: Stack, hdf5_libdir: Path) -> None:
    """Emit Bash stack assignments."""
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


def verify_stack() -> None:
    """Verify imports and parallel capabilities of the rebuilt Python stack."""
    print("\nVerifying parallel I/O support", flush=True)
    import mpi4py

    mpi4py.rc.initialize = False
    import h5py
    import netCDF4
    import xarray
    from mpi4py import MPI

    if not (h5py.get_config().mpi and netCDF4.__has_parallel4_support__):
        raise RuntimeError("Parallel HDF5 and NetCDF-4 support are required.")
    print("MPI", MPI.Get_library_version().strip())
    print("xarray", xarray.__version__)
    print("h5py HDF5", h5py.h5.get_libversion())
    print("netCDF4 NetCDF-C", netCDF4.__netcdf4libversion__)

    # Conda's serial copies stay in the environment; require the parallel ones.
    loaded = sorted(
        {
            Path(fields[-1]).resolve()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if len(fields := line.split()) > 5 and LIBRARY.match(Path(fields[-1]).name)
        }
    )
    for path in loaded:
        print("loaded", path)
    serial = [
        path for path in loaded if path.is_relative_to(Path(sys.prefix).resolve())
    ]
    if serial:
        raise RuntimeError(f"Serial Conda HDF5/NetCDF libraries loaded: {serial}")


def find_mpi_module(native_prefix: Path) -> Stack | None:
    """Return the newest MPI module providing mpicc for a source build."""
    for mpi_module in module_names(MPI_QUERY):
        try:
            env = module_environment(mpi_module)
        except RuntimeError as exc:
            print(f"Rejecting {mpi_module}: {exc}", file=sys.stderr)
            continue
        mpicc = shutil.which("mpicc", path=env.get("PATH"))
        if mpicc:
            return Stack(
                mpi_module, "source-build", Path(mpicc), native_prefix, native_prefix
            )
    return None


def configure_stack(*, build_custom: bool = False) -> None:
    """Use a module stack; build HDF5 and NetCDF-C only when forced or none exists."""
    stack = find_module_stack()
    if stack is not None and not build_custom:
        print(
            f"Using system stack {stack.mpi_module} + {stack.netcdf_module}",
            file=sys.stderr,
        )
    else:
        if stack is None:
            print(
                "No compatible MPI/NetCDF module pair; building HDF5 and NetCDF-C",
                file=sys.stderr,
            )
        prefix = Path(sys.prefix).resolve()
        native_prefix = BUILD_PREFIX.expanduser().resolve()
        if native_prefix == prefix or any(
            native_prefix.is_relative_to((prefix / name).resolve())
            for name in ("lib", "lib64")
        ):
            raise RuntimeError(
                "Custom libraries must be outside the cleaned lib directories"
            )
        remove_libraries(native_prefix)
        # Only MPI is needed while building replacement HDF5 and NetCDF-C.
        mpi = stack or find_mpi_module(native_prefix)
        env = module_environment(mpi.mpi_module) if mpi else dict(os.environ)
        stack = build_source_stack(mpi, prefix=native_prefix, env=env)
    hdf5_libdir = compile_probe(stack, stack_environment(stack))
    emit_stack(stack, hdf5_libdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action, help_text in {
        "clean": "Remove the old Python and native I/O stack",
        "find": "Use a module stack; build from source only when none is found",
        "build": "Force a source build of parallel HDF5 and NetCDF-C",
        "verify": "Check Python parallel I/O support",
    }.items():
        actions.add_argument(
            f"--{action}",
            dest="action",
            action="store_const",
            const=action,
            help=help_text,
        )
    args = parser.parse_args()
    if args.action == "clean":
        clean_stack(Path(sys.prefix))
    elif args.action == "verify":
        verify_stack()
    else:
        configure_stack(build_custom=args.action == "build")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
