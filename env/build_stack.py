"""Build the native MPI, HDF5, and NetCDF libraries."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BUILD_PREFIX = Path(
    os.environ.get("XRMPI_STACK_PREFIX", str(Path.home() / ".local/hpc_parallel_io"))
)
BUILD_JOBS = int(os.getenv("HPC_STACK_BUILD_N_JOBS", "4"))


@dataclass(frozen=True)
class Stack:
    mpi_module: str
    netcdf_module: str
    mpicc: Path
    netcdf_root: Path
    hdf5_root: Path


STABLE_TAG = re.compile(r"(?:hdf5[_-])?v?(\d+(?:[._]\d+)+)")


def newest_stable(tags: list[str]) -> tuple[str, str] | None:
    """Return the highest stable version and its tag, ignoring pre-releases."""
    releases = [
        (tuple(int(part) for part in re.split(r"[._]", match[1])), tag)
        for tag in tags
        if (match := STABLE_TAG.fullmatch(tag))
    ]
    if not releases:
        return None
    numbers, tag = max(releases)
    return ".".join(map(str, numbers)), tag


def latest_release(repo: str) -> tuple[str, str]:
    """Return the newest stable release version and tag from the git tags."""
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", f"https://github.com/{repo}.git"],
        check=True,
        capture_output=True,
        text=True,
    )
    release = newest_stable(
        [line.rpartition("refs/tags/")[2] for line in result.stdout.splitlines()]
    )
    if release is None:
        raise RuntimeError(f"No stable release tag found for {repo}")
    print(f"Latest {repo} release: {release[1]}", file=sys.stderr)
    return release


def clone_source(repo: str, ref: str, source_dir: Path) -> None:
    """Shallow-clone one tag or branch of a GitHub repository."""
    print(f"Cloning {repo} at {ref}...", file=sys.stderr)
    subprocess.run(
        [
            "git",
            "-c",
            "advice.detachedHead=false",
            "clone",
            "--depth",
            "1",
            "--branch",
            ref,
            f"https://github.com/{repo}.git",
            str(source_dir),
        ],
        check=True,
        stdout=sys.stderr,
    )


def download_and_extract(url: str, archive: Path, source_dir: Path) -> None:
    """Download the Open MPI tarball; its git tree needs autotools and submodules."""
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
    version, _ = latest_release("open-mpi/ompi")
    archive = work_dir / f"openmpi-{version}.tar.gz"
    source_dir = work_dir / "openmpi"
    release_series = ".".join(version.split(".")[:2])
    url = f"https://download.open-mpi.org/release/open-mpi/v{release_series}/openmpi-{version}.tar.gz"

    print(f"Downloading Open MPI {version}...", file=sys.stderr)
    download_and_extract(url, archive, source_dir)
    for command in (
        ["./configure", f"--prefix={prefix}"],
        ["make", f"-j{BUILD_JOBS}"],
        ["make", "install"],
    ):
        subprocess.run(command, cwd=source_dir, check=True, stdout=sys.stderr)

    mpicc = prefix / "bin" / "mpicc"
    if not mpicc.is_file():
        raise RuntimeError("Open MPI built successfully, but mpicc was not installed")
    return mpicc


def conda_prefixes(env: dict[str, str] | None) -> list[str]:
    """Return Conda prefixes visible to the build, in literal and resolved forms."""
    env = os.environ if env is None else env
    candidates = [Path(sys.prefix)]
    for name in ("CONDA_PREFIX", "CONDA_EXE"):
        if value := os.environ.get(name) or env.get(name):
            path = Path(value)
            candidates.append(path.parents[1] if name == "CONDA_EXE" else path)
    for entry in env.get("PATH", "").split(os.pathsep):
        if entry and (Path(entry).parent / "conda-meta").is_dir():
            candidates.append(Path(entry).parent)
    return list(
        dict.fromkeys(
            str(form)
            for path in candidates
            if (path / "conda-meta").is_dir()
            for form in (path, path.resolve())
        )
    )


def _cmake_build(
    source_dir: Path,
    prefix: Path,
    mpicc: Path,
    options: dict[str, str],
    env: dict[str, str] | None,
) -> None:
    build_dir = source_dir.with_name(source_dir.name + "-build")
    # The shared native prefix must not link libraries from any Conda prefix.
    ignored = conda_prefixes(env)
    settings = {
        "CMAKE_INSTALL_PREFIX": str(prefix),
        "CMAKE_C_COMPILER": str(mpicc),
        "CMAKE_PREFIX_PATH": str(prefix),
        "CMAKE_IGNORE_PREFIX_PATH": ";".join(ignored),
        "CMAKE_IGNORE_PATH": ";".join(
            f"{root}/{name}" for root in ignored for name in ("include", "lib", "lib64")
        ),
        "CMAKE_INSTALL_LIBDIR": "lib",
        "CMAKE_INSTALL_RPATH": str(prefix / "lib"),
        "BUILD_SHARED_LIBS": "ON",
    } | options
    for command in (
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            *[f"-D{key}={value}" for key, value in settings.items()],
        ],
        ["cmake", "--build", str(build_dir), "--parallel", str(BUILD_JOBS)],
        ["cmake", "--install", str(build_dir)],
    ):
        subprocess.run(command, check=True, stdout=sys.stderr, env=env)


def build_parallel_hdf5(
    work_dir: Path,
    prefix: Path,
    mpicc: Path,
    *,
    version: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    version = version or os.environ.get("XRMPI_HDF5_VERSION")
    ref = os.environ.get("XRMPI_HDF5_TAG")
    if ref is None and version:
        # HDF5 tags drop the "hdf5_" prefix from 2.1.1 onward.
        ref = f"hdf5_{version}" if version.startswith("1.") else version
    ref = ref or latest_release("HDFGroup/hdf5")[1]
    source_dir = work_dir / "hdf5"
    clone_source("HDFGroup/hdf5", ref, source_dir)
    _cmake_build(
        source_dir,
        prefix,
        mpicc,
        {
            "HDF5_ENABLE_PARALLEL": "ON",
            # NetCDF-4 requires zlib; HDF5 2.x renamed the option, default OFF.
            "HDF5_ENABLE_ZLIB_SUPPORT": "ON",
            "HDF5_ENABLE_Z_LIB_SUPPORT": "ON",
            "ZLIB_USE_EXTERNAL": "OFF",
            "HDF5_BUILD_CPP_LIB": "OFF",
            "HDF5_BUILD_FORTRAN": "OFF",
            "HDF5_BUILD_JAVA": "OFF",
            "HDF5_BUILD_EXAMPLES": "OFF",
            "HDF5_BUILD_TOOLS": "ON",
            "BUILD_TESTING": "OFF",
        },
        env,
    )


def build_parallel_netcdf(
    work_dir: Path,
    prefix: Path,
    mpicc: Path,
    *,
    version: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    version = version or os.environ.get("XRMPI_NETCDF_VERSION")
    ref = os.environ.get("XRMPI_NETCDF_TAG") or (f"v{version}" if version else None)
    ref = ref or latest_release("Unidata/netcdf-c")[1]
    source_dir = work_dir / "netcdf-c"
    clone_source("Unidata/netcdf-c", ref, source_dir)
    _cmake_build(
        source_dir,
        prefix,
        mpicc,
        {
            "HDF5_ROOT": str(prefix),
            "HDF5_PREFER_PARALLEL": "ON",
            "ENABLE_NETCDF_4": "ON",
            "ENABLE_DAP": "OFF",
            "ENABLE_NCZARR": "OFF",
            "ENABLE_HDF4": "OFF",
            "ENABLE_TESTS": "OFF",
            "NETCDF_ENABLE_HDF5": "ON",
            "NETCDF_ENABLE_HDF4": "OFF",
            "NETCDF_ENABLE_DAP": "OFF",
            "NETCDF_ENABLE_NCZARR": "OFF",
            "NETCDF_ENABLE_TESTS": "OFF",
        },
        env,
    )


def build_source_stack(
    selected: Stack | None = None,
    *,
    prefix: Path = BUILD_PREFIX,
    env: dict[str, str] | None = None,
) -> Stack:
    """Build custom native libraries with one MPI compiler and configured versions."""
    env = dict(os.environ if env is None else env)
    prefix.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        existing_mpicc = (
            str(selected.mpicc)
            if selected
            else shutil.which("mpicc", path=env.get("PATH"))
        )
        if existing_mpicc is None:
            print("No mpicc found in PATH; building Open MPI...", file=sys.stderr)
            mpicc = build_openmpi(work_dir, prefix)
        else:
            mpicc = Path(existing_mpicc)

        env |= {"HDF5_DIR": str(prefix), "NETCDF4_DIR": str(prefix)}
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                str(prefix / "lib"),
                str(prefix / "lib64"),
                env.get("LD_LIBRARY_PATH", ""),
            ]
        ).rstrip(os.pathsep)
        env["PATH"] = os.pathsep.join([str(mpicc.parent), env.get("PATH", "")])
        for name in (
            "HDF5_INCDIR",
            "HDF5_INCLUDEDIR",
            "HDF5_LIBDIR",
            "HDF5_PKGCONFIG_NAME",
        ):
            env.pop(name, None)
        build_parallel_hdf5(work_dir, prefix, mpicc, env=env)
        build_parallel_netcdf(work_dir, prefix, mpicc, env=env)

    return Stack(
        mpi_module=selected.mpi_module if selected else "source-build",
        netcdf_module="source-build",
        mpicc=mpicc,
        netcdf_root=prefix,
        hdf5_root=prefix,
    )
