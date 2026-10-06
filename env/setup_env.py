#!/usr/bin/env python3
"""Set up xrmpi's own environment and rebuild its parallel I/O stack.

Use an active non-base Conda environment or virtualenv, or pass a Conda name.
Without either, create/update the xmpi environment from environment.yml.
The separate .workspace setup does not call this installer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
STACK_PACKAGES = {"xarray", "mpi4py", "h5py", "netcdf4"}
BUILD_REQUIREMENTS = [
    "pip",
    "setuptools>=77",
    "wheel",
    "cython>=3,<4",
    "numpy",
    "packaging",
    "pkgconfig",
    "cftime",
    "certifi",
]
READ_DEPENDENCIES = """import json, sys, tomllib
from pathlib import Path
projects = []
for name in sys.argv[1:]:
    with (Path(name) / 'pyproject.toml').open('rb') as stream:
        projects.append(tomllib.load(stream)['project'])
print(json.dumps(projects))
"""


def run(
    command: list[str], *, env: dict[str, str] | None = None, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=True, env=env, text=True, capture_output=capture
    )


def conda_executable() -> Path | None:
    candidate = os.environ.get("CONDA_EXE") or shutil.which("conda")
    return Path(candidate).resolve() if candidate else None


def install_miniconda() -> Path:
    prefix = Path.home() / "miniconda3"
    conda = prefix / "bin" / "conda"
    if conda.is_file():
        return conda
    prefix.mkdir(parents=True, exist_ok=True)
    installer = prefix / "miniconda.sh"
    with (
        urllib.request.urlopen(MINICONDA_URL, timeout=60) as response,
        installer.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    try:
        run(["bash", str(installer), "-b", "-u", "-p", str(prefix)])
    finally:
        installer.unlink(missing_ok=True)
    return conda


def conda_envs(conda: Path) -> dict[str, Path]:
    result = run([str(conda), "env", "list", "--json"], capture=True)
    return {
        Path(prefix).name: Path(prefix) for prefix in json.loads(result.stdout)["envs"]
    }


def environment_prefix(
    env_name: str | None, environment_file: Path
) -> tuple[Path, Path | None]:
    conda = conda_executable()
    if env_name is None:
        virtualenv = os.environ.get("VIRTUAL_ENV")
        if virtualenv:
            return Path(virtualenv).resolve(), None
        active_conda = os.environ.get("CONDA_DEFAULT_ENV")
        active_prefix = os.environ.get("CONDA_PREFIX")
        if active_conda and active_conda != "base" and active_prefix:
            if conda is None:
                raise RuntimeError(
                    "Conda executable not found for the active environment"
                )
            prefix = Path(active_prefix).resolve()
            run(
                [
                    str(conda),
                    "env",
                    "update",
                    "--prefix",
                    str(prefix),
                    "--file",
                    str(environment_file),
                ]
            )
            return prefix, conda
    env_name = env_name or "xmpi"
    if env_name == "base":
        raise RuntimeError("Use a non-base environment for xrmpi")
    conda = conda or install_miniconda()
    existing = conda_envs(conda)
    action = "update" if env_name in existing else "create"
    run(
        [
            str(conda),
            "env",
            action,
            "--name",
            env_name,
            "--file",
            str(environment_file),
        ]
    )
    return conda_envs(conda)[env_name], conda


def install_dependencies(python: Path, projects: list[Path]) -> None:
    """Install ordinary requirements before replacing the four stack packages."""
    result = run(
        [str(python), "-c", READ_DEPENDENCIES, *map(str, projects)], capture=True
    )
    metadata = json.loads(result.stdout)
    local_names = {project["name"].lower() for project in metadata}
    requirements = set(BUILD_REQUIREMENTS)
    for project in metadata:
        for requirement in project.get("dependencies", []):
            name = re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0].lower()
            if name not in STACK_PACKAGES | local_names:
                requirements.add(requirement)
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *sorted(requirements),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "env_name", nargs="?", help="Conda name; default: active env or xmpi"
    )
    args = parser.parse_args()
    env_dir = Path(__file__).resolve().parent
    repo_dir = env_dir.parent
    prefix, conda = environment_prefix(args.env_name, env_dir / "environment.yml")
    python = prefix / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Target environment Python not found: {python}")
    run([str(python), "-c", "import sys; sys.exit(sys.version_info < (3, 14))"])

    projects = [repo_dir]
    xgeo_dir = repo_dir.parent / "xgeo"
    if (xgeo_dir / "pyproject.toml").is_file():
        projects.insert(0, xgeo_dir)
    install_dependencies(python, projects)

    build_env = dict(os.environ)
    build_env["PATH"] = str(prefix / "bin") + os.pathsep + build_env.get("PATH", "")
    if conda is not None:
        build_env["CONDA_EXE"] = str(conda)
        build_env["CONDA_PREFIX"] = str(prefix)
    else:
        build_env.pop("CONDA_PREFIX", None)
        build_env["VIRTUAL_ENV"] = str(prefix)
    run(["bash", str(env_dir / "build_libs.sh"), str(prefix)], env=build_env)

    for project in projects:
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--editable",
                str(project),
            ]
        )
    run(
        [str(python), "-m", "pip", "check"],
        env=build_env,
    )
    print(f"Installed xrmpi in {prefix}")
    if conda is not None:
        print(f"Reactivate the environment: conda activate {prefix}")
    else:
        print(f"Source the stack hook: {prefix}/etc/conda/activate.d/mpi-netcdf.sh")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
