"""Remove only xarray, mpi4py, h5py, and netCDF4 from the target environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("xarray", "mpi4py", "h5py", "netCDF4")


def main() -> None:
    prefix = Path(sys.prefix)
    metadata = prefix / "conda-meta"
    if sys.prefix == sys.base_prefix and not metadata.is_dir():
        raise RuntimeError("Run cleanup with a Conda environment or virtualenv Python")
    wanted = {name.lower() for name in PACKAGES}
    installed = []
    for record in metadata.glob("*.json"):
        name = json.loads(record.read_text())["name"]
        if name.lower() in wanted:
            installed.append(name)
    if installed:
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda:
            raise RuntimeError("Conda is required to remove Conda-owned stack packages")
        subprocess.run(
            [
                conda,
                "remove",
                "--prefix",
                str(prefix),
                "--force-remove",
                "--yes",
                *sorted(installed),
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", *PACKAGES], check=True
    )


if __name__ == "__main__":
    main()
