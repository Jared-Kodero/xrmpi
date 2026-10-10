"""Run every script in ``mpi_cases/`` under ``mpiexec`` on several rank counts.

A script exits non-zero when any rank recorded a failed comparison against a
serial reference, so the assertion here is on the exit status; the scripts'
own output is attached to the failure message.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mpi4py")

HERE = Path(__file__).resolve().parent
CASES = sorted(path.stem for path in (HERE / "mpi_cases").glob("case_*.py"))
RANKS = (1, 2, 3, 4)


def _launcher() -> str | None:
    """Return an MPI launcher, preferring the one beside this interpreter."""
    beside = Path(sys.executable).with_name("mpiexec")
    if beside.exists():
        return str(beside)
    return shutil.which("mpiexec") or shutil.which("mpirun")


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("case", CASES)
def test_mpi_case(case: str, ranks: int) -> None:
    launcher = _launcher()
    if launcher is None:
        pytest.skip("no MPI launcher available")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HDF5_USE_FILE_LOCKING="FALSE")
    result = subprocess.run(
        [
            launcher,
            "-n",
            str(ranks),
            sys.executable,
            str(HERE / "mpi_cases" / f"{case}.py"),
        ],
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("XRMPI_TEST_TIMEOUT", "600")),
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"{case} failed on {ranks} rank(s)\n--- stdout ---\n{result.stdout[-6000:]}"
        f"\n--- stderr ---\n{result.stderr[-3000:]}"
    )
    assert "RESULT passed=" in result.stdout
