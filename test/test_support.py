"""Check independent setup, locking, and output without a native MPI install."""

from __future__ import annotations
import __future__

import ast
import fcntl
import importlib.util
import io
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from xrmpi.mpi import mpi_utils
from xrmpi.mpi.mpi_utils import SerialProgressBar

ROOT = Path(__file__).resolve().parents[1]


def load_diagnostics() -> ModuleType:
    """Load the real diagnostics implementation with only MPI types stubbed."""
    # Keep the declared dependency cached when patch.dict restores sys.modules.
    importlib.import_module("numpy")
    mpi_init = ModuleType("xrmpi.mpi.mpi_init")
    mpi_init.MPI = SimpleNamespace(Comm=object, Intracomm=object)
    spec = importlib.util.spec_from_file_location(
        "xrmpi.mpi.test_diagnostics", ROOT / "mpi/diagnostics.py"
    )
    assert spec is not None and spec.loader is not None
    diagnostics = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"xrmpi.mpi.mpi_init": mpi_init}):
        spec.loader.exec_module(diagnostics)
    return diagnostics


class OutputTests(unittest.TestCase):
    def test_imports_do_not_load_xgeo_or_mpi(self) -> None:
        script = """import importlib.abc
import sys

class RejectCoupling(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'xgeo', 'xr_mpi', 'mpi4py'}:
            raise AssertionError(f'Unexpected dependency: {fullname}')

sys.meta_path.insert(0, RejectCoupling())
from xrmpi.mpi.mpi_utils import SerialProgressBar
from xrmpi.mpi import mpi_utils
assert 'xrmpi._utils' not in sys.modules
assert not hasattr(mpi_utils, 'tmp')
print(SerialProgressBar.__module__)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip(), "xrmpi.mpi.mpi_utils")

    def test_streams_restore_after_exception(self) -> None:
        original = (sys.stdout, sys.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            destination = io.StringIO()
            progress = SerialProgressBar(
                total=1, file=destination, lockfile=Path(temporary) / "progress.lock"
            )
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with progress:
                    capture = progress._capture
                    print("stdout", flush=True)
                    print("stderr", file=sys.stderr, flush=True)
                    raise RuntimeError("stop")
            self.assertEqual(destination.getvalue(), "stdout\nstderr\n")
            assert capture is not None
            self.assertTrue(capture.closed)
            self.assertIsNone(progress._capture)
            self.assertIsNone(progress._redirect)
        self.assertEqual((sys.stdout, sys.stderr), original)

    def test_progress_preserves_existing_and_incidental_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "progress.txt"
            with destination.open("w+", encoding="utf-8") as stream:
                stream.write("existing output\n")
                stream.flush()
                progress = SerialProgressBar(
                    range(3),
                    description="Writing NetCDF file",
                    file=stream,
                    lockfile=Path(temporary) / "progress.lock",
                )
                observed: list[int] = []
                for item in progress:
                    observed.append(item)
                    print(f"record {item}", flush=True)
                stream.seek(0)
                output = stream.read()
            self.assertEqual(observed, [0, 1, 2])
            self.assertTrue(output.startswith("existing output\n"))
            for item in observed:
                self.assertIn(f"record {item}\n", output)
            self.assertEqual(output.count("100%"), 1)
            self.assertIsNone(progress._capture)

    def test_progress_restores_streams_when_iteration_fails(self) -> None:
        def failing_items() -> Iterator[int]:
            yield 1
            raise RuntimeError("stop")

        original = (sys.stdout, sys.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            destination = io.StringIO()
            progress = SerialProgressBar(
                failing_items(),
                total=3,
                file=destination,
                lockfile=Path(temporary) / "progress.lock",
            )
            with self.assertRaisesRegex(RuntimeError, "stop"):
                list(progress)
            self.assertNotIn("100%", destination.getvalue())
            self.assertIsNone(progress._capture)
        self.assertEqual((sys.stdout, sys.stderr), original)

    def test_progress_supports_python_streams_and_reuse(self) -> None:
        original = (sys.stdout, sys.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            destination = io.StringIO()
            progress = SerialProgressBar(
                total=1, file=destination, lockfile=Path(temporary) / "progress.lock"
            )
            for _ in range(2):
                with progress:
                    print("stdout")
                    print("stderr", file=sys.stderr)
                    progress.update()
                self.assertFalse(progress._started)
                self.assertEqual((sys.stdout, sys.stderr), original)
            self.assertEqual(destination.getvalue().count("100%"), 2)
            self.assertEqual(destination.getvalue().count("stdout\nstderr\n"), 2)

    def test_progress_restores_streams_when_renderer_start_fails(self) -> None:
        original = (sys.stdout, sys.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            progress = SerialProgressBar(
                total=1,
                file=io.StringIO(),
                lockfile=Path(temporary) / "progress.lock",
            )
            progress._interactive = True
            with (
                patch.object(
                    progress, "_start_interactive", side_effect=RuntimeError("renderer")
                ),
                self.assertRaisesRegex(RuntimeError, "renderer"),
            ):
                progress.__enter__()
            self.assertFalse(progress._started)
            self.assertIsNone(progress._capture)
            self.assertIsNone(progress._redirect)
        self.assertEqual((sys.stdout, sys.stderr), original)

    def test_progress_restores_streams_when_final_output_fails(self) -> None:
        original = (sys.stdout, sys.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            destination = io.StringIO()
            progress = SerialProgressBar(
                total=1, file=destination, lockfile=Path(temporary) / "progress.lock"
            )
            with self.assertRaisesRegex(OSError, "output"):
                with progress:
                    capture = progress._capture
                    print("captured")
                    progress.print_progress = Mock(side_effect=OSError("output"))
            assert capture is not None
            self.assertTrue(capture.closed)
            self.assertEqual(destination.getvalue(), "captured\n")
            self.assertFalse(progress._started)
        self.assertEqual((sys.stdout, sys.stderr), original)


class LockingTests(unittest.TestCase):
    def test_logging_and_progress_wait_for_other_process(self) -> None:
        script = """import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

mode, lock_path, destination = sys.argv[1:]
if mode == 'logging':
    mpi_init = ModuleType('xrmpi.mpi.mpi_init')
    mpi_init.MPI = SimpleNamespace(Comm=object, Intracomm=object)
    sys.modules['xrmpi.mpi.mpi_init'] = mpi_init
    from xrmpi.mpi.diagnostics import MPIDiagnostics
    diagnostics = MPIDiagnostics()
    diagnostics.comm = SimpleNamespace(rank=0, size=1)
    diagnostics.is_root = lambda root: root == 0
    diagnostics._mpi_lock = Path(lock_path)
else:
    from xrmpi.mpi.mpi_utils import SerialProgressBar

with open(destination, 'w', encoding='utf-8') as stream:
    print('ready', flush=True)
    if mode == 'logging':
        diagnostics.log('locked %s', 'output', prefix=False, file=stream)
    else:
        progress = SerialProgressBar(file=stream, lockfile=lock_path)
        progress.print_progress('locked output', flush=True)
"""
        for mode in ("logging", "progress"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "lock"
                destination = Path(temporary) / "output.txt"
                with path.open("a") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            script,
                            mode,
                            str(path),
                            str(destination),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    try:
                        assert child.stdout is not None
                        with selectors.DefaultSelector() as selector:
                            selector.register(child.stdout, selectors.EVENT_READ)
                            self.assertTrue(
                                selector.select(timeout=10), "Child never started"
                            )
                        self.assertEqual(child.stdout.readline().strip(), "ready")
                        with self.assertRaises(subprocess.TimeoutExpired):
                            child.wait(timeout=0.1)
                        self.assertEqual(destination.read_text(), "")
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                        _, stderr = child.communicate(timeout=10)
                        self.assertEqual(child.returncode, 0, stderr)
                        self.assertEqual(destination.read_text(), "locked output\n")
                    finally:
                        if child.poll() is None:
                            child.kill()
                        child.communicate(timeout=10)

    def test_logger_exception_releases_lock(self) -> None:
        module = load_diagnostics()
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics = module.MPIDiagnostics()
            diagnostics.comm = SimpleNamespace(rank=0, size=1)
            diagnostics.is_root = lambda root: root == 0
            diagnostics._mpi_lock = Path(temporary) / "lock"
            logger = Mock(side_effect=RuntimeError("logger"))
            with self.assertRaisesRegex(RuntimeError, "logger"):
                diagnostics.log("message %s", "argument", logger=logger)
            logger.assert_called_once_with("[MPI RANK 0] message %s", "argument")
            with diagnostics._mpi_lock.open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_abort_hook_records_error_and_releases_lock(self) -> None:
        module = load_diagnostics()
        world = SimpleNamespace(
            rank=0,
            bcast=Mock(side_effect=lambda value, root: value),
            Abort=Mock(),
        )
        module.MPI.COMM_WORLD = world
        destination = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("sys.excepthook", new=sys.__excepthook__),
            patch("sys.stderr", new=destination),
            patch.object(module.time, "sleep"),
        ):
            diagnostics = module.MPIDiagnostics()
            diagnostics.tmp_dir = Path(temporary)
            diagnostics.alive = lambda comm: True
            self.assertTrue(diagnostics._install_abort_hook())
            sys.excepthook(RuntimeError, RuntimeError("stop"), None)
            world.Abort.assert_called_once_with(1)
            error_file = next(Path(temporary).glob("*.error"))
            record = json.loads(error_file.read_text())
            self.assertEqual(record["rank"], 0)
            self.assertEqual(record["type"], "RuntimeError")
            self.assertEqual(record["message"], "stop")
            self.assertTrue(error_file.with_suffix(".error.done").is_file())
            with error_file.with_suffix(".error.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertIn("RuntimeError: stop", destination.getvalue())


class ScratchTests(unittest.TestCase):
    def test_context_allocates_and_cleans_mpi_scratch(self) -> None:
        script = """import sys
from types import ModuleType, SimpleNamespace

class Comm:
    rank = 0
    size = 1
    broadcasts = 0

    def Get_size(self) -> int:
        return 1

    def Get_rank(self) -> int:
        return 0

    def bcast(self, value, root: int):
        self.broadcasts += 1
        return value

    def Barrier(self) -> None:
        pass

    @staticmethod
    def Get_parent() -> None:
        return None

mpi_init = ModuleType('xrmpi.mpi.mpi_init')
mpi_init.MPI = SimpleNamespace(Comm=Comm, Intracomm=Comm, COMM_WORLD=Comm(), COMM_NULL=None)
mpi_init.require_mpi = lambda: None
mpi_init.world_size = lambda: 1
sys.modules['xrmpi.mpi.mpi_init'] = mpi_init
from xrmpi.mpi.context import MPIContext
context = MPIContext()
assert context.comm.broadcasts == 1
assert context.tmp_dir.parent.name == 'xrmpi'
assert context.tmp_dir.parent.parent.name == 'TMP'
assert context._mpi_lock == context.tmp_dir / '.mpi.lock'
context.log('MPI scratch okay', prefix=False)
print(context.tmp_dir)
"""
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ)
            for name in ("SLURM_JOB_TMPDIR", "PBS_JOBTMP", "SCRATCH", "WORK"):
                env.pop(name, None)
            env["TMPDIR"] = temporary
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
            message, directory = result.stdout.splitlines()
            self.assertEqual(message, "MPI scratch okay")
            self.assertTrue(Path(directory).is_relative_to(temporary))
            self.assertFalse(Path(directory).exists())

    def test_netcdf_progress_reuses_existing_context_scratch(self) -> None:
        class Dataset:
            sizes = {"time": 2}
            data_vars = ("value",)

            def isel(self, indexers: dict[str, slice]) -> Dataset:
                return self

            def to_netcdf(self, file: Path, **kwargs: Any) -> None:
                file.write_text("first record")

        class Context:
            def __init__(self, tmpdir: Path) -> None:
                self.tmp_dir = tmpdir
                self.comm = Mock()

        namespace: dict[str, Any] = {
            "Path": Path,
            "xr": SimpleNamespace(Dataset=Dataset, DataArray=type("DataArray", (), {})),
            "MPIContext": Context,
            "SerialProgressBar": SerialProgressBar,
            "encode_dataset_time": lambda data: data,
            "nc_append": Mock(),
        }
        # Execute the real writer path with array/NetCDF I/O stubbed out.
        for filename, names in (
            (
                "core/netcdf.py",
                {"resolve_unlimited_dim", "dataset_to_netcdf", "to_netcdf_serial"},
            ),
            ("core/io.py", {"to_netcdf"}),
        ):
            source = ROOT / filename
            tree = ast.parse(source.read_text())
            functions = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            ]
            code = compile(
                ast.Module(body=functions, type_ignores=[]),
                str(source),
                "exec",
                flags=__future__.annotations.compiler_flag,
            )
            exec(code, namespace)

        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "MPI-scratch"
            scratch.mkdir()
            context = Context(scratch)
            destination = io.StringIO()
            with patch.object(
                mpi_utils, "TemporaryFile", wraps=tempfile.TemporaryFile
            ) as capture:
                namespace["to_netcdf"](
                    Dataset(),
                    Path(temporary) / "output.nc",
                    mpi_context=context,
                    stdout=destination,
                )
            self.assertEqual(capture.call_args.kwargs["dir"], scratch)
            self.assertTrue((scratch / ".mpi.lock").is_file())
            self.assertTrue(scratch.is_dir())
            self.assertEqual(destination.getvalue().count("100%"), 1)
            self.assertEqual(namespace["nc_append"].call_count, 1)
            context.comm.bcast.assert_not_called()
            context.comm.Barrier.assert_not_called()


class InstallerTests(unittest.TestCase):
    def test_setup_installs_only_xrmpi_with_or_without_sibling(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "xrmpi_setup_env", ROOT / "env/setup_env.py"
        )
        assert spec is not None and spec.loader is not None
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)

        for sibling in (False, True):
            with self.subTest(sibling=sibling), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                repo = root / "packages/xrmpi"
                (repo / "env").mkdir(parents=True)
                shutil.copy2(ROOT / "pyproject.toml", repo / "pyproject.toml")
                if sibling:
                    xgeo = repo.parent / "xgeo"
                    xgeo.mkdir()
                    (xgeo / "pyproject.toml").write_text("invalid sibling metadata")
                python = root / "environment/bin/python"
                python.parent.mkdir(parents=True)
                python.touch()
                commands: list[list[str]] = []

                def run(
                    command: list[str],
                    *,
                    env: dict[str, str] | None = None,
                    capture: bool = False,
                ) -> subprocess.CompletedProcess[str]:
                    commands.append(command)
                    if command[1:3] == ["-c", setup.READ_DEPENDENCIES]:
                        return subprocess.run(
                            [sys.executable, *command[1:]],
                            check=True,
                            text=True,
                            capture_output=True,
                        )
                    return subprocess.CompletedProcess(command, 0, stdout="")

                with (
                    patch.object(setup, "__file__", str(repo / "env/setup_env.py")),
                    patch.object(setup.sys, "argv", ["setup_env.py", "xmpi"]),
                    patch.object(
                        setup,
                        "environment_prefix",
                        return_value=(python.parents[1], root / "conda"),
                    ),
                    patch.object(setup, "run", side_effect=run),
                    patch("sys.stdout", new=io.StringIO()),
                ):
                    self.assertEqual(setup.main(), 0)

                editable = [command for command in commands if "--editable" in command]
                self.assertEqual(len(editable), 1)
                self.assertEqual(editable[0][-1], str(repo))
                self.assertIn("--no-deps", editable[0])
                self.assertIn("--no-build-isolation", editable[0])
                build_index = next(
                    index
                    for index, command in enumerate(commands)
                    if command[:2] == ["bash", str(repo / "env/build_libs.sh")]
                )
                self.assertGreater(commands.index(editable[0]), build_index)
                self.assertFalse(
                    any(
                        "xgeo" in argument
                        for command in commands
                        for argument in command
                    )
                )


if __name__ == "__main__":
    unittest.main()
