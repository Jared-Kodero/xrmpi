"""Make the repository importable as ``xrmpi`` and relax the NetCDF probe.

The package root is the repository root (``package-dir = {xrmpi = "."}``), so a
plain checkout is not importable as ``xrmpi`` until it is installed. Importing
this module registers the checkout under that name when needed.

``MPIXarray`` refuses to be constructed without parallel NetCDF-4, although
nothing but parallel output needs it. The test cases exercise everything else,
so the probe is disabled when the installed netCDF4 lacks parallel support.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ensure_xrmpi() -> None:
    """Import ``xrmpi`` from this checkout, whatever else is installed.

    Tests must exercise the code next to them, not a different installed copy.
    """
    module = sys.modules.get("xrmpi")
    if (
        module is not None
        and Path(getattr(module, "__file__", "")).resolve() == ROOT / "__init__.py"
    ):
        return
    for name in [m for m in sys.modules if m == "xrmpi" or m.startswith("xrmpi.")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "xrmpi", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["xrmpi"] = module
    spec.loader.exec_module(module)


def relax_parallel_probe() -> None:
    """Skip the parallel-NetCDF check when the build lacks it."""
    try:
        import netCDF4
    except ImportError:
        return
    if getattr(netCDF4, "__has_parallel4_support__", 0):
        return
    from xrmpi.core.core import MPIXarray

    MPIXarray._missing_pnetcdf = lambda self: None


ensure_xrmpi()
