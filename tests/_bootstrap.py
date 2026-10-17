"""Make the repository importable as ``xrmpi``.

The package root is the repository root (``package-dir = {xrmpi = "."}``), so a
plain checkout is not importable as ``xrmpi`` until it is installed. Importing
this module registers the checkout under that name when needed.
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


ensure_xrmpi()
