"""xrmpi: MPI-distributed xarray operations and parallel NetCDF I/O.

The public API provides:

- ``MPIContext``: the communicator, rank information, and MPI diagnostics.
- ``MPIXarray``: an xarray-like wrapper for a rank-local distributed object.
- ``open_dataset``: open a NetCDF file lazily and partition it across ranks.
- ``new_dataarray`` and ``new_dataset``: build distributed objects from fill
  functions and global sizes.
- ``partition``: split a root-owned xarray object across ranks.
- ``to_netcdf`` and ``append_to_netcdf``: serial or collective NetCDF output.

Implementations live in ``xrmpi.core``, ``xrmpi.mpi``, and ``xrmpi.mpp``.
Geospatial analysis, plotting, colormaps, and memory-mapped storage belong
in the separate ``xgeo`` package.
Both packages can be installed independently.

Public objects load on first attribute access. Importing ``xrmpi`` alone
therefore does not import mpi4py or initialize MPI. Resolving an MPI entry
point, including through ``from xrmpi import *``, loads its implementation
and may initialize MPI. The package defaults ``HDF5_USE_FILE_LOCKING`` to
``FALSE`` only when the user has not set it.

Example, executed by every participating rank::

    import xrmpi as xm

    mpi = xm.MPIContext()
    data = xm.open_dataset("data.nc", mpi, partition_dim="time")
    result = data.mean(dim="time")
    data.to_netcdf("output.nc", parallel=True)

Collective operations require all ranks in the communicator to participate.
Parallel NetCDF output requires the matching MPI/HDF5/NetCDF-C stack
installed by ``env/build_libs.sh`` or the author's separate workspace setup.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

if TYPE_CHECKING:
    from .core.core import MPIXarray
    from .core.io import (
        append_to_netcdf,
        new_dataarray,
        new_dataset,
        open_dataset,
        partition,
        to_netcdf,
    )
    from .mpi.context import MPIContext

__all__ = [
    "MPIContext",
    "MPIXarray",
    "append_to_netcdf",
    "new_dataarray",
    "new_dataset",
    "open_dataset",
    "partition",
    "to_netcdf",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "MPIContext": (".mpi.context", "MPIContext"),
    "MPIXarray": (".core.core", "MPIXarray"),
    "append_to_netcdf": (".core.io", "append_to_netcdf"),
    "new_dataarray": (".core.io", "new_dataarray"),
    "new_dataset": (".core.io", "new_dataset"),
    "open_dataset": (".core.io", "open_dataset"),
    "partition": (".core.io", "partition"),
    "to_netcdf": (".core.io", "to_netcdf"),
}


def __getattr__(name: str) -> Any:
    """Load and cache a public object from its implementation module."""
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported objects in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
