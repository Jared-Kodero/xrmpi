"""xrmpi: MPI-distributed xarray operations and parallel NetCDF I/O.

The public API provides:

- ``MPIContext``: the communicator, rank information, and MPI diagnostics.
- ``MPIXarray``: an xarray-like wrapper for a rank-local distributed object.
- ``open_distributed_dataset``, ``create_distributed_dataarray``,
  ``create_distributed_dataset``, and ``distribute_data``: distributed-array
  construction and partitioning.
- ``empty_distributed_dataset`` and ``is_distributed_empty``: helpers for
  ranks with empty partitions.
- ``to_netcdf`` and ``nc_append``: serial or collective NetCDF output.

Implementations live in ``xrmpi.core``, ``xrmpi.mpi``, and ``xrmpi.mpp``.
Geospatial analysis, plotting, colormaps, and memory-mapped storage belong
in the separate ``xgeo`` package.

Public objects load on first attribute access. Importing ``xrmpi`` alone
therefore does not import mpi4py or initialize MPI. Resolving an MPI entry
point, including through ``from xrmpi import *``, loads its implementation
and may initialize MPI. The package defaults ``HDF5_USE_FILE_LOCKING`` to
``FALSE`` only when the user has not set it.

Example, executed by every participating rank::

    import xrmpi as xm

    mpi = xm.MPIContext()
    data = xm.open_distributed_dataset("data.nc", mpi, partition_dim="time")
    result = data.mean(dim="time")
    data.to_netcdf("output.nc", parallel=True)

Collective operations require all ranks in the communicator to participate.
Parallel NetCDF output requires the matching MPI/HDF5/NetCDF-C stack
installed by ``env/setup_env.py`` or the author's separate workspace setup.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

if TYPE_CHECKING:
    from .core.core import MPIXarray
    from .core.io import (
        create_distributed_dataarray,
        create_distributed_dataset,
        distribute_data,
        empty_distributed_dataset,
        is_distributed_empty,
        nc_append,
        open_distributed_dataset,
        to_netcdf,
    )
    from .mpi.context import MPIContext

__all__ = [
    "MPIContext",
    "MPIXarray",
    "create_distributed_dataarray",
    "create_distributed_dataset",
    "distribute_data",
    "empty_distributed_dataset",
    "is_distributed_empty",
    "nc_append",
    "open_distributed_dataset",
    "to_netcdf",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "MPIContext": (".mpi.context", "MPIContext"),
    "MPIXarray": (".core.core", "MPIXarray"),
    "create_distributed_dataarray": (".core.io", "create_distributed_dataarray"),
    "create_distributed_dataset": (".core.io", "create_distributed_dataset"),
    "distribute_data": (".core.io", "distribute_data"),
    "empty_distributed_dataset": (".core.io", "empty_distributed_dataset"),
    "is_distributed_empty": (".core.io", "is_distributed_empty"),
    "nc_append": (".core.io", "nc_append"),
    "open_distributed_dataset": (".core.io", "open_distributed_dataset"),
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
