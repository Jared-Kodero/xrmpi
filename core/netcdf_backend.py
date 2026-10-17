"""Adapter to xarray's NetCDF4 backend.

This is the only module that touches xarray internals (``dump_to_store``,
``ArrayWriter``, ``cf_encoder``). Supported xarray versions are pinned in
``pyproject.toml`` and exercised by ``tests/test_local.py``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from xarray import conventions
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import ArrayWriter, ensure_dtype_not_object
from xarray.coding.times import encode_cf_datetime, encode_cf_timedelta
from xarray.namedarray.pycompat import is_chunked_array

try:
    from xarray.backends.writers import dump_to_store
except ImportError:  # xarray < 2025.x
    from xarray.backends.api import dump_to_store

# netCDF4 < 1.7.2 assigns to ``ndarray.shape``, which NumPy >= 2.5 deprecates.
warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from ..mpi.mpi_init import MPI


class _SchemaWriter(ArrayWriter):
    """Write eager variables now and leave chunked placeholders to the MPI phase."""

    def __init__(self, unlimited: Iterable[Hashable]) -> None:
        super().__init__()
        self.unlimited = frozenset(unlimited)

    def add(self, source: Any, target: Any, region: Any = None) -> None:
        if not is_chunked_array(source):
            return super().add(source, target, region)
        # Unlimited dimensions only grow when written, so touch the last index.
        variable = target.get_array(needs_lock=False)
        if self.unlimited.intersection(variable.dimensions) and all(source.shape):
            index = tuple(
                n - 1 if dim in self.unlimited else 0
                for dim, n in zip(variable.dimensions, source.shape, strict=True)
            )
            variable[index] = np.zeros((), variable.dtype)


def default_encoding(
    data: xr.Dataset,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    *,
    zlib: bool = True,
    complevel: int = 4,
    shuffle: bool = True,
    chunks: Mapping[Hashable, Iterable[int]] | None = None,
) -> dict[Hashable, dict[str, Any]]:
    """Return per-variable xarray encoding; ``encoding`` overrides the defaults."""
    merged: dict[Hashable, dict[str, Any]] = {}
    if zlib:
        for name, variable in data.data_vars.items():
            if variable.ndim and variable.dtype.kind in "biufcmM":
                merged[name] = {
                    "zlib": True,
                    "complevel": complevel,
                    "shuffle": shuffle,
                }
    for name, shape in (chunks or {}).items():
        if name in data.variables:
            merged.setdefault(name, {})["chunksizes"] = tuple(int(n) for n in shape)
    for name, options in (encoding or {}).items():
        merged.setdefault(name, {}).update(options)
    return merged


def time_kinds(
    variables: Mapping[Hashable, xr.Variable],
) -> dict[Hashable, tuple[str, Any]]:
    """Map variables needing CF time encoding to ``(kind, sample value or None)``.

    ``kind`` is ``"datetime"`` or ``"timedelta"``. The sample is one element, used to
    build placeholders for cftime (object) variables.
    """
    kinds: dict[Hashable, tuple[str, Any]] = {}
    for name, variable in variables.items():
        sample = variable.values.flat[0] if variable.size else None
        if variable.dtype.kind == "m":
            kinds[name] = ("timedelta", sample)
        elif variable.dtype.kind == "M" or (
            variable.dtype.kind == "O" and hasattr(sample, "calendar")
        ):
            kinds[name] = ("datetime", sample)
    return kinds


def agree_time_encoding(
    comm: MPI.Comm,
    kind: str,
    variable: xr.Variable,
    encoding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ``encoding`` with units, calendar and dtype shared by every rank.

    xarray infers these from the values it is given, so ranks that encode only their
    own slab would disagree. Every rank contributes a small sample (its first three
    distinct values and its last), and xarray's own inference runs on the pooled sample.
    """
    values = np.unique(np.asarray(variable.values).ravel())
    pooled = np.unique(
        np.concatenate(comm.allgather(np.concatenate([values[:3], values[-1:]])))
    )
    options = dict(encoding)
    if not pooled.size:
        return options
    if kind == "timedelta":
        encoded, units = encode_cf_timedelta(
            pooled, options.get("units"), options.get("dtype")
        )
        options["units"] = units
    else:
        encoded, units, calendar = encode_cf_datetime(
            pooled, options.get("units"), options.get("calendar"), options.get("dtype")
        )
        options.update(units=units, calendar=calendar)
    options.setdefault("dtype", encoded.dtype)
    return options


def encode_variable(
    name: Hashable, variable: xr.Variable, encoding: Mapping[str, Any]
) -> xr.Variable:
    """CF-encode one variable exactly as the NetCDF4 store would."""
    variable = variable.copy(deep=False)
    variable.encoding = dict(encoding)
    encoded, _ = conventions.cf_encoder({name: variable}, {})
    return ensure_dtype_not_object(encoded[name], name=name)


def create_schema(
    path: str,
    schema: xr.Dataset,
    *,
    unlimited_dims: Iterable[Hashable] = (),
    nofill: bool = False,
) -> None:
    """Create ``path`` with xarray's NetCDF4 store and write the eager variables.

    Chunked variables of ``schema`` are defined, but their data are not written. Each
    variable's ``.encoding`` is used as in :meth:`xarray.Dataset.to_netcdf`.
    """
    unlimited = tuple(unlimited_dims)
    store = NetCDF4DataStore.open(path, mode="w", format="NETCDF4")
    try:
        if nofill:
            store.ds.set_fill_off()
        dump_to_store(schema, store, _SchemaWriter(unlimited), unlimited_dims=unlimited)
    finally:
        store.close()
