"""Append records to an existing NetCDF file along its unlimited dimension.

xarray's ``mode="a"`` adds variables to a file but cannot extend an unlimited
dimension, so this small utility stays separate from the writers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import netCDF4
import xarray as xr

from .netcdf_backend import encode_variable

if TYPE_CHECKING:
    from os import PathLike

# Attributes that decide how xarray encodes values; reused from the stored variable.
_CF_ENCODING_KEYS = (
    "units",
    "calendar",
    "scale_factor",
    "add_offset",
    "_FillValue",
    "missing_value",
)


def append_to_netcdf(
    data: xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    dim: str = "time",
    *,
    zlib: bool | None = None,
    complevel: int | None = None,
    shuffle: bool | None = None,
) -> None:
    """Append ``data`` to ``file`` along its unlimited dimension ``dim``.

    Values are encoded like xarray does, reusing the units, calendar, packing and
    dtype already stored for existing variables. Variables missing from the file are
    created; variables without ``dim`` are only written when created.
    """
    dataset = data.to_dataset() if isinstance(data, xr.DataArray) else data
    if dim not in dataset.sizes:
        raise ValueError(f"Append dimension {dim!r} not present in the data.")
    compression = {
        key: value
        for key, value in (
            ("zlib", zlib),
            ("complevel", complevel),
            ("shuffle", shuffle),
        )
        if value is not None
    }

    with netCDF4.Dataset(file, mode="r+") as nc:
        if dim not in nc.dimensions:
            raise ValueError(f"Append dimension {dim!r} not found in {file}.")
        if not nc.dimensions[dim].isunlimited():
            unlimited = [name for name, d in nc.dimensions.items() if d.isunlimited()]
            raise ValueError(
                f"{dim!r} is not unlimited in {file}; available: {unlimited or 'none'}."
            )
        offset = nc.dimensions[dim].size

        for name, variable in dataset.variables.items():
            stored = nc.variables.get(name)
            encoding = dict(variable.encoding)
            if stored is not None:
                encoding.update(
                    {
                        k: stored.getncattr(k)
                        for k in _CF_ENCODING_KEYS
                        if k in stored.ncattrs()
                    },
                    dtype=stored.dtype,
                )
            encoded = encode_variable(name, variable, encoding)

            if stored is None:
                missing = [d for d in encoded.dims if d not in nc.dimensions]
                if missing:
                    raise ValueError(
                        f"Cannot create {name}: missing dimensions {missing}."
                    )
                attrs = dict(encoded.attrs)
                stored = nc.createVariable(
                    name,
                    encoded.dtype,
                    encoded.dims,
                    fill_value=attrs.pop("_FillValue", None),
                    **(compression if encoded.dims else {}),
                )
                stored.setncatts(attrs)
            elif dim not in variable.dims:
                continue

            stored.set_auto_maskandscale(False)
            values = encoded.transpose(*stored.dimensions).values.astype(stored.dtype)
            index = tuple(
                slice(offset, offset + variable.sizes[dim]) if d == dim else slice(None)
                for d in stored.dimensions
            )
            stored[index if dim in variable.dims else ...] = values
