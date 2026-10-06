"""Calculate chunks for MPI partitioning and NetCDF4/HDF5 output."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import xarray as xr
from dask import array as dask_array

from ..mpp.mpp_domains_define import mpp_compute_extent

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

# Keep chunks below HDF5's 4 GiB hard limit; target half the limit for filter overhead.
MAX_SAVE_CHUNK_BYTES = 2**31


def get_native_chunk_sizes(data: xr.Dataset, dim: Hashable) -> int | None:
    """Return the common native-aligned boundary interval for a dimension.

    Returns
    -------
    int or None
        Smallest interval whose boundaries align with every available native chunk grid,
        or None if native chunking is unavailable.

    """
    sizes: set[int] = set()
    for variable in data.data_vars.values():
        if dim not in variable.dims:
            continue
        chunksizes = variable.encoding.get("chunksizes")
        if chunksizes is not None:
            size = int(chunksizes[variable.get_axis_num(dim)])
        else:
            preferred = variable.encoding.get("preferred_chunks")
            if not isinstance(preferred, dict) or dim not in preferred:
                return None
            size = int(preferred[dim])
        if size <= 0:
            return None
        sizes.add(size)

    return math.lcm(*sizes) if sizes else None


def get_usable_native_chunk(length: int, native_chunk: int | None) -> bool:
    """Return whether a native chunk provides a useful on-disk partition."""
    if length <= 1 or native_chunk is None or native_chunk <= 1:
        return False
    return math.ceil(length / native_chunk) > 1


def get_effective_chunk_size(
    length: int, native_chunk: int | None, mpi_size: int
) -> int:
    """Return the distribution_chunk length climtools should retain for one
    dimension."""
    if length <= 0:
        return 1

    if get_usable_native_chunk(length, native_chunk):
        return cast("int", native_chunk)

    return max(1, math.ceil(length / mpi_size))


def get_chunk_info(data: xr.Dataset, mpi_size: int) -> dict[str, int]:
    """Calculate effective distribution_chunk sizes for all Dataset dimensions."""
    return {
        str(dim): get_effective_chunk_size(
            int(length), get_native_chunk_sizes(data, dim), mpi_size
        )
        for dim, length in data.sizes.items()
    }


def chunk_alignment_holds(length: int, chunk_size: int, size: int) -> bool:
    """Return whether rank bounds for this ``(length, chunk_size, size)`` fall on chunk
    edges."""
    if length <= 0:
        return True
    chunk_count = math.ceil(length / chunk_size)
    return chunk_count >= min(length, size)


def get_chunk_bounds(
    length: int, chunk_size: int, rank: int, size: int
) -> tuple[int, int]:
    """Partition a dimension into per-rank distribution_chunk bounds on chunk
    boundaries."""
    if length <= 0:
        return 0, 0

    if not chunk_alignment_holds(length, chunk_size, size):
        return mpp_compute_extent(length, rank, size)

    chunk_count = math.ceil(length / chunk_size)
    quotient, remainder = divmod(chunk_count, size)
    first_chunk = rank * quotient + min(rank, remainder)
    local_chunks = quotient + int(rank < remainder)
    start = min(first_chunk * chunk_size, length)
    stop = min((first_chunk + local_chunks) * chunk_size, length)
    return start, stop


def prune_chunk_info(
    chunk_info: Mapping[str, int], value: xr.Dataset | xr.DataArray
) -> dict[str, int]:
    """Restrict a distribution_chunk mapping to dimensions actually present on
    ``value``."""
    return {
        str(dim): int(chunk_info[str(dim)])
        for dim in value.dims
        if str(dim) in chunk_info
    }


def _other_dims_bytes(
    itemsize: int,
    dims: Iterable[Hashable],
    shape: Iterable[int],
    partition_dim: Hashable | None,
) -> int:
    """Return bytes contributed by one partition-dimension element."""
    return itemsize * math.prod(
        length for dim, length in zip(dims, shape, strict=True) if dim != partition_dim
    )


def _cap_partition_chunk_to_hdf5_limit(preferred: int, other_bytes: int) -> int:
    """Shrink a partition-dimension save_chunk length to fit the HDF5 4 GiB chunk
    limit."""
    if other_bytes <= 0 or preferred * other_bytes <= MAX_SAVE_CHUNK_BYTES:
        return preferred
    return max(1, MAX_SAVE_CHUNK_BYTES // other_bytes)


def get_partition_chunk_size(
    ds: xr.Dataset, partition_dim: str | None, mpi_size: int
) -> int | None:
    """Return the per-rank-aligned HDF5 save_chunk length for ``partition_dim``."""
    if partition_dim is None or partition_dim not in ds.sizes:
        return ds.sizes.get(partition_dim)

    length = int(ds.sizes[partition_dim])
    preferred = max(1, math.ceil(length / mpi_size))

    other_bytes = max(
        (
            _other_dims_bytes(
                variable.dtype.itemsize, variable.dims, variable.shape, partition_dim
            )
            for variable in ds.data_vars.values()
            if partition_dim in variable.dims
        ),
        default=1,
    )
    return _cap_partition_chunk_to_hdf5_limit(preferred, other_bytes)


def _validate_explicit_chunk_bytes(
    ds: xr.Dataset, explicit: Mapping[str, tuple[int, ...]]
) -> None:
    """Reject explicit chunks that exceed the save-size limit."""
    offenders: list[str] = []
    for name, shape in explicit.items():
        if name not in ds.variables:
            continue
        itemsize = ds.variables[name].dtype.itemsize
        nbytes = itemsize * math.prod(shape)
        if nbytes > MAX_SAVE_CHUNK_BYTES:
            offenders.append(
                f"{name!r}: chunk shape {shape} * itemsize {itemsize} = "
                f"{nbytes / 2**30:.2f} GiB (limit {MAX_SAVE_CHUNK_BYTES / 2**30:.0f} GiB)"
            )
    if offenders:
        raise ValueError(
            "Explicit chunks exceed the HDF5 4 GiB limit: " + "; ".join(offenders)
        )


def get_chunks(
    ds: xr.Dataset,
    chunks: Mapping[str, Iterable[int]] | None,
    partition_dim: str | None = None,
    partition_length: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return explicit or existing save_chunk shapes for every variable."""
    if chunks is not None:
        explicit = {
            name: tuple(int(length) for length in shape)
            for name, shape in chunks.items()
        }
        _validate_explicit_chunk_bytes(ds, explicit)
        return explicit

    output: dict[str, tuple[int, ...]] = {}
    for name, da in ds.variables.items():
        if da.ndim == 0 or any(length == 0 for length in da.shape):
            continue
        chunked = da if da.chunks is not None else da.chunk("auto")
        if chunked.chunks is None:
            shape = tuple(int(length) for length in da.shape)
        else:
            shape = tuple(max(chunked.chunksizes[dim]) for dim in da.dims)
        if (
            partition_dim is not None
            and partition_length is not None
            and partition_dim in da.dims
        ):
            axis = da.dims.index(partition_dim)
            shape = shape[:axis] + (int(partition_length),) + shape[axis + 1 :]
        output[name] = shape
    return output


def _largest_divisor_at_most(value: int, ceiling: int) -> int:
    """Return the largest divisor of ``value`` that is at most ``ceiling``."""
    if value <= 0 or ceiling <= 0:
        return max(1, ceiling)
    if ceiling >= value:
        return value
    best = 1
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            if candidate <= ceiling:
                best = max(best, candidate)
            paired = value // candidate
            if paired <= ceiling:
                best = max(best, paired)
        candidate += 1
    return best


def compute_save_chunks(
    value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any], mpi_size: int
) -> dict[str, tuple[int, ...]]:
    """Derive save chunks for a distributed object using global metadata.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Local slice of a distributed object on the current MPI rank.
    meta : mapping
        Distribution metadata returned by ``mpp_get_meta``. May describe
        one or several active partition dimensions (a Cartesian
        partition); each is handled independently, using that axis's
        own division count from ``meta["cart"]["grid_shape"]`` (falling
        back to ``mpi_size`` when there is no Cartesian grid, i.e. the
        single-dimension case, which reproduces the prior behavior
        exactly).
    mpi_size : int
        Number of MPI ranks the data is distributed across.

    Returns
    -------
    dict
        Mapping from variable name to save chunk tuple, identical across all ranks.

    Raises
    ------
    ValueError
        If ``meta["chunk_info"]`` lacks a partition dimension.

    """
    dims = tuple(str(d) for d in meta["dims"])
    global_sizes = {str(d): int(sz) for d, sz in meta["global_sizes"].items()}
    chunk_info = meta["chunk_info"]
    missing = [d for d in dims if d not in chunk_info]
    if missing:
        raise ValueError(f"chunk_info missing partition dimensions: {missing!r}.")

    cart = meta.get("cart")
    divisor_source: dict[str, int] = {}
    for axis, d in enumerate(dims):
        global_size = global_sizes[d]
        distribution_chunk = int(chunk_info[d])
        divisions = int(cart["grid_shape"][axis]) if cart is not None else mpi_size
        aligned = chunk_alignment_holds(global_size, distribution_chunk, divisions)

        boundary_gcd = global_size
        if not aligned and divisions > 1:
            boundaries = [
                mpp_compute_extent(global_size, i, divisions)[1]
                for i in range(divisions - 1)
            ]
            if boundaries:
                boundary_gcd = math.gcd(*boundaries)

        divisor_source[d] = distribution_chunk if aligned else boundary_gcd

    if isinstance(value, xr.Dataset):
        variables = list(value.variables.items())
    else:
        variables = [
            (str(value.name) if value.name is not None else "__array__", value)
        ]

    output: dict[str, tuple[int, ...]] = {}
    for name, variable in variables:
        if variable.ndim == 0 or any(int(length) == 0 for length in variable.shape):
            continue

        var_dims = tuple(str(d) for d in variable.dims)
        shape = tuple(
            global_sizes[var_dim] if var_dim in dims else int(length)
            for var_dim, length in zip(var_dims, variable.shape, strict=True)
        )
        mock = dask_array.zeros(shape, dtype=variable.dtype, chunks="auto")

        save_chunk: list[int] = []
        for var_dim, length, blocks in zip(var_dims, shape, mock.chunks, strict=True):
            proposed = int(max(blocks)) if blocks else int(length)
            if var_dim not in dims:
                save_chunk.append(proposed)
                continue

            # Bound other axes conservatively; other partition axes may span their full
            # global size.
            other_bytes = variable.dtype.itemsize * math.prod(
                (global_sizes[d] if d in dims and d != var_dim else int(blk_length))
                for d, blk_length in zip(var_dims, shape, strict=True)
                if d != var_dim
            )
            capped = _cap_partition_chunk_to_hdf5_limit(proposed, other_bytes)
            save_chunk.append(_largest_divisor_at_most(divisor_source[var_dim], capped))

        output[str(name)] = tuple(save_chunk)

    return output
