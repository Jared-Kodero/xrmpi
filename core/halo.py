"""Halo exchange and whole-field movement for xarray objects.

:mod:`xgeo.mpp` works on plain arrays and :class:`~xgeo.mpp.
mpp_domains.Domain` objects and knows nothing about xarray. This module is
the adapter: it reads the partition metadata off a Dataset or DataArray,
calls the mpp primitives, and puts the metadata back on the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v
from ..mpp.ext_domains import dim_comm
from ..mpp.mpp_do_update import (
    mpp_complete_update_domains,
    mpp_start_update_domains,
)
from ..mpp.mpp_domains import Domain, DomainMismatchError
from ..mpp.mpp_domains_define import mpp_compute_extent
from ..mpp.mpp_domains_util import mpp_get_neighbor_pe
from .chunks import prune_chunk_info
from .meta import (
    mpp_concat_along,
    mpp_operand_meta,
    mpp_update_meta,
    strip_mpi_meta,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from ..mpi.context import MPIContext


def _haloed_variable_names(
    value: xr.Dataset | xr.DataArray, partition_dim: Hashable
) -> tuple[Hashable, ...]:
    """Return variables that vary along ``partition_dim``."""
    if isinstance(value, xr.Dataset):
        return tuple(
            name for name, var in value.variables.items() if partition_dim in var.dims
        )
    names = [value.name] if partition_dim in value.dims else []
    names.extend(
        name
        for name, coord in value.coords.items()
        if partition_dim in coord.dims and name != value.name
    )
    return tuple(names)


def _fill_chunk(
    template: xr.Dataset | xr.DataArray, dim: str, n: int, fill_value: Any
) -> xr.Dataset | xr.DataArray:
    """Build an ``n``-long, all-``fill_value`` chunk along ``dim``."""
    shaped = template.isel({dim: [0] * n})
    if isinstance(shaped, xr.Dataset):
        filled = shaped.copy(deep=False)
        for name, var in shaped.data_vars.items():
            if dim in var.dims:
                filled[name] = xr.full_like(
                    var,
                    fill_value,
                    # Pass the fill value itself so NumPy preserves xarray's
                    # scalar-promotion rules.
                    dtype=np.result_type(var.dtype, fill_value),
                )
        return filled
    return xr.full_like(
        shaped,
        fill_value,
        # See the matching comment in the Dataset branch above: use
        # fill_value's value-based promotion, not its array dtype.
        dtype=np.result_type(shaped.dtype, fill_value),
    )


def _exchange_halo_blocks(
    value: xr.Dataset | xr.DataArray,
    partition_dim: Hashable,
    before: int,
    after: int,
    *,
    domain: Domain,
    left_rank: int | None,
    right_rank: int | None,
) -> tuple[xr.Dataset | xr.DataArray | None, xr.Dataset | xr.DataArray | None]:
    """Exchange boundary slabs with adjacent ranks."""
    haloed = _haloed_variable_names(value, partition_dim)

    def _local_array(name: Hashable) -> xr.Variable:
        """Return the rank-local variable behind a data or coordinate name."""
        if isinstance(value, xr.Dataset):
            return value[name].variable
        if name == value.name:
            return value.variable
        return value.coords[name].variable

    # Move each partition axis to axis 0 so mixed variable layouts share one halo
    # kernel.
    axes = {name: _local_array(name).dims.index(partition_dim) for name in haloed}
    fields = {
        name: np.moveaxis(np.asarray(_local_array(name).values), axes[name], 0)
        for name in haloed
    }

    # Use start/complete to consume only halo slabs and avoid full-array copies.
    update = mpp_start_update_domains(
        fields,
        domain,
        str(partition_dim),
        0,
        before=before,
        after=after,
        left_rank=left_rank,
        right_rank=right_rank,
    )
    recv_before, recv_after, left_pad, right_pad = mpp_complete_update_domains(update)

    def _received(name: Hashable, side: str) -> np.ndarray[Any, Any] | None:
        """This name's exchanged slab, moved back to its original axis, or None."""
        pad = left_pad if side == "before" else right_pad
        if pad == 0:
            return None
        slab = (recv_before if side == "before" else recv_after)[name]
        return np.moveaxis(slab, 0, axes[name])

    def _reconstruct(side: str) -> xr.Dataset | xr.DataArray | None:
        """Reconstruct an xarray object from the exchanged arrays, or None if
        unpadded."""
        if (left_pad if side == "before" else right_pad) == 0:
            return None
        if isinstance(value, xr.Dataset):
            pieces = {}
            for name, var in value.variables.items():
                received = _received(name, side) if name in haloed else None
                pieces[name] = (
                    var
                    if received is None
                    else xr.Variable(var.dims, received, attrs=var.attrs)
                )
            return xr.Dataset(pieces, attrs=value.attrs)
        data_var = xr.Variable(
            value.dims, _received(value.name, side), attrs=value.attrs
        )
        new_coords = {}
        for coord_name, coord in value.coords.items():
            received = _received(coord_name, side) if coord_name in haloed else None
            new_coords[coord_name] = (
                coord.variable
                if received is None
                else xr.Variable(coord.dims, received, attrs=coord.attrs)
            )
        return xr.DataArray(data_var, coords=new_coords, name=value.name)

    return _reconstruct("before"), _reconstruct("after")


def mpp_halo_exchange(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable | None = None,
    *,
    before: int,
    after: int,
    periodic: bool = False,
    exchange_coords: bool = True,
) -> tuple[xr.Dataset | xr.DataArray, int, int]:
    """Pad ``value`` with boundary slices from the adjacent ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed object to pad.
    dim : Hashable, optional
        The partition axis to exchange along.
    before, after : int
        Number of elements requested from the neighbor below/above along ``dim``.
    periodic : bool, optional
        Wrap the neighbor lookup at the global boundary instead of
        leaving that side unpadded (rank 0's lower neighbor becomes the
        last rank, and symmetrically on the upper side).
    exchange_coords : bool, optional
        Whether coordinates varying along ``dim`` take part. An operation
        that reads coordinate *values* across the rank boundary needs them
        -- ``differentiate`` divides by a spacing that straddles it -- but
        one that reads only data values and then trims back to its own
        compute domain does not, which is most of them. Exchanging them
        anyway costs far more than their size suggests: joining an index
        coordinate makes xarray rebuild a pandas Index over the padded
        extent, measured here at roughly four times the cost of joining the
        data alone. Pass False when the caller restores the coordinate
        itself; the padded object then carries none along ``dim``.

    Returns
    -------
    tuple[xarray.Dataset or xarray.DataArray, int, int]
        ``(padded, left_pad, right_pad)``: the padded object (replicated metadata
        stripped, since it is no longer a clean partition) and the number of elements
        actually prepended/appended (equal to ``before``/``after`` except at a global
        edge, where it is 0).

    Raises
    ------
    ValueError
        If ``value`` is not distributed, ``dim`` is missing or disagrees with an active
        partition dimension, ``before``/``after`` are negative, or any rank's local
        partition along ``dim`` is shorter than ``before``/``after``.

    """
    meta = mpp_operand_meta(value)
    if meta is None:
        raise ValueError("requires a distributed xarray object")
    partition_dims = meta["dims"]
    if dim is None:
        if len(partition_dims) > 1:
            raise ValueError(
                "dim is required for partition dimensions "
                + f"{tuple(str(d) for d in partition_dims)!r}."
            )
        partition_dim = partition_dims[0]
    elif dim not in partition_dims:
        raise ValueError(
            f"dim={dim!r} is not active; choose from "
            + f"{tuple(str(d) for d in partition_dims)!r}."
        )
    else:
        partition_dim = dim
    if before < 0 or after < 0:
        raise ValueError("before and after must be >= 0")

    from .planning import _agree

    _agree(
        mpi_context,
        (
            "mpp_halo_exchange",
            str(partition_dim),
            int(before),
            int(after),
            bool(periodic),
            bool(exchange_coords),
        ),
    )

    if before == 0 and after == 0:
        # A zero-width halo is purely local; skip all communication.
        return value, 0, 0

    if not exchange_coords:
        along_dim = [
            name for name, coord in value.coords.items() if partition_dim in coord.dims
        ]
        if along_dim:
            value = value.drop_vars(along_dim)

    comm = mpi_context.comm
    # Resolve halo neighbors through the Cartesian-aware domain helper.
    domain = Domain.from_meta(meta, comm)
    left_rank, right_rank = mpp_get_neighbor_pe(
        domain, str(partition_dim), periodic=periodic
    )

    # The halo width is validated collectively inside mpp_start_update_domains.
    before_block, after_block = _exchange_halo_blocks(
        value,
        partition_dim,
        before,
        after,
        domain=domain,
        left_rank=left_rank,
        right_rank=right_rank,
    )

    pieces = [
        piece for piece in (before_block, value, after_block) if piece is not None
    ]
    padded = (
        value if len(pieces) <= 1 else mpp_concat_along(pieces, value, partition_dim)
    )
    return (
        strip_mpi_meta(padded),
        before if before_block is not None else 0,
        after if after_block is not None else 0,
    )


def mpp_global_field_xr(
    mpi_context: MPIContext,
    coordinate: xr.DataArray,
    dim: str,
    comm: MPI.Comm,
    *,
    start: int,
    stop: int,
    global_size: int,
) -> xr.DataArray | None:
    """Gather a coordinate distributed along ``dim`` into its global form.

    Follows FMS ``mpp_global_field``: each rank contributes its compute-domain
    slice and the root reassembles the whole axis, verifying that the slices
    tile it exactly with no gap or overlap.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    coordinate : xarray.DataArray
        This rank's slice of the coordinate.
    dim : str
        Partitioned dimension.
    comm : mpi4py.MPI.Comm
        Communicator varying along ``dim``.
    start, stop : int
        This rank's half-open bounds along ``dim``.
    global_size : int
        Global length of ``dim``.

    Returns
    -------
    xarray.DataArray or None
        The reassembled coordinate on rank 0 of ``comm``, None elsewhere.

    Raises
    ------
    DomainMismatchError
        If the gathered slices do not tile the axis exactly.
    """
    axis = coordinate.get_axis_num(dim)
    pieces = gather_v((start, stop, np.asarray(coordinate.values)), comm, root=0)
    if comm.rank != 0 or pieces is None:
        return None

    cursor = 0
    ordered = sorted(pieces, key=lambda item: item[0])
    for piece_start, piece_stop, values in ordered:
        if piece_start != cursor:
            raise DomainMismatchError(
                f"Coordinate {coordinate.name!r}: expected start {cursor}, "
                + f"got {piece_start}."
            )
        if values.shape[axis] != piece_stop - piece_start:
            raise DomainMismatchError(
                f"Coordinate {coordinate.name!r} slice length "
                + f"{values.shape[axis]} != {piece_stop - piece_start}."
            )
        cursor = piece_stop
    if cursor != global_size:
        raise DomainMismatchError(
            f"Coordinate {coordinate.name!r} covers {cursor}/{global_size} elements."
        )

    rebuilt = xr.DataArray(
        np.concatenate([values for _, _, values in ordered], axis=axis),
        dims=coordinate.dims,
        name=coordinate.name,
        attrs=dict(coordinate.attrs),
    )
    rebuilt.encoding = dict(coordinate.encoding)
    return rebuilt


def mpp_redistribute(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    dim: str,
    *,
    new_coord: np.ndarray[Any, Any],
    old_pos: np.ndarray[Any, Any],
    fill_value: Any,
) -> xr.Dataset | xr.DataArray:
    """Move ``value`` onto a new decomposition of ``dim``.

    Follows FMS ``mpp_redistribute``: each rank derives, from the shared
    position map alone, which of its elements every other rank needs and which
    it must receive, so no layout metadata travels with the payload.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.Dataset or xarray.DataArray
        Object to redistribute.
    meta : mapping
        Current distribution metadata.
    dim : str
        Dimension being redistributed.
    new_coord : numpy.ndarray
        Coordinate values of the target decomposition.
    old_pos : numpy.ndarray
        For each new global position, the old global position feeding it, or
        -1 where the target has no source and takes ``fill_value``.
    fill_value : Any
        Value for target positions with no source.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        This rank's slice of the redistributed object.
    """
    comm = dim_comm(meta, dim, mpi_context)
    rank, size = comm.rank, comm.size

    old_start = int(meta["starts"][dim])
    old_stop = int(meta["stops"][dim])
    old_starts, _old_stops = zip(*gather_v((old_start, old_stop), comm), strict=True)
    old_starts_arr = np.asarray(old_starts, dtype=np.int64)

    new_length = int(new_coord.shape[0])
    new_starts_all = np.fromiter(
        (mpp_compute_extent(new_length, r, size)[0] for r in range(size)),
        dtype=np.int64,
        count=size,
    )
    new_start, new_stop = mpp_compute_extent(new_length, rank, size)

    def _owner_of(
        global_positions: np.ndarray[Any, Any], starts: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Return the rank owning a global position."""
        return np.searchsorted(starts, global_positions, side="right") - 1

    owned_mask = (old_pos >= old_start) & (old_pos < old_stop)
    p_owned = np.nonzero(owned_mask)[0]  # ascending new positions I feed
    g_owned = old_pos[p_owned]  # corresponding old global positions (mine)
    dest_of_p_owned = _owner_of(p_owned, new_starts_all)

    self_payload: xr.Dataset | xr.DataArray | None = None
    send_requests: list[MPI.Request] = []

    # Source and destination ranks derive the same position map, so payload metadata is
    # unnecessary.
    my_local_p = np.arange(new_start, new_stop, dtype=np.int64)
    my_local_g = old_pos[new_start:new_stop] if new_length > 0 else my_local_p
    my_is_fill = my_local_g == -1
    my_owner = np.full(my_local_p.shape, -1, dtype=np.int64)
    if (~my_is_fill).any():
        my_owner[~my_is_fill] = _owner_of(my_local_g[~my_is_fill], old_starts_arr)

    incoming_sources = sorted(
        {int(s) for s in np.unique(my_owner) if s >= 0 and s != rank}
    )

    for dest in range(size):
        mask = dest_of_p_owned == dest
        if not mask.any():
            continue
        local_old_idx = g_owned[mask] - old_start
        payload = value.isel({dim: local_old_idx})
        if dest == rank:
            self_payload = payload
        else:
            send_requests.append(comm.isend(payload, dest=dest))

    # Use ``recv`` for pickled payloads because it probes size; ``irecv`` requires a
    # buffer-size guess.
    received = {source: comm.recv(source=source) for source in incoming_sources}
    MPI.Request.Waitall(send_requests)

    if new_stop <= new_start:
        empty = value.isel({dim: slice(0, 0)})
        result = empty.assign_coords({dim: new_coord[new_start:new_stop]})
    else:
        pieces: list[xr.Dataset | xr.DataArray] = []
        slot_pieces: list[np.ndarray[Any, Any]] = []

        self_mask = my_owner == rank
        if self_mask.any():
            if self_payload is None:
                raise AssertionError("Missing planned self-contribution.")
            pieces.append(self_payload)
            slot_pieces.append(np.nonzero(self_mask)[0])

        if my_is_fill.any():
            n_fill = int(my_is_fill.sum())
            pieces.append(_fill_chunk(value, dim, n_fill, fill_value))
            slot_pieces.append(np.nonzero(my_is_fill)[0])

        for source in incoming_sources:
            mask = my_owner == source
            pieces.append(received[source])
            slot_pieces.append(np.nonzero(mask)[0])

        combined = mpp_concat_along(pieces, value, dim)
        slots = np.concatenate(slot_pieces)
        final_order = np.argsort(slots, kind="stable")
        result = combined.isel({dim: final_order})
        result = result.assign_coords({dim: new_coord[new_start:new_stop]})

    result = strip_mpi_meta(result)
    chunk_info = prune_chunk_info(meta["chunk_info"], result)
    remaining_dims = tuple(d for d in meta["dims"] if d != dim)
    if remaining_dims:
        all_dims = meta["dims"]
        global_size = {d: int(meta["global_sizes"][d]) for d in remaining_dims}
        start = {d: int(meta["starts"][d]) for d in remaining_dims}
        stop = {d: int(meta["stops"][d]) for d in remaining_dims}
        global_size[dim] = new_length
        start[dim] = new_start
        stop[dim] = new_stop
        mpp_update_meta(
            result,
            dim=all_dims,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info=chunk_info,
            cart=meta.get("cart"),
        )
    else:
        mpp_update_meta(
            result,
            dim=dim,
            global_size=new_length,
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
    return result
