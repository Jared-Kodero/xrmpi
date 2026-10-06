"""Provide the MPI-distributed xarray wrapper."""

from __future__ import annotations

import operator as _operator
from functools import wraps
from typing import TYPE_CHECKING, Any

import xarray as xr

from ..mpi.mpi_init import MPI
from .distribute import mpp_repartition
from .elementwise import (
    mpp_align,
    mpp_apply,
    mpp_bfill,
    mpp_cumprod,
    mpp_cumsum,
    mpp_diff,
    mpp_differentiate,
    mpp_evaluate,
    mpp_ffill,
    mpp_interp,
    mpp_matmul,
    mpp_median,
    mpp_pad,
    mpp_quantile,
    mpp_roll,
    mpp_shift,
    mpp_where,
)
from .handles import MPIGroupBy, MPIResample, MPIRolling, MPIXarrayWeighted
from .indexing import mpp_isel, mpp_reindex, mpp_sel, mpp_sortby
from .io import mpp_attach_save_chunks
from .meta import PARTITIONED_ATTR as _PARTITIONED_ATTR
from .meta import mpp_get_meta, mpp_set_meta, strip_mpi_meta
from .reductions import (
    mpp_all_reduce,
    mpp_any_reduce,
    mpp_coarsen_reduce,
    mpp_first_reduce,
    mpp_last_reduce,
    mpp_max_reduce,
    mpp_mean_reduce,
    mpp_min_reduce,
    mpp_prod_reduce,
    mpp_rolling_reduce,
    mpp_std,
    mpp_sum_reduce,
    mpp_var,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping
    from os import PathLike
    from types import EllipsisType
    from typing import Literal

    import numpy as np

    from ..mpi.context import MPIContext


#: Lightweight partition marker; full distribution state remains in ``mpi_meta``.

#: Sentinel distinguishing "no fill value given" from a genuine ``other=None``
#: in :meth:`MPIXarray.where`.
_WHERE_UNSET = object()

#: Sentinel distinguishing "fill_value not given" (defer to the engine's own
#: ``NaN`` default) from a genuine ``fill_value=None`` in :meth:`MPIXarray.reindex`.
_FILL_VALUE_UNSET = object()

#: Read-only xarray properties safe to expose without changing partition ownership.
_SAFE_PASSTHROUGH_ATTRS: frozenset[str] = frozenset(
    {
        "attrs",
        "chunks",
        "chunksizes",
        "coords",
        "data_vars",
        "dims",
        "dtype",
        "dtypes",
        "encoding",
        "indexes",
        "name",
        "nbytes",
        "ndim",
        "shape",
        "size",
        "sizes",
        "T",
        "values",
        "variables",
    }
)

#: Allow only methods proven rank-local and partition-preserving.
#: Communication-dependent methods require dedicated implementations.
_SAFE_PASSTHROUGH_METHODS: frozenset[str] = frozenset(
    {
        "astype",
        "assign_attrs",
        "assign_coords",
        "chunk",
        "clip",
        "compute",
        "drop_vars",
        "expand_dims",
        "fillna",
        "isnull",
        "load",
        "notnull",
        "persist",
        "rename",
        "rename_dims",
        "rename_vars",
        "reset_coords",
        "round",
        "set_coords",
        "transpose",
    }
)


# store this we will monkey patch to all ds/da binary op on MPIXarray instance
dataarray_binary_op = xr.DataArray._binary_op
dataset_binary_op = xr.Dataset._binary_op


class MPIXarray:
    """MPI aware xarray Dataset/DataArray object

    Parameters
    ----------
    data : MPIXarray or xarray.Dataset or xarray.DataArray
        Rank-local data. Existing ``MPIXarray`` instances are adopted unchanged.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime or communicator used by distributed operations.
    meta : dict, optional
        Explicit distribution metadata. If omitted, metadata is read from ``data``.
    auto_partition : bool, default True
        Partition replicated input that has no distribution metadata.
    dim : Hashable or {"auto"}, default "auto"
        Dimension used when auto-partitioning.
    chunk_info : mapping of str to int, optional
        Chunk-size hints used when auto-partitioning.
    log_partitions : bool, default False
        Log the partition layout when auto-partitioning.

    Notes
    -----
    Distribution metadata is stored on ``.meta`` and attached to xarray attributes
    only while an engine operation is executing. Replicated input is assumed to be
    present on every rank; use :func:`~.constructors.distribute_data` for
    root-owned input.

    """

    #: Route elementwise ufunc calls through ``mpp_apply``; reject reductions and
    #: in-place ``out=`` writes.
    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Dispatch NumPy ufuncs through the distributed wrapper."""
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        return self.apply(ufunc, *inputs, **kwargs)

    def __init__(
        self,
        data: MPIXarray | xr.Dataset | xr.DataArray,
        mpi_context: MPIContext | MPI.Intracomm,
        meta: dict[str, Any] | None = None,
        *,
        auto_partition: bool = True,
        dim: Hashable | Literal["auto"] = "auto",
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> None:
        """Initialize a distributed xarray wrapper."""

        from ..mpi.context import MPIContext

        if not isinstance(mpi_context, MPIContext):
            mpi_context = MPIContext(mpi_context)

        self._missing_pnetcdf()

        if isinstance(data, MPIXarray):
            self.data = data.data
            self.meta = data.meta
            self._runtime = data._runtime
            return

        if meta is None:
            meta = mpp_get_meta(data)
            if meta is not None:
                data = strip_mpi_meta(data)
        self.data = data
        self.meta = meta
        self._runtime = mpi_context

        if self.meta is None and auto_partition:
            partitioned = mpp_repartition(
                mpi_context,
                self.data,
                dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            )
            new_meta = mpp_get_meta(partitioned)
            if new_meta is not None:
                partitioned = strip_mpi_meta(partitioned)
            self.data = partitioned
            self.meta = new_meta

        self.data = mark_partitioned(self.data, self.meta)

    def __repr__(self) -> str:
        """Return the distributed wrapper representation."""
        return repr(self.data)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped xarray object."""
        if name == "data":
            raise AttributeError(name)
        if name in _SAFE_PASSTHROUGH_ATTRS:
            return getattr(self.data, name)
        if name in _SAFE_PASSTHROUGH_METHODS:
            return self._safe_method_wrapper(name)
        raise AttributeError(f"MPIXarray has no distribution-safe attribute {name!r}.")

    def _safe_method_wrapper(self, name: str) -> Callable[..., Any]:
        """Return a bound, ``mpp_apply()``-based forward of ``self.data.<name>``."""

        def call(*args: Any, **kwargs: Any) -> Any:
            """Call the wrapped xarray method through MPI-aware apply."""
            return self.apply(
                lambda d, *a, **kw: getattr(d, name)(*a, **kw), self, *args, **kwargs
            )

        call.__name__ = name
        call.__doc__ = (
            f"MPI-aware forward of xarray's `.{name}(...)`, via `mpp_apply()`. "
            f"See MPIXarray's `_SAFE_PASSTHROUGH_METHODS`."
        )
        return call

    def _missing_pnetcdf(self) -> None:
        """Raise an error if parallel NetCDF-4 support is missing on root rank."""

        import netCDF4

        if netCDF4.__has_parallel4_support__:
            return
        else:
            raise RuntimeError("netCDF4 lacks parallel HDF5 support.")

    # Binary operators require compatible partition ownership; matmul may contract it.

    def __add__(self, other: Any) -> Any:
        """Elementwise addition (``self + other``); see the note above."""
        return self.apply(_operator.add, self, other)

    def __radd__(self, other: Any) -> Any:
        """Elementwise addition (``other + self``); see the note above."""
        return self.apply(_operator.add, other, self)

    def __sub__(self, other: Any) -> Any:
        """Elementwise subtraction (``self - other``); see the note above."""
        return self.apply(_operator.sub, self, other)

    def __rsub__(self, other: Any) -> Any:
        """Elementwise subtraction (``other - self``); see the note above."""
        return self.apply(_operator.sub, other, self)

    def __mul__(self, other: Any) -> Any:
        """Elementwise multiplication (``self * other``); see the note above."""
        return self.apply(_operator.mul, self, other)

    def __rmul__(self, other: Any) -> Any:
        """Elementwise multiplication (``other * self``); see the note above."""
        return self.apply(_operator.mul, other, self)

    def __truediv__(self, other: Any) -> Any:
        """Elementwise division (``self / other``); see the note above."""
        return self.apply(_operator.truediv, self, other)

    def __rtruediv__(self, other: Any) -> Any:
        """Elementwise division (``other / self``); see the note above."""
        return self.apply(_operator.truediv, other, self)

    def __floordiv__(self, other: Any) -> Any:
        """Elementwise floor division (``self // other``); see the note above."""
        return self.apply(_operator.floordiv, self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        """Elementwise floor division (``other // self``); see the note above."""
        return self.apply(_operator.floordiv, other, self)

    def __mod__(self, other: Any) -> Any:
        """Elementwise modulo (``self % other``); see the note above."""
        return self.apply(_operator.mod, self, other)

    def __rmod__(self, other: Any) -> Any:
        """Elementwise modulo (``other % self``); see the note above."""
        return self.apply(_operator.mod, other, self)

    def __pow__(self, other: Any) -> Any:
        """Elementwise exponentiation (``self ** other``); see the note above."""
        return self.apply(_operator.pow, self, other)

    def __rpow__(self, other: Any) -> Any:
        """Elementwise exponentiation (``other ** self``); see the note above."""
        return self.apply(_operator.pow, other, self)

    def __neg__(self) -> Any:
        """Elementwise negation (``-self``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.neg, self)

    def __pos__(self) -> Any:
        """Elementwise unary plus (``+self``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.pos, self)

    def __abs__(self) -> Any:
        """Elementwise absolute value (``abs(self)``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.abs, self)

    def __and__(self, other: Any) -> Any:
        """Elementwise logical/bitwise AND (``self & other``); see the note above."""
        return self.apply(_operator.and_, self, other)

    def __rand__(self, other: Any) -> Any:
        """Elementwise logical/bitwise AND (``other & self``); see the note above."""
        return self.apply(_operator.and_, other, self)

    def __or__(self, other: Any) -> Any:
        """Elementwise logical/bitwise OR (``self | other``); see the note above."""
        return self.apply(_operator.or_, self, other)

    def __ror__(self, other: Any) -> Any:
        """Elementwise logical/bitwise OR (``other | self``); see the note above."""
        return self.apply(_operator.or_, other, self)

    def __xor__(self, other: Any) -> Any:
        """Elementwise logical/bitwise XOR (``self ^ other``); see the note above."""
        return self.apply(_operator.xor, self, other)

    def __rxor__(self, other: Any) -> Any:
        """Elementwise logical/bitwise XOR (``other ^ self``); see the note above."""
        return self.apply(_operator.xor, other, self)

    def __invert__(self) -> Any:
        """Elementwise logical/bitwise NOT (``~self``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.invert, self)

    def __lt__(self, other: Any) -> Any:
        """Elementwise ``self < other``; see the note above."""
        return self.apply(_operator.lt, self, other)

    def __le__(self, other: Any) -> Any:
        """Elementwise ``self <= other``; see the note above."""
        return self.apply(_operator.le, self, other)

    def __gt__(self, other: Any) -> Any:
        """Elementwise ``self > other``; see the note above."""
        return self.apply(_operator.gt, self, other)

    def __ge__(self, other: Any) -> Any:
        """Elementwise ``self >= other``; see the note above."""
        return self.apply(_operator.ge, self, other)

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        """Elementwise ``self == other`` -- returns an array, not a bool."""
        return self.apply(_operator.eq, self, other)

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        """Elementwise ``self != other`` -- returns an array, not a bool."""
        return self.apply(_operator.ne, self, other)

    # Elementwise equality intentionally leaves ``MPIXarray`` unhashable, matching
    # xarray.
    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        """Truth value (``if mpixarray:``, ``bool(mpixarray)``)."""
        if self.meta is not None:
            raise ValueError("Distributed MPIXarray has no single truth value.")
        return bool(self.data)

    def __getitem__(self, key: Any) -> Any:
        """Select a Dataset variable by name; mirrors ``xarray.Dataset[key]``."""
        if isinstance(key, str):
            return self.apply(lambda d, k: d[k], self, key)
        raise TypeError(f"Dataset variable key must be str, got {type(key).__name__}.")

    def __matmul__(self, other: Any) -> MPIXarray:
        """Matrix multiplication (``self @ other``); redirects to :meth:`matmul`."""
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> Any:
        """Matrix multiplication (``other @ self``); MPI-aware like :meth:`matmul`."""
        result = mpp_matmul(self._runtime, unwrap(other), self._prepare())
        return finalize(result, self._runtime)

    def _prepare(self) -> xr.Dataset | xr.DataArray:
        """Return ``self.data`` with ``self.meta`` reattached to ``.attrs``."""
        if self.meta is None:
            return self.data
        prepared = self.data.copy(deep=False)
        mpp_set_meta(prepared, self.meta)
        return prepared

    def load(self) -> MPIXarray:
        """Materialize this rank's local data in place.

        Lazy dask-backed data would otherwise be recomputed from scratch
        on every downstream call; use this once, right after
        construction, when the result will be reused more than once.

        Returns
        -------
        MPIXarray
            ``self``, mutated in place so this chains onto a constructor
            call, e.g. ``xgeo.create_distributed_dataarray(...).load()``.

        """
        self.data = self.data.load()
        return self

    def to_netcdf(
        self,
        file: str | PathLike[str],
        unlimited_dim: str | Iterable[str] | None = None,
        partition_dim: str | None = None,
        *,
        parallel: bool = False,
        batch_size: int = 24,
        format: str = "NETCDF4",
        shuffle: bool = True,
        zlib: bool = True,
        complevel: int = 4,
        show_progress: bool = True,
        stdout: Any = None,
        chunks: Mapping[str, Iterable[int]] | None = None,
        hints: str | None = None,
        nofill: bool = True,
        allow_serial: bool = False,
    ) -> None:
        """Write this xarray object to NetCDF.

        Parameters
        ----------
        file : str or os.PathLike
            Output path.
        unlimited_dim : str or iterable of str, optional
            Unlimited dimension names.
        partition_dim : str, optional
            Dimension used for MPI partitioning.
        parallel : bool, default False
            Use MPI-parallel NetCDF output.
        batch_size : int, default 24
            Number of slices per serial append.
        format : str, default "NETCDF4"
            NetCDF format used for serial output.
        shuffle : bool, default True
            Enable the HDF5 shuffle filter.
        zlib : bool, default True
            Enable zlib compression.
        complevel : int, default 4
            zlib compression level.
        show_progress : bool, default True
            Show serial write progress.
        stdout : Any, optional
            Stream used for progress output.
        chunks : mapping, optional
            Explicit NetCDF variable chunk shapes.
        hints : str, optional
            Semicolon-separated MPI-IO hints.
        nofill : bool, default True
            Disable NetCDF pre-filling for parallel output.
        allow_serial : bool, default False
            Allow the parallel writer to run with one MPI rank.

        Raises
        ------
        ValueError
            If distributed data are passed to the serial writer.

        """
        from .io import to_netcdf

        if not parallel and self.meta is not None:
            raise ValueError(
                "Distributed data requires parallel=True for NetCDF output."
            )

        prepared = self._prepare()
        if parallel:
            prepared = mpp_attach_save_chunks(self._runtime, prepared)
        if self.meta is not None and mpp_get_meta(prepared) is None:
            prepared = prepared.copy(deep=False)
            mpp_set_meta(prepared, self.meta)

        to_netcdf(
            prepared,
            file,
            self._runtime,
            unlimited_dim,
            partition_dim,
            parallel=parallel,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
            chunks=chunks,
            hints=hints,
            nofill=nofill,
            allow_serial=allow_serial,
        )

    def repartition(
        self,
        dim: Hashable | Literal["auto"] = "auto",
        *,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> MPIXarray:
        """Partition a replicated object across MPI ranks.

        Parameters
        ----------
        dim : Hashable or {"auto"}, optional
            New partition dimension.
        chunk_info : mapping of str to int, optional
            Effective chunk-size hints.
        log_partitions : bool, optional
            Log the resulting rank layout.

        Returns
        -------
        MPIXarray
            Rank-local slice with ``.meta`` set.

        """
        return finalize(
            mpp_repartition(
                self._runtime,
                self._prepare(),
                dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            self._runtime,
        )

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        *,
        partition_dim: Hashable | Literal["auto"] | None = None,
        **indexers_kwargs: Any,
    ) -> MPIXarray:
        """Index with global integer coordinates on the partition dimension.

        Parameters
        ----------
        indexers : mapping, optional
            Integer indexers using global coordinates on the partition dimension.
        partition_dim : Hashable or {"auto"} or None, optional
            Dimension to scatter a resulting single-element partition dimension across, when a slice indexer collapses it to length one.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        MPIXarray
            Indexed object with ``.meta`` updated.

        """
        return finalize(
            mpp_isel(
                self._runtime,
                self._prepare(),
                indexers,
                partition_dim=partition_dim,
                **indexers_kwargs,
            ),
            self._runtime,
        )

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: Any = None,
        drop: bool = False,
        *,
        partition_dim: Hashable | Literal["auto"] | None = None,
        **indexers_kwargs: Any,
    ) -> MPIXarray:
        """Index with global coordinate labels on the partition dimension.

        Parameters
        ----------
        indexers : mapping, optional
            Label indexers using global semantics on the partition dimension.
        method : str, optional
            Inexact matching method passed to xarray.
        tolerance : Any, optional
            Maximum distance for inexact matches.
        drop : bool, optional
            Drop selected coordinate variables.
        partition_dim : Hashable or {"auto"} or None, optional
            Dimension to scatter a resulting single-element partition dimension across, when a label slice collapses it to length one.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        MPIXarray
            Indexed object with ``.meta`` updated.

        """
        return finalize(
            mpp_sel(
                self._runtime,
                self._prepare(),
                indexers,
                method,
                tolerance,
                drop,
                partition_dim=partition_dim,
                **indexers_kwargs,
            ),
            self._runtime,
        )

    def sum(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Sum over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.

        """
        return finalize(
            mpp_sum_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                min_count=min_count,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def prod(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Multiply over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.

        """
        return finalize(
            mpp_prod_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                min_count=min_count,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def mean(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the mean over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.

        """
        return finalize(
            mpp_mean_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def min(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the minimum over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.

        """
        return finalize(
            mpp_min_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def max(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the maximum over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.

        """
        return finalize(
            mpp_max_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def any(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Return whether any value is true over the requested dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Logical OR over the requested dimensions, with ``.meta`` updated.

        """
        return finalize(
            mpp_any_reduce(
                self._runtime,
                self._prepare(),
                dim,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def all(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Return whether every value is true over the requested dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Logical AND over the requested dimensions, with ``.meta`` updated.

        """
        return finalize(
            mpp_all_reduce(
                self._runtime,
                self._prepare(),
                dim,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def first(
        self,
        dim: str,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Select the first valid value along one dimension.

        Parameters
        ----------
        dim : str
            Dimension to pick a position along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Selected object with ``.meta`` updated.

        """
        return finalize(
            mpp_first_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def last(
        self,
        dim: str,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Select the last valid value along one dimension.

        Parameters
        ----------
        dim : str
            Dimension to pick a position along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Selected object with ``.meta`` updated.

        """
        return finalize(
            mpp_last_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def var(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the variance over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        ddof : int, optional
            Delta degrees of freedom; the divisor is ``N - ddof``.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.

        """
        return finalize(
            mpp_var(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                ddof=ddof,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def std(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the standard deviation over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        ddof : int, optional
            Delta degrees of freedom; the divisor is ``N - ddof``.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.

        """
        return finalize(
            mpp_std(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                ddof=ddof,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    # -- Groupby (xarray-styled entry points; mpp_groupby_reduce/mpp_resample_reduce
    #    are internal engine dispatch names, not part of this public surface) -

    def groupby(
        self, dim: Hashable, labels: xr.DataArray | np.ndarray[Any, Any]
    ) -> MPIGroupBy:
        """Group by ``labels`` along ``dim``, mirroring ``xarray.Dataset.groupby``.

        Parameters
        ----------
        dim : Hashable
            Dimension being grouped and reduced.
        labels : array-like
            Group key for every position along this rank's local ``dim`` axis.

        Returns
        -------
        MPIGroupBy
            Chainable handle; call ``.sum()``, ``.mean()``, ``.count()``,
            ``.min()``, or ``.max()`` on it to get the reduced :class:`MPIXarray`.

        """
        return MPIGroupBy(self, dim, labels)

    def resample(self, dim: Hashable, freq: str) -> MPIResample:
        """Resample a datetime dimension using xarray semantics.

        Parameters
        ----------
        dim : Hashable
            Datetime dimension to resample.
        freq : str
            Pandas offset alias (e.g.

        Returns
        -------
        MPIResample
            Chainable handle; call ``.sum()``, ``.mean()``, ``.count()``,
            ``.min()``, or ``.max()`` on it to get the reduced :class:`MPIXarray`.

        """
        return MPIResample(self, dim, freq)

    def weighted(
        self, weights: MPIXarray | xr.Dataset | xr.DataArray
    ) -> MPIXarrayWeighted:
        """Weighted-reduction handle, mirroring ``xarray.Dataset.weighted``.

        Parameters
        ----------
        weights : MPIXarray, xarray.Dataset, or xarray.DataArray
            Weights (e.g. ``cos(lat)`` area weights). A raw
            ``xarray.Dataset``/``DataArray`` is passed straight through
            to every composed operation unwrapped, exactly like any other
            raw-array binary operand (see the note above ``__add__``) --
            it is *not* auto-partitioned, since weights are overwhelmingly
            replicated (the same full array on every rank, broadcasting
            against whichever dimension ``self`` happens to be
            partitioned along) rather than independently distributed. An
            already-distributed ``MPIXarray`` is also accepted and, like
            any other binary operand, must be compatibly partitioned with
            ``self``.

        Returns
        -------
        MPIXarrayWeighted
            Chainable handle; call ``.sum()``, ``.mean()``, or
            ``.sum_of_weights()`` on it to get the reduced :class:`MPIXarray`.

        """
        return MPIXarrayWeighted(self, weights)

    def align(
        self,
        other: MPIXarray | xr.Dataset | xr.DataArray,
        dim: Hashable | Literal["auto"] | None = None,
        *,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> tuple[MPIXarray, MPIXarray]:
        """Return ``(self, other)`` partitioned identically across ranks.

        Parameters
        ----------
        other : MPIXarray, xarray.Dataset, or xarray.DataArray
            Operand to align against ``self``.
        dim : Hashable or {"auto"} or None, optional
            Dimension to partition both operands along when neither is yet distributed.
        chunk_info : mapping, optional
            Forwarded to ``repartition`` when neither operand is yet distributed.
        log_partitions : bool, optional
            Forwarded to ``repartition`` when neither operand is yet distributed.

        Returns
        -------
        tuple of MPIXarray
            ``(left, right)``, each with matching distribution metadata.

        """
        left, right = mpp_align(
            self._runtime,
            self._prepare(),
            unwrap(other),
            dim,
            chunk_info=chunk_info,
            log_partitions=log_partitions,
        )
        return finalize(left, self._runtime), finalize(right, self._runtime)

    def reindex(
        self,
        indexers: Mapping[Hashable, Any] | None = None,
        *,
        method: str | None = None,
        tolerance: float | Iterable[float] | None = None,
        fill_value: Any = _FILL_VALUE_UNSET,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
        **indexers_kwargs: Any,
    ) -> MPIXarray:
        """Reindex onto new coordinate labels, redistributing if needed.

        Parameters
        ----------
        indexers : mapping, optional
            New coordinate labels per dimension.
        method : str, optional
            Forwarded to xarray's ``reindex``.
        tolerance : float or iterable of float, optional
            Forwarded to xarray's ``reindex``.
        fill_value : Any, optional
            Value used for labels with no match.
        chunk_info : mapping, optional
            Forwarded to ``repartition`` when a partition dimension is reindexed.
        log_partitions : bool, optional
            Forwarded to ``repartition`` when a partition dimension is reindexed.
        **indexers_kwargs : Any
            Additional indexers given as keywords.

        Returns
        -------
        MPIXarray
            The reindexed object.

        """
        kwargs: dict[str, Any] = {
            "method": method,
            "tolerance": tolerance,
            "chunk_info": chunk_info,
            "log_partitions": log_partitions,
        }
        if fill_value is not _FILL_VALUE_UNSET:
            kwargs["fill_value"] = fill_value
        return finalize(
            mpp_reindex(
                self._runtime, self._prepare(), indexers, **kwargs, **indexers_kwargs
            ),
            self._runtime,
        )

    def sortby(
        self,
        by: Hashable | Any | Iterable[Hashable | Any],
        *,
        ascending: bool = True,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> MPIXarray:
        """Sort by one or more keys, redistributing if needed.

        Parameters
        ----------
        by : Hashable, DataArray, or sequence of these
            Sort key(s): variable/coordinate name(s) or explicit DataArray(s).
        ascending : bool, optional
            Sort order.
        chunk_info : mapping, optional
            Forwarded to ``repartition`` when a partition dimension is sorted.
        log_partitions : bool, optional
            Forwarded to ``repartition`` when a partition dimension is sorted.

        Returns
        -------
        MPIXarray
            The sorted object.

        """
        return finalize(
            mpp_sortby(
                self._runtime,
                self._prepare(),
                by,
                ascending=ascending,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            self._runtime,
        )

    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

        Parameters
        ----------
        func : callable
            Partition-preserving, rank-local function of ``args``/``kwargs``.
        *args : Any
            Positional arguments to ``func``.
        **kwargs : Any
            Keyword arguments to ``func``.

        Returns
        -------
        MPIXarray or Any
            ``func``'s result, wrapped if it is an xarray Dataset/DataArray, otherwise returned as-is.

        """
        unwrapped_args = tuple(unwrap(arg) for arg in args)
        unwrapped_kwargs = {name: unwrap(value) for name, value in kwargs.items()}
        result = mpp_apply(self._runtime, func, *unwrapped_args, **unwrapped_kwargs)
        return finalize(result, self._runtime)

    def matmul(self, right: MPIXarray | Any) -> MPIXarray:
        """Matrix multiplication (``self @ right``), correct under MPI.

        Parameters
        ----------
        right : MPIXarray or Any
            Right operand: an xarray DataArray (distributed or not, wrapped or not) or a plain array/scalar.

        Returns
        -------
        MPIXarray
            The matrix product.

        """
        result = mpp_matmul(self._runtime, self._prepare(), unwrap(right))
        return finalize(result, self._runtime)

    def rolling_reduce(
        self,
        dim: Hashable,
        window: int,
        reduce: str = "mean",
        *,
        center: bool = True,
        min_periods: int | None = None,
    ) -> MPIXarray:
        """Windowed reduction along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to roll over.
        window : int
            Window size, as in ``xarray.DataArray.rolling``.
        reduce : str, optional
            Name of the reduction to call on the rolling object (e.g.
        center : bool, optional
            As in ``xarray.DataArray.rolling``.
        min_periods : int or None, optional
            As in ``xarray.DataArray.rolling``.

        Returns
        -------
        MPIXarray
            Rolled-and-reduced object with ``.meta`` preserved.

        """
        result = mpp_rolling_reduce(
            self._runtime,
            self._prepare(),
            dim,
            window,
            reduce,
            center=center,
            min_periods=min_periods,
        )
        return finalize(result, self._runtime)

    def coarsen_reduce(
        self,
        dim: Hashable,
        window: int,
        reduce: str = "mean",
        *,
        boundary: str = "exact",
        side: str = "left",
        coord_func: str = "mean",
    ) -> MPIXarray:
        """Block reduction along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to coarsen along.
        window : int
            Block size, as in ``xarray.DataArray.coarsen``.
        reduce : str, optional
            Name of the reduction to call on the coarsen object (e.g.
        boundary : {"exact", "trim", "pad"}, optional
            As in ``xarray.DataArray.coarsen``.
        side : {"left"}, optional
            Only "left" (the ``xarray`` default) is supported for a distributed dimension so far.
        coord_func : str, optional
            As in ``xarray.DataArray.coarsen``.

        Returns
        -------
        MPIXarray
            Coarsened-and-reduced object with ``.meta`` updated to match the new, block-reduced length along ``dim``.

        """
        result = mpp_coarsen_reduce(
            self._runtime,
            self._prepare(),
            dim,
            window,
            reduce,
            boundary=boundary,
            side=side,
            coord_func=coord_func,
        )
        return finalize(result, self._runtime)

    def rolling(
        self,
        dim: Hashable,
        window: int,
        *,
        center: bool = True,
        min_periods: int | None = None,
    ) -> MPIRolling:
        """Windowed rolling handle, mirroring ``xarray.DataArray.rolling``.

        Parameters
        ----------
        dim : Hashable
            Dimension to roll over.
        window : int
            Window size, as in ``xarray.DataArray.rolling``.
        center : bool, optional
            As in ``xarray.DataArray.rolling``.
        min_periods : int or None, optional
            As in ``xarray.DataArray.rolling``.

        Returns
        -------
        MPIRolling
            Chainable handle; call ``.mean()``, ``.sum()``, ``.min()``, ``.max()``, ``.std()``, or ``.count()`` on it to get the rolled-and-reduced :class:`MPIXarray`.

        """
        return MPIRolling(self, dim, window, center=center, min_periods=min_periods)

    def evaluate(self, expression: str, /, **variables: Any) -> Any:
        """Evaluate a string expression, respecting normal operator precedence.

        Parameters
        ----------
        expression : str
            A Python expression referencing ``variables`` by name, e.g.
        **variables : Any
            Values bound to the names used in ``expression``.

        Returns
        -------
        MPIXarray or Any
            The expression's value, wrapped if it is an xarray Dataset/DataArray, otherwise returned as-is.

        """
        unwrapped = {name: unwrap(value) for name, value in variables.items()}
        result = mpp_evaluate(self._runtime, expression, **unwrapped)
        return finalize(result, self._runtime)

    def where(
        self,
        cond: MPIXarray | xr.Dataset | xr.DataArray | Any,
        other: MPIXarray | xr.Dataset | xr.DataArray | Any = _WHERE_UNSET,
        *,
        drop: bool = False,
    ) -> MPIXarray:
        """Elementwise selection, mirroring ``xarray.DataArray.where``.

        Parameters
        ----------
        cond : MPIXarray, xarray.Dataset, xarray.DataArray, or Any
            Boolean condition, following ``xarray.DataArray.where``.
        other : MPIXarray, xarray.Dataset, xarray.DataArray, or Any, optional
            Fill value where ``cond`` is False.
        drop : bool, optional
            Must be False when ``self`` is distributed; ``drop=True`` can remove a different number of positions on different ranks.

        Returns
        -------
        MPIXarray
            The selected object, with ``.meta`` unchanged.

        """
        args = (cond,) if other is _WHERE_UNSET else (cond, other)
        return finalize(
            mpp_where(
                self._runtime,
                self._prepare(),
                *(unwrap(arg) for arg in args),
                drop=drop,
            ),
            self._runtime,
        )

    def cumsum(
        self,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Cumulative sum along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to accumulate along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.

        Returns
        -------
        MPIXarray
            Cumulative sum with ``.meta`` unchanged.

        """
        return finalize(
            mpp_cumsum(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
            ),
            self._runtime,
        )

    def cumprod(
        self,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Cumulative product along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to accumulate along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.

        Returns
        -------
        MPIXarray
            Cumulative product with ``.meta`` unchanged.

        """
        return finalize(
            mpp_cumprod(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
            ),
            self._runtime,
        )

    def median(
        self,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Median over ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.

        Returns
        -------
        MPIXarray
            Reduced object.

        """
        return finalize(
            mpp_median(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
            ),
            self._runtime,
        )

    def quantile(
        self,
        q: float | Iterable[float],
        dim: Hashable,
        *,
        method: str = "linear",
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Quantile(s) over ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        q : float or iterable of float
            Quantile(s) to compute, in ``[0, 1]``, following ``xarray.DataArray.quantile``.
        dim : Hashable
            Dimension to reduce.
        method : str, optional
            Interpolation method, following ``xarray.DataArray.quantile``.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.

        Returns
        -------
        MPIXarray
            Reduced object, with a new ``quantile`` dimension/coordinate
            when ``q`` is a sequence of more than one value.

        """
        return finalize(
            mpp_quantile(
                self._runtime,
                self._prepare(),
                q,
                dim,
                method=method,
                skipna=skipna,
                keep_attrs=keep_attrs,
            ),
            self._runtime,
        )

    def diff(
        self,
        dim: Hashable,
        n: int = 1,
        *,
        label: Literal["upper", "lower"] = "upper",
    ) -> MPIXarray:
        """``n``-th order difference along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to difference along.
        n : int, optional
            Order of the difference.
        label : {"upper", "lower"}, optional
            As in ``xarray.DataArray.diff``.

        Returns
        -------
        MPIXarray
            The differenced object, ``n`` elements shorter along ``dim`` globally, with ``.meta`` updated to match.

        """
        return finalize(
            mpp_diff(self._runtime, self._prepare(), dim, n, label=label), self._runtime
        )

    def shift(
        self,
        dim: Hashable,
        periods: int = 1,
        *,
        fill_value: Any = _FILL_VALUE_UNSET,
    ) -> MPIXarray:
        """Shift by ``periods`` along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to shift along.
        periods : int, optional
            Number of positions to shift by.
        fill_value : Any, optional
            As in ``xarray.DataArray.shift``; defaults to xarray's own dtype-aware NA fill when omitted.

        Returns
        -------
        MPIXarray
            The shifted object, same shape and distribution as ``self``.

        """
        kwargs: dict[str, Any] = {}
        if fill_value is not _FILL_VALUE_UNSET:
            kwargs["fill_value"] = fill_value
        return finalize(
            mpp_shift(self._runtime, self._prepare(), dim, periods, **kwargs),
            self._runtime,
        )

    def pad(
        self,
        dim: Hashable,
        pad_width: tuple[int, int],
        *,
        mode: str = "constant",
        constant_values: Any = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Pad along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to pad along.
        pad_width : tuple[int, int]
            ``(before, after)``, as in ``xarray.DataArray.pad``.
        mode : str, optional
            Padding mode, as in ``xarray.DataArray.pad``. Only
            ``"constant"`` is supported when ``dim`` is distributed.
        constant_values : Any, optional
            Fill value(s) for ``mode="constant"``.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.

        Returns
        -------
        MPIXarray
            The padded object.

        Raises
        ------
        NotImplementedError
            If ``dim`` is distributed and ``mode`` is not ``"constant"``.

        """
        return finalize(
            mpp_pad(
                self._runtime,
                self._prepare(),
                dim,
                pad_width,
                mode=mode,
                constant_values=constant_values,
                keep_attrs=keep_attrs,
            ),
            self._runtime,
        )

    def roll(self, dim: Hashable, shift_by: int) -> MPIXarray:
        """Circularly shift by ``shift_by`` along ``dim``, wrapping at the edge.

        Parameters
        ----------
        dim : Hashable
            Dimension to roll along.
        shift_by : int
            Number of positions to roll by; positive rolls toward higher indices, matching ``xarray.DataArray.roll``.

        Returns
        -------
        MPIXarray
            The rolled object, same shape and distribution as ``self``.

        """
        return finalize(
            mpp_roll(self._runtime, self._prepare(), dim, shift_by), self._runtime
        )

    def ffill(self, dim: Hashable, limit: int | None = None) -> MPIXarray:
        """Forward-fill along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to fill along.
        limit : int or None, optional
            As in ``xarray.DataArray.ffill``.

        Returns
        -------
        MPIXarray
            The forward-filled object, same shape and distribution as ``self``.

        """
        return finalize(
            mpp_ffill(self._runtime, self._prepare(), dim, limit), self._runtime
        )

    def bfill(self, dim: Hashable, limit: int | None = None) -> MPIXarray:
        """Backward-fill along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to fill along.
        limit : int or None, optional
            As in ``xarray.DataArray.bfill``.

        Returns
        -------
        MPIXarray
            The backward-filled object, same shape and distribution as ``self``.

        """
        return finalize(
            mpp_bfill(self._runtime, self._prepare(), dim, limit), self._runtime
        )

    def interp(
        self, dim: Hashable, new_coord: Any, method: str = "linear", **kwargs: Any
    ) -> MPIXarray:
        """Interpolate onto ``new_coord`` along ``dim``, correct when distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to interpolate along.
        new_coord : array-like
            This rank's own local slice of the new target coordinate.
        method : str, optional
            As in ``xarray.DataArray.interp``.
        **kwargs : Any
            Forwarded to ``xarray.DataArray.interp``.

        Returns
        -------
        MPIXarray
            Interpolated result, with ``.meta`` recomputed for the new length along ``dim``.

        """
        return finalize(
            mpp_interp(
                self._runtime, self._prepare(), dim, new_coord, method, **kwargs
            ),
            self._runtime,
        )

    def differentiate(
        self,
        coord: Hashable,
        edge_order: Literal[1, 2] = 1,
        datetime_unit: Any = None,
    ) -> MPIXarray:
        """Differentiate along ``coord``, correct when ``coord`` is distributed.

        Parameters
        ----------
        coord : Hashable
            Coordinate to differentiate along.
        edge_order : {1, 2}, optional
            As in ``xarray.DataArray.differentiate``.
        datetime_unit : Any, optional
            As in ``xarray.DataArray.differentiate``.

        Returns
        -------
        MPIXarray
            The derivative, same shape and distribution as ``self``.

        """
        return finalize(
            mpp_differentiate(
                self._runtime,
                self._prepare(),
                coord,
                edge_order=edge_order,
                datetime_unit=datetime_unit,
            ),
            self._runtime,
        )


def mark_partitioned(
    data: xr.Dataset | xr.DataArray, meta: dict[str, Any] | None
) -> xr.Dataset | xr.DataArray:
    """Set the lightweight partition marker from MPI metadata.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to copy and mark.
    meta : dict or None
        Distribution metadata, or None for replicated data.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Shallow copy whose partition marker matches ``meta``.

    """
    marked = data.copy(deep=False)
    if meta is None:
        marked.attrs.pop(_PARTITIONED_ATTR, None)
        if isinstance(marked, xr.Dataset):
            for variable in marked.variables.values():
                variable.attrs.pop(_PARTITIONED_ATTR, None)
        return marked

    dims = meta["dims"]
    marked.attrs[_PARTITIONED_ATTR] = True
    if isinstance(marked, xr.Dataset):
        for variable in marked.variables.values():
            variable.attrs.pop(_PARTITIONED_ATTR, None)
            if any(dim in variable.dims for dim in dims):
                variable.attrs[_PARTITIONED_ATTR] = True
    return marked


def finalize(result: Any, mpi_context: MPIContext) -> Any:
    """Wrap an engine result when it is an xarray object.

    Parameters
    ----------
    result : Any
        Result returned by ``_MPIXarrayOps``.
    mpi_context : MPIContext
        Runtime bound to the wrapped result.

    Returns
    -------
    MPIXarray or Any
        Wrapped xarray result, or the original non-xarray value.

    """
    if isinstance(result, (xr.Dataset, xr.DataArray)):
        return MPIXarray(result, mpi_context, auto_partition=False)
    return result


def unwrap(value: Any) -> Any:
    """Prepare an ``MPIXarray`` operand for engine use.

    Parameters
    ----------
    value : Any
        Candidate operand.

    Returns
    -------
    Any
        Prepared xarray data for ``MPIXarray`` input, otherwise ``value``.

    """
    return value._prepare() if isinstance(value, MPIXarray) else value


@wraps(dataarray_binary_op)
def _da_binary_op(
    self: xr.DataArray, other: Any, f: Any, reflexive: bool = False
) -> Any:
    """Defer mixed binary operations to ``MPIXarray``."""
    if isinstance(other, MPIXarray):
        return NotImplemented
    return dataarray_binary_op(self, other, f, reflexive)


@wraps(dataset_binary_op)
def _ds_binary_op(
    self: xr.Dataset,
    other: Any,
    f: Any,
    reflexive: bool = False,
    join: Any = None,
) -> Any:
    """Defer mixed binary operations to ``MPIXarray``."""
    if isinstance(other, MPIXarray):
        return NotImplemented
    return dataset_binary_op(self, other, f, reflexive, join)


xr.DataArray._binary_op = _da_binary_op
xr.Dataset._binary_op = _ds_binary_op
