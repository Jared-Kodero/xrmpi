"""Provide chainable rolling, groupby, resample, and weighted handles for MPIXarray."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable
    from typing import Any, Literal

    import numpy as np

    from .core import MPIXarray


class _RollingReduceMixin:
    """Provide zero-argument rolling reductions."""

    def _reduce(self, reduce: str) -> MPIXarray:
        """Apply the named reduction; supplied by the concrete handle."""
        raise NotImplementedError

    def mean(self) -> MPIXarray:
        """Rolling mean."""
        return self._reduce("mean")

    def sum(self) -> MPIXarray:
        """Rolling sum."""
        return self._reduce("sum")

    def min(self) -> MPIXarray:
        """Rolling minimum."""
        return self._reduce("min")

    def max(self) -> MPIXarray:
        """Rolling maximum."""
        return self._reduce("max")

    def std(self) -> MPIXarray:
        """Rolling standard deviation."""
        return self._reduce("std")

    def count(self) -> MPIXarray:
        """Rolling valid-value count."""
        return self._reduce("count")


class _PartitionReduceMixin:
    """Provide partition-aware groupby and resample reductions."""

    def _reduce(
        self,
        op: Literal["sum", "mean", "count", "min", "max"],
        *,
        skipna: bool | None,
        keep_attrs: bool | None,
        partition_dim: Hashable | Literal["auto"] | None,
    ) -> MPIXarray:
        """Apply the named reduction; supplied by the concrete handle."""
        raise NotImplementedError

    def _reduce_kw(
        self,
        op: Literal["sum", "mean", "count", "min", "max"],
        skipna: bool | None,
        keep_attrs: bool | None,
        partition_dim: Hashable | Literal["auto"] | None,
    ) -> MPIXarray:
        """Forward keyword args to :meth:`_reduce`; shared by every method below."""
        return self._reduce(
            op, skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def sum(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Sum within each partition.

        Parameters
        ----------
        skipna : bool | None
            Whether to ignore missing values.
        keep_attrs : bool | None
            Whether to preserve xarray attributes.
        partition_dim : Hashable | Literal['auto'] | None
            Partition dimension to use for the result.

        Returns
        -------
        MPIXarray
            Reduced result.

        """
        return self._reduce_kw("sum", skipna, keep_attrs, partition_dim)

    def mean(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Mean within each partition.

        Parameters
        ----------
        skipna : bool | None
            Whether to ignore missing values.
        keep_attrs : bool | None
            Whether to preserve xarray attributes.
        partition_dim : Hashable | Literal['auto'] | None
            Partition dimension to use for the result.

        Returns
        -------
        MPIXarray
            Reduced result.

        """
        return self._reduce_kw("mean", skipna, keep_attrs, partition_dim)

    def count(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Valid-value count within each partition.

        Parameters
        ----------
        skipna : bool | None
            Whether to ignore missing values.
        keep_attrs : bool | None
            Whether to preserve xarray attributes.
        partition_dim : Hashable | Literal['auto'] | None
            Partition dimension to use for the result.

        Returns
        -------
        MPIXarray
            Reduced result.

        """
        return self._reduce_kw("count", skipna, keep_attrs, partition_dim)

    def min(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Minimum within each partition.

        Parameters
        ----------
        skipna : bool | None
            Whether to ignore missing values.
        keep_attrs : bool | None
            Whether to preserve xarray attributes.
        partition_dim : Hashable | Literal['auto'] | None
            Partition dimension to use for the result.

        Returns
        -------
        MPIXarray
            Reduced result.

        """
        return self._reduce_kw("min", skipna, keep_attrs, partition_dim)

    def max(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Maximum within each partition.

        Parameters
        ----------
        skipna : bool | None
            Whether to ignore missing values.
        keep_attrs : bool | None
            Whether to preserve xarray attributes.
        partition_dim : Hashable | Literal['auto'] | None
            Partition dimension to use for the result.

        Returns
        -------
        MPIXarray
            Reduced result.

        """
        return self._reduce_kw("max", skipna, keep_attrs, partition_dim)


class MPIRolling(_RollingReduceMixin):
    """Chainable rolling-window handle, mirroring ``xarray``'s ``.rolling(...)``.

    Returned by :meth:`MPIXarray.rolling`; call one of the reduction
    methods below to compute the windowed reduction via
    :meth:`MPIXarray.rolling_reduce`.

    Parameters
    ----------
    parent : MPIXarray
        Object to roll over.
    dim : Hashable
        Dimension to roll over.
    window : int
        Window size, as in ``xarray.DataArray.rolling``.
    center : bool, optional
        As in ``xarray.DataArray.rolling``.
    min_periods : int or None, optional
        As in ``xarray.DataArray.rolling``.

    """

    def __init__(
        self,
        parent: MPIXarray,
        dim: Hashable,
        window: int,
        *,
        center: bool = True,
        min_periods: int | None = None,
    ) -> None:
        """Initialize a rolling-operation handle."""
        self._parent = parent
        self._dim = dim
        self._window = window
        self._center = center
        self._min_periods = min_periods

    def _reduce(self, reduce: str) -> MPIXarray:
        """Dispatch a rolling reduction through the parent wrapper."""
        return self._parent.rolling_reduce(
            self._dim,
            self._window,
            reduce,
            center=self._center,
            min_periods=self._min_periods,
        )


class MPIGroupBy(_PartitionReduceMixin):
    """Chainable groupby handle, mirroring ``xarray``'s ``.groupby(...)``.

    Returned by :meth:`MPIXarray.groupby`; call one of the reduction
    methods below to compute the grouped reduction via the internal
    ``mpp_groupby_reduce`` engine dispatch.

    Parameters
    ----------
    parent : MPIXarray
        Object being grouped.
    dim : Hashable
        Dimension being grouped and reduced.
    labels : array-like
        Group key for every position along this rank's local ``dim`` axis.

    """

    def __init__(
        self,
        parent: MPIXarray,
        dim: Hashable,
        labels: xr.DataArray | np.ndarray[Any, Any],
    ) -> None:
        """Initialize a groupby-operation handle."""
        self._parent = parent
        self._dim = dim
        self._labels = labels

    def _reduce(
        self,
        op: Literal["sum", "mean", "count", "min", "max"],
        *,
        skipna: bool | None,
        keep_attrs: bool | None,
        partition_dim: Hashable | Literal["auto"] | None,
    ) -> MPIXarray:
        """Dispatch a grouped reduction through the parent wrapper."""
        from .core import finalize
        from .groupby import mpp_groupby_reduce

        return finalize(
            mpp_groupby_reduce(
                self._parent._runtime,
                self._parent._prepare(),
                self._dim,
                self._labels,
                op,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._parent._runtime,
        )


class MPIResample(_PartitionReduceMixin):
    """Chainable resample handle, mirroring ``xarray``'s ``.resample(...)``.

    Returned by :meth:`MPIXarray.resample`; call one of the reduction
    methods below to compute the resampled reduction via the internal
    ``mpp_resample_reduce`` engine dispatch.

    Parameters
    ----------
    parent : MPIXarray
        Object being resampled.
    dim : Hashable
        Datetime dimension to resample.
    freq : str
        Pandas offset alias (e.g. "D", "MS", "YS").

    """

    def __init__(self, parent: MPIXarray, dim: Hashable, freq: str) -> None:
        """Initialize a resample-operation handle."""
        self._parent = parent
        self._dim = dim
        self._freq = freq

    def _reduce(
        self,
        op: Literal["sum", "mean", "count", "min", "max"],
        *,
        skipna: bool | None,
        keep_attrs: bool | None,
        partition_dim: Hashable | Literal["auto"] | None,
    ) -> MPIXarray:
        """Dispatch a resampled reduction through the parent wrapper."""
        from .core import finalize
        from .groupby import mpp_resample_reduce

        return finalize(
            mpp_resample_reduce(
                self._parent._runtime,
                self._parent._prepare(),
                self._dim,
                self._freq,
                op,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._parent._runtime,
        )


class MPIXarrayWeighted:
    """Represent weighted reductions on an :class:`MPIXarray`.

    Parameters
    ----------
    parent : MPIXarray
        Object to reduce.
    weights : MPIXarray or xarray object
        Weights broadcast against ``parent`` using normal binary-operation rules.
    """

    def __init__(
        self, parent: MPIXarray, weights: MPIXarray | xr.Dataset | xr.DataArray
    ) -> None:
        """Initialize a weighted-operation handle."""
        self._parent = parent
        self._weights = weights

    def sum_of_weights(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Sum weights applied to valid parent values.

        Parameters
        ----------
        dim : Hashable or iterable of Hashable, optional
            Dimension or dimensions to reduce.
        keep_attrs : bool or None, optional
            Preserve attributes.

        Returns
        -------
        MPIXarray
            Sum of weights after masking invalid parent values.
        """
        mask = self._parent.notnull()
        masked_weights = mask * self._weights
        return masked_weights.sum(dim=dim, skipna=False, keep_attrs=keep_attrs)

    def sum(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Weighted sum over ``dim``."""
        return (self._parent * self._weights).sum(
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
        )

    def mean(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Weighted mean over ``dim`` -- ``sum() / sum_of_weights()``."""
        weighted_sum = self.sum(
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
        )
        weights_sum = self.sum_of_weights(dim=dim, keep_attrs=keep_attrs)
        return weighted_sum / weights_sum
