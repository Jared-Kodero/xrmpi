"""Expose geographic, plot, xgeo_core_calculation, and preprocessing xarray accessors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from ...xr_mpi import io as xgeo_xarray_io
from ..core import climtools, xnpy
from ..core import climtools as xgeo_core_utils
from ..core import preprocess as xgeo_core_preprocess
from ..core import stats as xgeo_core_calc
from ..viz import plot
from . import xr_utils as xgeo_xarray_utils

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any, Literal

    import numpy as np
    from IPython.display import DisplayHandle
    from matplotlib.collections import PathCollection
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
    from mpi4py import MPI

    from ...xrmpi.mpi.context import MPIContext
    from ..viz.plot import GeoPlot


class GeoBase:
    """Operations shared by the DataArray and Dataset accessors."""

    __slots__ = ("_obj",)

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset) -> None:
        """Initialize the geographic accessor."""
        self._obj = xarray_obj

    def __repr__(self) -> str:
        """Return the geographic accessor representation."""
        kind = type(self._obj).__name__
        dims = ", ".join(f"{name}: {size}" for name, size in self._obj.sizes.items())
        return f"<xgeo accessor on {kind} ({dims})>"

    # -- regridding and masking ------------------------------------------
    def regrid(
        self,
        grid_out: xr.Dataset | xr.DataArray | None = None,
        grid_out_resolution: float | None = None,
        method: Literal[
            "bilinear",
            "conservative",
            "conservative_normed",
            "patch",
            "nearest_s2d",
            "nearest_d2s",
        ] = "bilinear",
        unmapped_to_nan: bool = True,
        parallel: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """Regrid onto the horizontal grid of ``grid_out`` or a target resolution.

        Parameters
        ----------
        grid_out : xarray.Dataset or xarray.DataArray, optional
            Object whose 'lat' and 'lon' coordinates define the target grid.
        grid_out_resolution : float, optional
            Target grid resolution in degrees if ``grid_out`` is not provided.
        method : {"bilinear", "conservative", "conservative_normed", "patch", "nearest_s2d", "nearest_d2s"}, default "bilinear"
            ESMF regridding method.
        unmapped_to_nan : bool, default True
            Whether to set unmapped target points to NaN.
        parallel : bool, default False
            Build the weights in parallel with Dask.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object on the target grid.
        """

        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        return xgeo_xarray_utils.regrid(self._obj, **kwargs)

    def mask(
        self,
        mask: xr.DataArray | xr.Dataset | str | Path | None = None,
        data_var: str = "land",
        valid_value: float = 1,
        parallel: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """Mask grid cells that do not match a specified land-sea mask value.

        Parameters
        ----------
        mask : xarray.DataArray, xarray.Dataset, str, pathlib.Path, or None, optional
            Categorical land-sea mask.
        data_var : str, default "land"
            Name of the mask variable to extract when ``mask`` is a Dataset or a path to a Dataset.
        valid_value : float or int, default 1
            Mask value identifying grid cells to retain.
        parallel : bool, default False
            Whether to perform mask remapping in parallel with Dask.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            A latitude- and longitude-sorted object with cells outside the retained mask category replaced by NaN.

        Raises
        ------
        KeyError
            If ``mask`` resolves to a Dataset that does not contain ``data_var``.
        TypeError
            If ``mask`` cannot be resolved to an xarray.DataArray.

        """
        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        return xgeo_xarray_utils.mask(self._obj, **kwargs)

    # -- coordinate helpers ----------------------------------------------
    def add_local_solar_time(
        self,
        *,
        lon: str = "lon",
        time: str = "time",
        name: str = "lst",
    ) -> xr.Dataset | xr.DataArray:
        """Add mean local solar time as a coordinate.

        Parameters
        ----------
        lon : str, default "lon"
            Name of the longitude coordinate, in degrees east.
        time : str, default "time"
            Name of the UTC time coordinate.
        name : str, default "lst"
            Name given to the new coordinate.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object with the local solar time coordinate attached.

        """
        return xgeo_xarray_utils.add_local_solar_time(
            self._obj, lon=lon, time=time, name=name
        )

    def wrap_lon(
        self,
        convention: Literal["-180/180", "0/360"] = "-180/180",
        lon: str = "lon",
    ) -> xr.Dataset | xr.DataArray:
        """Wrap longitude coordinates to the specified convention."""
        return xgeo_xarray_utils.wrap_lon(self._obj, convention=convention, lon=lon)

    def add_cyclic_point(self, lon: str = "lon") -> xr.Dataset | xr.DataArray:
        """Append a cyclic longitude point, closing the seam at the date line.

        Parameters
        ----------
        lon : str, default "lon"
            Name of the longitude dimension.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object with one extra longitude point.

        """

        return xgeo_xarray_utils.add_cyclic_point(self._obj, lon=lon)

    # -- selection --------------------------------------------------------
    def sel_transect(
        self,
        x: float | None = None,
        y: float | None = None,
        orientation: float = 0.0,
        width: float = 1.0,
        *,
        xdim: str | None = None,
        ydim: str | None = None,
        geometry: Literal["xy", "latlon"] = "latlon",
        snap: bool = True,
        drop: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """Select cells lying within a transect on a rectilinear xarray grid.

        Parameters
        ----------
        x, y : float | None
            Transect centre.
        orientation : float
            Transect orientation in degrees clockwise from the positive y direction.
        width : float
            Transect width in approximate grid-cell units.
        xdim, ydim : str | None
            Names of the x and y coordinates.
        geometry : Literal['xy', 'latlon']
            ``"xy"`` for planar coordinates or ``"latlon"`` for longitude-latitude coordinates in degrees.
        snap : bool
            Snap the supplied centre coordinates to the nearest grid point.
        drop : bool
            Drop coordinate locations outside the transect.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Selected transect subset.

        """
        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        return xgeo_xarray_utils.sel_transect(self._obj, **kwargs)

    # -- NetCDF output -----------------------------------------------------
    def append(
        self,
        file: str | Path,
        dim: str = "time",
        mode: Literal["a", "r+"] = "r+",
        format: str = "NETCDF4",
        shuffle: bool | None = None,
        zlib: bool | None = None,
        complevel: int | None = None,
    ) -> None:
        """Append the bound Dataset to an existing file along an unlimited dimension.

        Parameters
        ----------
        file : str or pathlib.Path
            NetCDF4 file with read/write access.
        dim : str, default "time"
            Unlimited dimension to append along.
        mode : {"a", "r+"}, default "r+"
            File access mode passed to netCDF4.Dataset.
        format : str, default "NETCDF4"
            NetCDF format passed to netCDF4.Dataset.
        shuffle : bool, optional
            Whether to apply the shuffle filter to newly created variables.
        zlib : bool, optional
            Whether to apply zlib compression to newly created variables.
        complevel : int, optional
            Compression level, between 1 and 9.

        Returns
        -------
        None

        """
        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        return xgeo_xarray_io.nc_append(self._obj, **kwargs)

    def to_netcdf(
        self,
        file: str | Path,
        mpi_context: MPIContext | MPI.Intracomm | None = None,
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
        """Write the bound Dataset or DataArray to NetCDF.

        Parameters
        ----------
        file : str or pathlib.Path
            Output path.
        mpi_context : MPIContext or mpi4py.MPI.Intracomm, optional
            MPI context or communicator.
        unlimited_dim : str or iterable of str, optional
            Dimension(s) made unlimited in the NetCDF schema.
        partition_dim : str, optional
            Dimension partitioned across MPI ranks in parallel mode.
        parallel : bool, default False
            Use the MPI-parallel NetCDF-4 writer.
        batch_size : int, default 24
            Number of slices along the unlimited dimension written per serial append.
        format : str, default "NETCDF4"
            NetCDF format.
        shuffle : bool, default True
            Apply the HDF5 shuffle filter.
        zlib : bool, default True
            Apply zlib compression.
        complevel : int, default 4
            Compression level, between 1 and 9.
        show_progress : bool, default True
            Display a progress bar while writing serially.
        stdout : file-like, optional
            Stream the serial progress bar is written to.
        chunks : mapping of str to iterable of int, optional
            Explicit chunk shape passed to the parallel writer.
        hints : str, optional
            Semicolon-separated MPI-IO hints in key=value format.
        nofill : bool, default True
            Disable NetCDF pre-filling during parallel initialization.
        allow_serial : bool, default False
            Permit execution when running with a single MPI rank.

        Returns
        -------
        None

        """

        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        return xgeo_xarray_io.to_netcdf(self._obj, **kwargs)

    def to_xnpy(
        self,
        path: str | Path,
        *,
        mode: Literal["w", "w-"] = "w-",
        scheduler: Literal["threads", "synchronous"] = "threads",
        num_workers: int | None = None,
    ):
        """Write a NumPy or xarray object to a memory-mappable XNpy store.

        All array writes (variables, coordinates, columns, masks) are built as
        :func:`dask.delayed` tasks and executed in a single :func:`dask.compute`
        call. Chunked xarray variables are written incrementally to an NPY memory
        map using :func:`numpy.lib.format.open_memmap` through
        :func:`dask.array.store`, avoiding materialization of the complete variable
        in memory. Unchunked payloads smaller than the slab size are written with
        :func:`numpy.save`, and larger ones are streamed in bounded first-axis
        slabs.

        Parameters
        ----------
        path : str or pathlib.Path
            Store directory path.
        mode : {"w", "w-"}, default "w-"
            Write mode. ``"w"`` replaces an existing store, and ``"w-"`` requires
            that the store does not already exist.
        scheduler : {"threads", "synchronous"}, default "threads"
            Dask scheduler used to execute all write tasks. ``"threads"`` writes
            concurrently and ``"synchronous"`` writes serially in the calling
            thread.
        num_workers : int, optional
            Number of workers passed to :func:`dask.compute`. If omitted, the Dask
            default is used.


        Notes
        -----
        For xarray objects, each variable is written according to its storage layout.
        Variables with ``variable.chunks is not None`` are written chunk-by-chunk
        through an NPY memory map. Unchunked variables are materialized and written
        with :func:`numpy.save`, or streamed in slabs when large. Bare NumPy arrays
        are written the same way.
        """
        return xnpy.to_xnpy(
            self._obj,
            path,
            mode=mode,
            scheduler=scheduler,
            num_workers=num_workers,
        )

    def shared_memory(self, *, readonly: bool = True) -> climtools.SharedMemoryObject:
        """Create an interprocess shared-memory representation.

        The numerical buffers backing the xarray object are copied into
        operating-system shared memory and may subsequently be attached by
        other processes without serializing or copying the bulk array data.

        When the returned object is transferred through Python
        multiprocessing, the receiving process reconstructs the original
        :class:`xarray.DataArray` or :class:`xarray.Dataset`. Its numerical
        arrays reference the shared-memory buffers directly.

        Parameters
        ----------
        readonly : bool, default True
            If True, reconstructed shared-memory arrays are marked
            read-only. If False, processes may modify the same underlying
            memory and synchronization is the responsibility of the caller.

        Returns
        -------
        SharedMemoryObject
            Shared-memory transport object. The creating process owns the
            allocated shared-memory segments and must keep this object alive
            while receiving processes are using them.

        Notes
        -----
        Pointer-free NumPy dtypes are stored directly in shared memory.
        Object-dtype arrays cannot be safely shared as raw memory because
        they contain process-local Python object pointers and therefore fall
        back to ordinary serialization.

        Lazy arrays, including Dask-backed xarray variables, are materialized
        when the shared-memory object is created.

        The returned object should normally be used as a context manager so
        that owned shared-memory segments are released after all worker
        processes have completed.

        See Also
        --------
        multiprocessing.shared_memory.SharedMemory
            Python interface to operating-system shared memory.
        """
        return climtools.SharedMemoryObject(self._obj, readonly=readonly)


@xr.register_dataarray_accessor("xgeo")
class GeoDataArray(GeoBase):
    """DataArray ``.xgeo`` accessor for geospatial, plot, and xgeo_core_calculation operations."""

    __slots__ = ()

    def geoplot(
        self,
        x: str | None = None,
        y: str | None = None,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        figsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = True,
        interactive: bool = False,
        method: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
        ] = "default",
        projection: Literal[
            "PlateCarree",
            "Mercator",
            "Robinson",
            "Mollweide",
            "Orthographic",
            "LambertConformal",
            "AlbersEqualArea",
            "Stereographic",
            "NorthPolarStereo",
            "SouthPolarStereo",
        ]
        | None = None,
        cmap: str | LinearSegmentedColormap | ListedColormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        units: str | None = None,
        levels: int | list[float] | tuple[float, ...] | None = None,
        extend: str | None = None,
        robust: bool = False,
        symmetrical: bool = False,
        rasterized: bool = False,
        title: str | dict[str, Any] | None = None,
        cbar_orientation: Literal["vertical", "horizontal"] | None = None,
        add_colorbar: bool = True,
        cbar_drawedges: bool = True,
        cbar_label: str | None = None,
        cbar_minimal_ticks: bool = True,
        map_global_extent: bool = False,
        set_map_extent: tuple[float, float, float, float] | None = None,
        map_ticks: bool = False,
        map_xtick_bins: int = 5,
        map_ytick_bins: int = 5,
        add_grid_bounds: bool = False,
        add_coastlines: bool = True,
        add_borders: bool = True,
        add_states: bool = True,
        add_ocean: bool = True,
        add_land: bool = True,
        add_lakes: bool = False,
        add_rivers: bool = False,
        p_value: xr.DataArray | None = None,
        pvalue_kwargs: dict[str, Any] | None = None,
        u_component: xr.DataArray | None = None,
        v_component: xr.DataArray | None = None,
        quiver_kwargs: dict[str, Any] | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        clabel: bool = False,
        clabel_fmt: str = "%1.0f",
        clabel_fontsize: float = 8,
        clabel_inline: bool = True,
        clabel_colors: str | None = None,
        clabel_kwargs: dict[str, Any] | None = None,
        cyclic: bool = False,
        **kwargs: Any,
    ) -> GeoPlot:
        """Plot the bound DataArray on a Cartopy map.

        Parameters
        ----------
        x, y, col, row : str, optional
            Coordinate names used for plot and faceting.
        col_wrap : int, optional
            Number of columns for wrapped facets.
        figsize : tuple[float, float], optional
            Figure size in inches.
        sharex, sharey : bool, default True
            Share horizontal and vertical axis limits across facet panels. Ignored
            for non-faceted plots.
        interactive : bool, default False
            Configure Matplotlib for interactive notebook use.
        method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
            Xarray plot method.
        projection : str, optional
            Cartopy projection name.
        cmap : str or matplotlib colormap, optional
            Colormap for the scalar field.
        norm : matplotlib.colors.Normalize, optional
            Color normalization.
        vmin, vmax : float, optional
            Scalar color limits.
        units : str, optional
            Units used for colorbar labeling.
        levels : int or sequence of float, optional
            Contour levels.
        extend : str, optional
            Colorbar extension mode.
        robust, rasterized : bool, default False
            Enable percentile scaling or rasterized artists.
        symmetrical : bool, default False
            Use symmetrical color limits around zero where supported.
        title : str or dict, optional
            Plot title specification.
        cbar_orientation : {"vertical", "horizontal"}, optional
            Colorbar orientation.
        add_colorbar, cbar_drawedges : bool
            Control colorbar creation and interval edges.
        cbar_label : str, optional
            Explicit colorbar label.
        cbar_minimal_ticks : bool, default True
            If True, reduce the number of ticks skipping every other one when possible.
        map_global_extent, add_grid_bounds : bool
            Control geographic extent and grid-boundary annotations.
        map_ticks : bool, default False
            Draw longitude and latitude ticks.
        map_xtick_bins : float, default 5
            Maximum number of longitude tick intervals.
        map_ytick_bins : float, default 5
            Maximum number of latitude tick intervals.
        set_map_extent : tuple[float, float, float, float], optional
            ``(lon_min, lon_max, lat_min, lat_max)``.
        add_coastlines, add_borders, add_states, add_ocean, add_land, add_lakes, add_rivers : bool
            Toggle Cartopy geographic features.
        p_value : xarray.DataArray, optional
            Pointwise p-values for significance markers.
        pvalue_kwargs : dict, optional
            Arguments passed to :meth:`significance`.
        u_component, v_component : xarray.DataArray, optional
            Vector components for a quiver overlay.
        quiver_kwargs, cbar_kwargs : dict, optional
            Quiver and colorbar options.
        clabel : bool, default False
            Label contour lines.
        clabel_fmt, clabel_colors : str, optional
            Contour-label format and color.
        clabel_fontsize : float, default 8
            Contour-label font size.
        clabel_inline : bool, default True
            Draw contour labels inline.
        clabel_kwargs : dict, optional
            Additional ``Axes.clabel`` arguments.
        cyclic : bool, default False
            Append a cyclic longitude point before plot.
        **kwargs : Any
            Additional xarray plot arguments.

        Returns
        -------
        GeoPlot
            Composable map object for geographic plots.
        """

        opts = xgeo_core_utils.exclude_key("self", dict(locals()))
        kwargs = opts.pop("kwargs")

        return plot.geoplot(self._obj, **opts, **kwargs)

    def animate(
        self,
        dim: str = "time",
        *,
        x: str | None = None,
        y: str | None = None,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        figsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = True,
        method: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
        ] = "default",
        projection: Literal[
            "PlateCarree",
            "Mercator",
            "Robinson",
            "Mollweide",
            "Orthographic",
            "LambertConformal",
            "AlbersEqualArea",
            "Stereographic",
            "NorthPolarStereo",
            "SouthPolarStereo",
        ]
        | None = None,
        cmap: str | LinearSegmentedColormap | ListedColormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        units: str | None = None,
        levels: int | list[float] | tuple[float, ...] | None = None,
        extend: str | None = None,
        robust: bool = False,
        symmetrical: bool = False,
        rasterized: bool = False,
        title: str | None = None,
        cbar_orientation: Literal["vertical", "horizontal"] = "vertical",
        add_colorbar: bool = True,
        cbar_drawedges: bool = True,
        cbar_label: str | None = None,
        cbar_minimal_ticks: bool = True,
        map_global_extent: bool = False,
        set_map_extent: tuple[float, float, float, float] | None = None,
        map_ticks: bool = False,
        map_xtick_bins: int = 5,
        map_ytick_bins: int = 5,
        add_grid_bounds: bool = False,
        add_coastlines: bool = True,
        add_borders: bool = True,
        add_states: bool = True,
        add_ocean: bool = True,
        add_land: bool = True,
        add_lakes: bool = False,
        add_rivers: bool = False,
        u_component: xr.DataArray | None = None,
        v_component: xr.DataArray | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        quiver_kwargs: dict[str, Any] | None = None,
        clabel: bool = False,
        clabel_fmt: str = "%1.0f",
        clabel_fontsize: float = 8,
        clabel_inline: bool = True,
        clabel_colors: str | None = None,
        clabel_kwargs: dict[str, Any] | None = None,
        cyclic: bool = False,
        indices: tuple[int, ...] | list[int] | np.ndarray[Any, Any] | None = None,
        outfile: Path | str | None = None,
        quality: Literal["low", "medium", "high"] = "medium",
        fps: int = 1,
        parallel: bool = True,
        frame_id: bool = True,
        **kwargs: Any,
    ) -> DisplayHandle | None:
        """Render the bound DataArray as an MP4 map animation.

        Parameters
        ----------
        dim : str, default "time"
            Animation dimension.
        x, y, col, row : str, optional
            Coordinate names used for plot and faceting.
        col_wrap : int, optional
            Number of columns for wrapped facets.
        figsize : tuple[float, float], optional
            Figure size in inches.
        sharex, sharey : bool, default True
            Share horizontal and vertical axis limits across facet panels in each
            frame. Ignored for non-faceted plots.
        method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
            Xarray plot method.
        projection : str, optional
            Cartopy projection name.
        cmap : str or matplotlib colormap, optional
            Colormap for the scalar field.
        norm : matplotlib.colors.Normalize, optional
            Color normalization.
        vmin, vmax : float, optional
            Scalar color limits.
        units : str, optional
            Units used for colorbar labeling.
        levels : int or sequence of float, optional
            Contour levels.
        extend : str, optional
            Colorbar extension mode.
        robust, rasterized : bool, default False
            Enable percentile scaling or rasterized artists.
        symmetrical : bool, default False
            Use symmetrical color limits around zero where supported.
        title : str, optional
            Base frame title.
        cbar_orientation : {"vertical", "horizontal"}, default "vertical"
            Colorbar orientation.
        add_colorbar, cbar_drawedges : bool
            Control colorbar creation and interval edges.
        cbar_label : str, optional
            Explicit colorbar label.
        cbar_minimal_ticks : bool, default True
            If True, reduce the number of ticks skipping every other one when possible.
        map_global_extent, add_grid_bounds : bool
            Control geographic extent and grid-boundary annotations.
        map_ticks : bool, default False
            Draw longitude and latitude ticks.
        map_xtick_bins : int, default 5
            Maximum number of longitude tick intervals.
        map_ytick_bins : int, default 5
            Maximum number of latitude tick intervals.
        set_map_extent : tuple[float, float, float, float], optional
            ``(lon_min, lon_max, lat_min, lat_max)``.
        add_coastlines, add_borders, add_states, add_ocean, add_land, add_lakes, add_rivers : bool
            Toggle Cartopy geographic features.
        u_component, v_component : xarray.DataArray, optional
            Vector components for a quiver overlay.
        cbar_kwargs, quiver_kwargs, clabel_kwargs : dict, optional
            Overlay and labeling options.
        clabel : bool, default False
            Label contour lines.
        clabel_fmt, clabel_colors : str, optional
            Contour-label format and color.
        clabel_fontsize : float, default 8
            Contour-label font size.
        clabel_inline : bool, default True
            Draw contour labels inline.
        cyclic : bool, default False
            Append a cyclic longitude point to each frame.
        indices : sequence of int or numpy.ndarray, optional
            Positional frame indices along ``dim``.
        outfile : str or pathlib.Path, optional
            MP4 output path.
        quality : {"low", "medium", "high"}, default "medium"
            Frame-resolution preset.
        fps : int, default 1
            Frames per second.
        parallel : bool, default True
            Render frames with multiprocessing.
        frame_id : bool, default True
            Include the frame identifier in titles.
        **kwargs : Any
            Additional xarray plot arguments.

        Returns
        -------
        IPython.display.DisplayHandle or None
            Notebook display handle when available.
        """

        opts = xgeo_core_utils.exclude_key("self", dict(locals()))
        kwargs = opts.pop("kwargs")

        return plot.animate(self._obj, **opts, **kwargs)

    def quiver(
        self,
        v: xr.DataArray,
        *,
        x: str = "lon",
        y: str = "lat",
        ax: Any = None,
        subsample: int | tuple[int, int] = (1, 1),
        add_key: bool = True,
        key_magnitude: float | None = None,
        key_units: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        """Draw quiver arrows, using the bound array as the zonal component.

        The reference key is placed automatically below the axis decorations
        (see :class:`xgeo.viz.plot_utils.AutoQuiverKey`).

        Parameters
        ----------
        v : xarray.DataArray
            Meridional component.
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on; the current axis when omitted.
        subsample : int or tuple of int, default (1, 1)
            Grid stride used to thin the arrows.
        add_key : bool, default True
            Draw a reference quiver key.
        key_magnitude : int or float, optional
            Reference arrow magnitude.
        key_units : str, optional
            Units shown on the key.
        **kwargs : Any
            Additional arguments forwarded to ``Axes.quiver``.

        Returns
        -------
        tuple
            ``(ax, quiver, quiver_key)``.

        """
        import matplotlib.pyplot as plt

        from ..viz.plot_utils import plot_quiver

        axis = plt.gca() if ax is None else ax
        quiver, quiver_key = plot_quiver(
            self._obj,
            v,
            axis.get_figure(),
            axis,
            x=x,
            y=y,
            subsample=subsample,
            add_key=add_key,
            key_magnitude=key_magnitude,
            key_units=key_units,
            **kwargs,
        )
        return axis, quiver, quiver_key

    def significance(
        self,
        *,
        x: str = "lon",
        y: str = "lat",
        ax: Any = None,
        level: float = 0.05,
        color: str = "grey",
        alpha: float = 0.3,
        marker: str | None = None,
        edgecolors: str | None = None,
        subsample: int | tuple[int, int] = (1, 1),
        size: float = 0.25,
    ) -> PathCollection:
        """Mark grid points of the bound p-value field below ``level``.

        Parameters
        ----------
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on.
        level : float, default 0.05
            Significance threshold.
        color : str, default "grey"
            Marker face color.
        alpha : float, default 0.3
            Marker opacity.
        marker : str, optional
            Marker style.
        edgecolors : str, optional
            Marker edge color.
        subsample : int or tuple of int, default (1, 1)
            Grid stride used to thin the markers.
        size : float, default 0.25
            Marker size.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter artist holding the markers.

        """

        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))

        return plot.plot_significance(self._obj, **kwargs)

    def corr(
        self,
        other: xr.DataArray,
        *,
        dim: str = "time",
        corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        """Correlate the bound array with ``other`` along ``dim``.

        Parameters
        ----------
        other : xarray.DataArray
            Second field, matching the bound array in dimensions and shape.
        dim : str, default "time"
            Dimension the correlation is computed along.
        corr_type : {"pearson", "spearman", "kendall"}, default "pearson"
            Correlation coefficient.
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            Alternative hypothesis used for the p-value.
        dask_scheduler : {"threads", "processes"}, default "threads"
            Scheduler used to evaluate a chunked input.

        Returns
        -------
        xarray.Dataset
            Dataset holding ``corr`` and ``p_value``.

        """

        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        kwargs["y"] = kwargs.pop("other")
        return xgeo_core_calc.corr(self._obj, kwargs)

    def pvalues(self, other: xr.DataArray, dim: str = "time") -> xr.DataArray:
        """Test the difference in mean between the bound array and ``other``.

        Parameters
        ----------
        other : xarray.DataArray
            Second sample, for example a second period.
        dim : str, default "time"
            Sample dimension.

        Returns
        -------
        xarray.DataArray
            Pointwise p-values of a Welch t-test.

        """

        return xgeo_core_calc.pvalues(self._obj, other, dim=dim)

    def trends(
        self,
        dim: str = "time",
        *,
        scale: float = 1,
        dask_scheduler: Literal["threads", "processes"] = "threads",
        polyfit: bool = False,
    ) -> xr.Dataset:
        """Compute a pointwise trend along ``dim``.

        Parameters
        ----------
        dim : str, default "time"
            Dimension the trend is computed along.
        scale : float, default 1
            Multiplier applied to the slope, to convert its time unit.
        dask_scheduler : {"threads", "processes"}, default "threads"
            Scheduler used to evaluate a chunked input.
        polyfit : bool, default False
            Use ordinary least squares instead of the modified Mann-Kendall test.

        Returns
        -------
        xarray.Dataset
            Trend statistics, including the slope and its p-value.

        """

        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        return xgeo_core_calc.trends(self._obj, **kwargs)

    def fillgaps(
        self,
        y: str = "lat",
        x: str = "lon",
        method: Literal["linear", "cubic", "nearest"] = "linear",
        *,
        max_cells: int = 5,
        max_iter: int = 5,
        nan_mask: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """
        Fill short, bounded NaN gaps along two dimensions.

        A NaN cell is eligible for interpolation when it belongs to a
        contiguous NaN run no longer than ``max_cells`` and that run is
        bounded by finite values on both sides along either ``x`` or ``y``.
        Multiple passes allow intersections between horizontal and vertical
        gaps to be resolved after neighboring cells have been filled.

        Parameters
        ----------
        y : str, default="lat"
            Name of the first interpolation dimension.
        x : str, default="lon"
            Name of the second interpolation dimension.
        method : {"linear", "cubic", "nearest"}, default="linear"
            Interpolation method passed to :func:`scipy.interpolate.griddata`.
        max_cells : int, default=5
            Maximum contiguous NaN run length, in grid cells, eligible for
            interpolation.
        max_iter : int, default=5
            Maximum number of interpolation passes.
        nan_mask : xarray.DataArray, optional
            Boolean mask identifying cells that must remain NaN. The mask may
            contain only ``(y, x)`` dimensions or additional dimensions
            broadcastable against ``da``.


        Returns
        -------
        xarray.DataArray
            DataArray with eligible NaN gaps interpolated. Dimensions,
            coordinates, name, and attributes are preserved.

        """
        kwargs = xgeo_core_utils.exclude_key("self", dict(locals()))
        return xgeo_xarray_utils.fillgaps(self._obj, **kwargs)


class PreprocessAccessor:
    """Dataset-specific preprocessing namespace."""

    __slots__ = ("_obj",)

    def __init__(self, ds: xr.Dataset) -> None:
        """Initialize the preprocessing accessor."""
        self._obj = ds

    def era5(self) -> xr.Dataset:
        """Preprocess an ERA5 dataset to standardize variable names, dimensions, and attributes.

        Returns
        -------
        xr.Dataset
            Preprocessed ERA5 dataset.

        """
        return xgeo_core_preprocess.era5(self._obj)

    def era5_land(self) -> xr.Dataset:
        """Preprocess an ERA5-Land dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed ERA5-Land dataset.

        """
        return xgeo_core_preprocess.era5_land(self._obj)

    def imerg(self) -> xr.Dataset:
        """Preprocess a GPM IMERG dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed IMERG dataset.

        """
        return xgeo_core_preprocess.imerg(self._obj)

    def cmorph(self) -> xr.Dataset:
        """Preprocess a CMORPH dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed CMORPH dataset.

        """
        return xgeo_core_preprocess.cmorph(self._obj)

    def gpcp(self) -> xr.Dataset:
        """Preprocess a GPCP dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed GPCP dataset.

        """
        return xgeo_core_preprocess.gpcp(self._obj)


@xr.register_dataset_accessor("xgeo")
class GeoDataset(GeoBase):
    """Dataset accessor extending the shared geographic operations."""

    __slots__ = ()

    @property
    def preprocess(self) -> PreprocessAccessor:
        """Return the preprocessing namespace.

        Returns
        -------
        PreprocessAccessor
            Preprocessing accessor.

        """
        return PreprocessAccessor(self._obj)


def fix_xarray(*, force: bool = False) -> tuple[Path, ...]:
    """Patch xarray source so IDEs resolve registered accessors for completion.

    Parameters
    ----------
    force : bool
        Whether to rebuild an existing source patch.

    Returns
    -------
    tuple[Path, ...]
        Paths modified by the xarray source patch.

    """

    from importlib.util import find_spec

    # Fast path: do not import xarray or integrations if already patched.
    xarray_spec = find_spec("xarray")
    if xarray_spec is None or xarray_spec.origin is None:
        raise RuntimeError("Cannot locate the xarray package.")

    marker = Path(xarray_spec.origin).resolve().parent / ".xgeo"
    xarray_init = Path(xarray_spec.origin).resolve()

    if not force and marker.exists():
        return ()

    # Everything below is only needed when creating/rebuilding the patch.
    import ast
    import importlib
    import inspect
    import json
    import os
    import sys

    import xarray as xr

    if not __package__:
        raise RuntimeError("The accessor module must be imported as part of a package.")

    begin = "XGEO_IDE_TYPING BEGIN"
    end = "XGEO_IDE_TYPING END"

    bridge_path = Path(__file__).resolve().parent / "xr_init.py"
    type_module = f"{__package__}.xr_init"

    bridge = (
        "from __future__ import annotations\n"
        "\n"
        "from .accessors import GeoDataArray as GeoDataArray\n"
        "from .accessors import GeoDataset as GeoDataset\n"
        "\n"
        '__all__ = ["GeoDataArray", "GeoDataset"]\n'
    )

    # Verify generated bridge before writing it.
    compile(bridge, str(bridge_path), "exec")

    bridge_changed = (
        not bridge_path.exists() or bridge_path.read_text(encoding="utf-8") != bridge
    )

    if bridge_changed:
        bridge_path.write_text(bridge, encoding="utf-8")

    # (xarray class, class name, local accessor definitions)
    targets: tuple[tuple[type, str, tuple[tuple[str, str], ...]], ...] = (
        (xr.DataArray, "DataArray", (("xgeo", "GeoDataArray"),)),
        (xr.Dataset, "Dataset", (("xgeo", "GeoDataset"),)),
    )

    optional: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("metpy.xarray", ("metpy",)),
        ("cf_xarray", ("cf",)),
        ("pint_xarray", ("pint",)),
        ("rioxarray", ("rio",)),
    )

    integration_names = tuple(module.partition(".")[0] for module, _ in optional)

    sources: dict[str, Path] = {}

    for cls, class_name, _ in targets:
        source_file = inspect.getsourcefile(cls)

        if source_file is None:
            raise RuntimeError(f"Cannot locate xarray.{class_name} source.")

        sources[class_name] = Path(source_file).resolve()

    # force=True restores pristine source before rebuilding the patch.
    if force:
        for path in set(sources.values()):
            backup = path.with_suffix(path.suffix + ".xgeo.bak")

            if backup.exists():
                path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
                backup.unlink()

    def stat_of(path: str | Path | None) -> list[int] | None:
        """Return an immutable file-stat signature."""
        if path is None:
            return None

        try:
            stat = os.stat(path)
        except OSError:
            return None

        return [stat.st_size, stat.st_mtime_ns]

    def signature() -> dict[str, Any]:
        """Return a stable signature for a source file."""
        integrations: dict[str, dict[str, Any] | None] = {}

        for name in integration_names:
            try:
                spec = find_spec(name)
            except ImportError, ValueError:
                spec = None

            if spec is None:
                integrations[name] = None
                continue

            integrations[name] = {"origin": spec.origin, "stat": stat_of(spec.origin)}

        return {
            "schema": 4,
            "python": (f"{sys.version_info.major}.{sys.version_info.minor}"),
            "files": {label: stat_of(path) for label, path in sources.items()},
            "integrations": integrations,
        }

    def discover() -> dict[type, list[tuple[str, str, str]]]:
        """Discover accessor registrations in a source file."""
        found: dict[type, list[tuple[str, str, str]]] = {
            cls: [] for cls, _, _ in targets
        }

        for module_name, names in optional:
            top = module_name.partition(".")[0]

            try:
                importlib.import_module(module_name)

            except ModuleNotFoundError as exc:
                missing = exc.name or ""

                if missing in {top, module_name} or module_name.startswith(
                    missing + "."
                ):
                    continue

                raise RuntimeError(
                    f"{module_name!r} missing dependency {missing!r}."
                ) from exc

            for name in names:
                registered = False

                for cls, class_name, _ in targets:
                    accessor = getattr(cls, name, None)

                    if accessor is None:
                        continue

                    registered = True

                    if not inspect.isclass(accessor):
                        raise RuntimeError(
                            f"{class_name}.{name} must be an accessor class."
                        )

                    if accessor.__qualname__ != accessor.__name__:
                        raise RuntimeError(
                            f"Nested accessor {class_name}.{name} is unsupported."
                        )

                    found[cls].append((name, accessor.__module__, accessor.__name__))

                if not registered:
                    raise RuntimeError(f"{module_name!r} did not register {name!r}.")

        return found

    def strip(source: str) -> str:
        """Remove previously injected source regions."""
        output: list[str] = []
        skipping = False

        for line in source.splitlines(keepends=True):
            if begin in line:
                skipping = True
                continue

            if end in line:
                skipping = False
                continue

            if not skipping:
                output.append(line)

        return "".join(output)

    def region(tag: str, indent: str, body: list[str]) -> str:
        """Wrap generated source with patch markers."""
        return f"{indent}# {begin} {tag}\n" + "".join(body) + f"{indent}# {end} {tag}\n"

    def build(class_name: str, stubs: list[tuple[str, str, str]]) -> tuple[str, str]:
        """Build the accessor bridge source."""
        aliases = {attr: f"_xgeo_{class_name}_{attr}" for attr, _, _ in stubs}

        imports = [
            "from typing import TYPE_CHECKING\n",
            "if TYPE_CHECKING:\n",
        ]

        for attr, module, name in stubs:
            imports.append(f"    from {module} import {name} as {aliases[attr]}\n")

        properties = [
            "    if TYPE_CHECKING:\n",
        ]

        for attr, _, _ in stubs:
            properties.append("        @property\n")
            properties.append(f"        def {attr}(self) -> {aliases[attr]}: ...\n")

        return (
            region(f"imports {class_name}", "", imports),
            region(f"properties {class_name}", "    ", properties),
        )

    discovered = discover()
    changed: list[Path] = []

    if bridge_changed:
        changed.append(bridge_path)

    for cls, class_name, own_accessors in targets:
        path = sources[class_name]

        backup = path.with_suffix(path.suffix + ".xgeo.bak")

        source_path = backup if backup.exists() else path

        raw = source_path.read_text(encoding="utf-8")

        pristine = strip(raw)

        if not backup.exists():
            backup.write_text(pristine, encoding="utf-8")

        stubs: list[tuple[str, str, str]] = [
            *((attr, type_module, type_name) for attr, type_name in own_accessors),
            *discovered[cls],
        ]

        for attr, _, _ in stubs:
            if not attr.isidentifier():
                raise RuntimeError(f"Accessor name {attr!r} is not a valid identifier.")

        import_region, property_region = build(class_name, stubs)

        tree = ast.parse(pristine, filename=str(path))

        node = next(
            (
                item
                for item in tree.body
                if (isinstance(item, ast.ClassDef) and item.name == class_name)
            ),
            None,
        )

        if node is None:
            raise RuntimeError(f"Cannot find xarray.{class_name} class definition.")

        head = node.body[0]

        is_docstring = (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
        )

        property_at = head.end_lineno if is_docstring else head.lineno - 1

        import_at = node.lineno - 1

        lines = pristine.splitlines(keepends=True)

        # Insert properties first because adding the module-level import
        # region afterward shifts the entire class downward.
        lines.insert(property_at, property_region)
        lines.insert(import_at, import_region)

        patched = "".join(lines)

        # Never write invalid Python into the installed xarray source.
        compile(patched, str(path), "exec")

        if patched != path.read_text(encoding="utf-8"):
            path.write_text(patched, encoding="utf-8")
            changed.append(path)

    # Make xarray import register the xgeo accessors at runtime.
    raw = xarray_init.read_text(encoding="utf-8")
    pristine = strip(raw)

    runtime_region = region(
        "runtime registration",
        "",
        [f"import {__package__}.accessors as _xgeo_accessors\n"],
    )

    patched = pristine.rstrip() + "\n\n" + runtime_region

    compile(patched, str(xarray_init), "exec")

    if patched != raw:
        xarray_init.write_text(patched, encoding="utf-8")
        changed.append(xarray_init)

    # The marker is created only after every source modification succeeds.
    TMP = marker.with_suffix(marker.suffix + ".TMP")

    TMP.write_text(json.dumps(signature(), sort_keys=True), encoding="utf-8")
    TMP.replace(marker)

    if changed:
        print("Updated xarray source for IDE typing:")
        print("\n".join(f"  {path}" for path in changed))
