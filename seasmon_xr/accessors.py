"""Xarray Accesor classes."""
from typing import List, Optional, Union

from dask import is_dask_collection
import dask.array as da
from dask.base import tokenize
import numpy as np
import pandas as pd
import xarray

from . import ops

__all__ = [
    "Anomalies",
    "Dekad",
    "IterativeAggregation",
    "Pentad",
    "PixelAlgorithms",
    "WhittakerSmoother",
]


class MissingTimeError(Exception):
    """Exception for missing time dimension when required."""


class AccessorBase:
    """Base class for accessors."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _check_for_timedim(self):
        if "time" not in self._obj.dims:
            return False
        return True


class AccessorTimeBase(AccessorBase):
    """Base class for accessors with required time dimension."""

    def __init__(self, xarray_obj):
        """Construct with DataArray|Dataset."""
        if not np.issubdtype(xarray_obj, np.datetime64):
            raise TypeError(
                "'This accessor is only available for "
                "DataArray with datetime64 dtype"
            )

        if not hasattr(xarray_obj, "time"):
            raise ValueError("Data array is missing 'time' accessor!")

        if "time" not in xarray_obj.dims:
            xarray_obj = xarray_obj.expand_dims("time")
        self._obj = xarray_obj

        super().__init__(xarray_obj)

    @property
    def year(self):
        return self._obj.time.dt.year

    @property
    def month(self):
        return self._obj.time.dt.month

    @property
    def day(self):
        return self._obj.time.dt.day


class Period(AccessorTimeBase):
    # pylint: disable=no-member,undefined-variable
    """
    Baseclass to extend time dimension with period functionality.

    Adds functionality for working with periods, such as dekads and pentads
    """

    @property
    def midx(self):
        return (
            self._obj.time.to_series()
            .apply(lambda x: min(self.max_per_month, ((x.day - 1) // self.ndays) + 1))
            .to_xarray()
        )

    @property
    def yidx(self):
        return (
            self._obj.time.to_series()
            .apply(lambda x: ((x.month - 1) * self.max_per_month))
            .to_xarray()
            + self.midx
        )

    @property
    def label(self):
        return (
            self.year.astype("str")
            .str.cat(self.month.astype("str").str.zfill(2))
            .str.cat(self.midx.astype("str"), sep=self._label)
        )


@xarray.register_dataset_accessor("dekad")
@xarray.register_dataarray_accessor("dekad")
class Dekad(Period):
    """Accessor class for dekad period."""

    ndays = 10
    max_per_month = 3
    _label = "d"


@xarray.register_dataset_accessor("pentad")
@xarray.register_dataarray_accessor("pentad")
class Pentad(Period):
    """Accessor class for pentad period."""

    ndays = 5
    max_per_month = 6
    _label = "p"


class IterativeAggregation(AccessorBase):
    """Class to aggregate multiple coordinate slices."""

    def sum(
        self,
        n: int = None,
        dim: str = "time",
        begin: Union[str, int, float] = None,
        end: Union[str, int, float] = None,
        method: str = None,
    ):
        """Generate sum-aggregations over dim for periods n."""
        yield from self._iteragg(np.nansum, n, dim, begin, end, method)

    def mean(
        self,
        n: int = None,
        dim: str = "time",
        begin: Union[str, int, float] = None,
        end: Union[str, int, float] = None,
        method: str = None,
    ):
        """Generate mean-aggregations over dim for slices of n."""
        yield from self._iteragg(np.nanmean, n, dim, begin, end, method)

    def full(
        self,
        n: int = None,
        dim: str = "time",
        begin: Union[str, int, float] = None,
        end: Union[str, int, float] = None,
        method: str = None,
    ):
        """Generate mean-aggregations over dim for slices of n."""
        yield from self._iteragg(None, n, dim, begin, end, method)

    def _iteragg(self, func, n, dim, begin, end, method):

        if dim not in self._obj.dims:
            raise ValueError(f"Dimension {dim} doesn't exist in xarray object!")

        _index = self._obj[dim].to_index()

        if n is None:
            n = self._obj[dim].size
        assert n != 0, "n must be non-zero"

        if begin is not None:
            try:
                begin_ix = _index.get_loc(begin, method=method) + 1
            except KeyError:
                raise ValueError(
                    f"Value {begin} for 'begin' not found in index for dim {dim}"
                ) from None
        else:
            begin_ix = self._obj.sizes[dim]

        if end is not None:
            try:
                end_ix = _index.get_loc(end, method=method)
            except KeyError:
                raise ValueError(
                    f"Value {end} for 'end' not found in index for dim {dim}"
                ) from None
        else:
            end_ix = 0

        for ii in range(begin_ix, 0, -1):
            jj = ii - n
            if ii <= end_ix:
                break
            if jj >= 0 and (ii - jj) == n:
                region = {dim: slice(jj, ii)}
                _obj = self._obj[region].assign_attrs(
                    {
                        "agg_start": str(_index[jj]),
                        "agg_stop": str(_index[ii - 1]),
                        "agg_n": _index[jj:ii].size,
                    }
                )

                if func is not None:
                    _obj = _obj.reduce(func, dim, keep_attrs=True)
                    if dim == "time":
                        _obj = _obj.expand_dims(time=[self._obj.time[ii - 1].values])

                yield _obj


class Anomalies(AccessorBase):
    """Class to calculate anomalies from reference."""

    def ratio(self, reference, offset=0):
        """Calculate anomaly as ratio."""
        return (self._obj + offset) / (reference + offset) * 100

    def diff(self, reference, offset=0):
        """Calculate anomaly as difference."""
        return (self._obj + offset) - (reference + offset)


class WhittakerSmoother(AccessorBase):
    """Class for applying different version of the Whittaker smoother."""

    def whits(
        self,
        nodata: Union[int, float],
        sg: xarray.DataArray = None,
        s: float = None,
        p: float = None,
    ) -> xarray.Dataset:
        """
        Apply whittaker with fixed S.

        Fixed S can be either provided as constant or
        as sgrid with a constant per pixel

        Args:
            ds: input dataset,
            nodata: nodata value
            sg: sgrid,
            s: S value
            p: Envelope value for asymmetric weights

        Returns:
            ds_out: xarray.Dataset with smoothed data
        """
        if not self._check_for_timedim():
            raise MissingTimeError("Whittaker filter requires a time dimension!")
        if sg is None and s is None:
            raise ValueError("Need S or sgrid")

        lmda = 10**sg if sg is not None else s

        if p is not None:

            xout = xarray.apply_ufunc(
                ops.ws2dpgu,
                self._obj,
                lmda,
                nodata,
                p,
                input_core_dims=[["time"], [], [], []],
                output_core_dims=[["time"]],
                dask="parallelized",
                keep_attrs=True,
            )

        else:

            xout = xarray.apply_ufunc(
                ops.ws2dgu,
                self._obj,
                lmda,
                nodata,
                input_core_dims=[["time"], [], []],
                output_core_dims=[["time"]],
                dask="parallelized",
                keep_attrs=True,
            )

        return xout

    def whitsvc(
        self,
        nodata: Union[int, float],
        lc: xarray.DataArray = None,
        srange: np.ndarray = None,
        p: float = None,
    ) -> xarray.Dataset:
        """
        Apply whittaker with V-curve optimization of S.

        Args:
            dim: dimension to use for filtering
            nodata: nodata value
            lc: lag1 autocorrelation DataArray,
            srange: values of S for V-curve optimization (mandatory if no autocorrelation raster)
            p: Envelope value for asymmetric weights

        Returns:
            ds_out: xarray.Dataset with smoothed data and sgrid
        """
        if not self._check_for_timedim():
            raise MissingTimeError("Whittaker filter requires a time dimension!")

        if lc is not None:
            if p is None:
                raise ValueError(
                    "If lc is set, a p value needs to be specified as well."
                )

            ds_out, sgrid = xarray.apply_ufunc(
                ops.ws2doptvplc,
                self._obj,
                nodata,
                p,
                lc,
                input_core_dims=[["time"], [], [], []],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

        else:

            if srange is None:
                raise ValueError("Need either lagcorr or srange!")

            if p:
                ds_out, sgrid = xarray.apply_ufunc(
                    ops.ws2doptvp,
                    self._obj,
                    nodata,
                    p,
                    srange,
                    input_core_dims=[["time"], [], [], ["dim0"]],
                    output_core_dims=[["time"], []],
                    dask="parallelized",
                    keep_attrs=True,
                )

            else:

                ds_out, sgrid = xarray.apply_ufunc(
                    ops.ws2doptv,
                    self._obj,
                    nodata,
                    srange,
                    input_core_dims=[["time"], [], ["dim0"]],
                    output_core_dims=[["time"], []],
                    dask="parallelized",
                    keep_attrs=True,
                )

        ds_out = ds_out.to_dataset(name=(ds_out.name or "band"))
        ds_out["sgrid"] = np.log10(sgrid).astype("float32")

        return ds_out

    def whitswcv(
        self,
        nodata: Union[int, float],
        srange: np.ndarray = None,
        p: float = None,
        robust: bool = True,
    ) -> xarray.Dataset:
        """
        Apply whittaker with GCV optimization of S.

        Args:
            dim: dimension to use for filtering
            nodata: nodata value
            srange: values of S for GCV optimization
            p: Envelope value for asymmetric weights
            robust (boolean): performs a robust fitting by computing robust weights if True

        Returns:
            ds_out: xarray.Dataset with smoothed data and sgrid
        """
        if p:

            if srange is None:
                srange = np.arange(-1.8, 4.2, 0.2, dtype=np.float64)

            ds_out, sgrid = xarray.apply_ufunc(
                ops.ws2dwcvp,
                self._obj,
                nodata,
                p,
                srange,
                robust,
                input_core_dims=[["time"], [], [], ["dim0"], []],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

        else:
            if srange is None:
                srange = np.arange(-1.8, 4.2, 0.2)

            ds_out, sgrid = xarray.apply_ufunc(
                ops.ws2dwcv,
                self._obj,
                nodata,
                srange,
                robust,
                input_core_dims=[["time"], [], ["dim0"], []],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

        ds_out = ds_out.to_dataset(name=(ds_out.name or "band"))
        ds_out["sgrid"] = np.log10(sgrid).astype("float32")

        return ds_out

    def whitint(self, labels_daily: np.ndarray, template: np.ndarray):
        """Compute temporal interpolation using the Whittaker filter."""
        if not self._check_for_timedim():
            raise MissingTimeError(
                "Whittaker temporal interpolation requires a time dimension!"
            )

        if self._obj.dtype != "int16":
            raise NotImplementedError(
                "Temporal interpolation works currently only with int16 input!"
            )

        template_out = np.zeros(np.unique(labels_daily).size, dtype="u1")

        ds_out = xarray.apply_ufunc(
            ops.tinterpolate,
            self._obj,
            template,
            labels_daily,
            template_out,
            input_core_dims=[["time"], ["dim0"], ["dim1"], ["dim2"]],
            output_core_dims=[["newtime"]],
            dask_gufunc_kwargs={"output_sizes": {"newtime": template_out.size}},
            output_dtypes=["int16"],
            dask="parallelized",
            keep_attrs=True,
        )

        return ds_out


class PixelAlgorithms(AccessorBase):
    """Set of algorithms to be applied to pixel timeseries."""

    def spi(
        self,
        calibration_start=None,
        calibration_stop=None,
    ):
        """Calculate the SPI along the time dimension."""
        if not self._check_for_timedim():
            raise MissingTimeError("SPI requires a time dimension!")

        from .ops.spi import spifun  # pylint: disable=import-outside-toplevel

        tix = self._obj.get_index("time")

        calstart_ix = 0
        if calibration_start is not None:
            calstart = pd.Timestamp(calibration_start)
            if calstart > tix[-1]:
                raise ValueError(
                    "Calibration start cannot be greater than last timestamp!"
                )
            calstart_ix = tix.get_loc(calstart, method="bfill")

        calstop_ix = tix.size
        if calibration_stop is not None:
            calstop = pd.Timestamp(calibration_stop)
            if calstop < tix[0]:
                raise ValueError(
                    "Calibration stop cannot be smaller than first timestamp!"
                )
            calstop_ix = tix.get_loc(calstop, method="ffill") + 1

        if calstart_ix >= calstop_ix:
            raise ValueError("calibration_start < calibration_stop!")

        if abs(calstop_ix - calstart_ix) <= 1:
            raise ValueError(
                "Timeseries too short for calculating SPI. Please adjust calibration period!"
            )

        res = xarray.apply_ufunc(
            spifun,
            self._obj,
            kwargs={
                "cal_start": calstart_ix,
                "cal_stop": calstop_ix,
            },
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=["int16"],
        )

        res.attrs.update(
            {
                "spi_calibration_start": str(tix[calstart_ix].date()),
                "spi_calibration_stop": str(tix[calstop_ix - 1].date()),
            }
        )

        return res

    def croo(self):
        """Compute current run of ones along time dimension."""
        if not self._check_for_timedim():
            raise MissingTimeError("CROO requires a time dimension!")

        xsort = self._obj.sortby("time", ascending=False)
        xtemp = xsort.where(xsort == 1).cumsum("time", skipna=False)
        xtemp = xtemp.where(~xtemp.isnull(), 0).argmax("time")
        x_crbt = xtemp + xsort.isel(time=0)

        return x_crbt

    def lroo(self):
        """Longest run of ones along time dimension."""
        if not self._check_for_timedim():
            raise MissingTimeError("LROO requires a time dimension!")

        return xarray.apply_ufunc(
            ops.lroo,
            self._obj,
            input_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=["uint8"],
        )

    def autocorr(self):
        """
        Calculate the autocorrelation along time.

        Returns:
            xarray.DataArray with lag1 autocorrelation
        """
        xx = self._obj
        if xx.dims[0] == "time":
            # I don't know how to tell xarray's map_blocks about
            # changing dtype and losing first dimension, so use
            # dask version directly
            if is_dask_collection(xx):
                # merge all time slices if not already
                if len(xx.chunks[0]) != 1:
                    xx = xx.chunk({"time": -1})

                data = da.map_blocks(
                    ops.autocorr_tyx, xx.data, dtype="float32", drop_axis=0
                )
            else:
                data = ops.autocorr_tyx(xx.data)

            coords = {k: c for k, c in xx.coords.items() if k != "time"}
            return xarray.DataArray(data=data, dims=xx.dims[1:], coords=coords)

        return xarray.apply_ufunc(
            ops.autocorr,
            xx,
            input_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=["float32"],
        )


class ZonalStatistics(AccessorBase):
    """Class to claculate zonal statistics."""

    def mean(
        self,
        zones: xarray.DataArray,
        zone_ids: Union[List, np.ndarray],
        dtype: str = "float64",
        dim_name: str = "zones",
        name: Optional[str] = None,
    ) -> xarray.DataArray:
        """Calculate the zonal mean.

        The mean for each pixel is calculated for each zone (group) in
        `zones`.

        The zones in the `zones` raster have to be numbered in a linear fashion,
        starting with 0 for the first zone and num_zones for the last zone.

        Args:
            zones: zonal pixels (Y,X)
            zone_ids: list or array with zone IDs (from 0 to (n-1))
            dtype: datatype
            dim_name: name for new output dimension
            name: name for output dataarray
        """
        from .ops.zonal import do_mean  # pylint: disable=import-outside-toplevel

        xx = self._obj

        if isinstance(xx, xarray.Dataset):
            raise NotImplementedError("zonal needs dataarray as input")

        if "nodata" not in xx.attrs:
            raise ValueError("Input xarray DataArray needs nodata attribute")

        if not isinstance(zones, xarray.DataArray):
            raise ValueError("Zones need to be xarray.DataArray!")

        if "nodata" not in zones.attrs:
            raise ValueError("Zones xarray DataArray needs nodata attribute")

        # set null values to nodata value
        xx = xx.where(xx.notnull(), xx.nodata)

        num_zones = len(zone_ids)
        dims = (xx.dims[0], dim_name, "stat")
        coords = {
            dims[0]: xx.coords[dims[0]],
            dim_name: zone_ids,
            "stat": ["mean", "valid"],
        }

        if is_dask_collection(xx):
            dask_name = name
            if isinstance(dask_name, str):
                dask_name = f"{name}-{tokenize(xx.data, zones.data, dtype)}"

            chunks = [xx.data.chunks[0], (num_zones,), (2,)]

            data = da.map_blocks(
                do_mean,
                xx.data,
                zones.data,
                num_zones,
                xx.nodata,
                zones.nodata,
                drop_axis=[1, 2],
                new_axis=[1, 2],
                chunks=chunks,
                dtype=dtype,
                name=dask_name,
            )
        else:

            data = do_mean(
                xx.data,
                zones.data,
                num_zones,
                xx.nodata,
                zones.nodata,
            )

        return xarray.DataArray(
            data=data, dims=dims, coords=coords, attrs={}, name=name
        )


@xarray.register_dataset_accessor("hdc")
@xarray.register_dataarray_accessor("hdc")
class HDC:
    """xarray accessor for HDC xarray tools."""

    def __init__(self, xarray_obj):
        self.algo = PixelAlgorithms(xarray_obj)
        self.anom = Anomalies(xarray_obj)
        self.iteragg = IterativeAggregation(xarray_obj)
        self.whit = WhittakerSmoother(xarray_obj)
        self.zonal = ZonalStatistics(xarray_obj)
