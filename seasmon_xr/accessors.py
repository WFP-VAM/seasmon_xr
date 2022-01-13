"""Xarray Accesor classes."""
from typing import Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray
from dask import is_dask_collection

from . import ops

__all__ = [
    "Anomalies",
    "IterativeAggregation",
    "LabelMaker",
    "PixelAlgorithms",
    "WhittakerSmoother",
]


class AccessorBase:
    """Base class for accessors."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xarray.register_dataarray_accessor("labeler")
class LabelMaker:
    """
    Class to extending xarray.Dataarray for 'time'.

    Adds the properties labelling the time values as either
    dekads or pentads.
    """

    def __init__(self, xarray_obj):
        """Construct with DataArray|Dataset."""
        if not np.issubdtype(xarray_obj, np.datetime64):
            raise TypeError(
                "'.labeler' accessor only available for "
                "DataArray with datetime64 dtype"
            )

        if not hasattr(xarray_obj, "time"):
            raise ValueError("Data array is missing 'time' accessor!")

        if "time" not in xarray_obj.dims:
            xarray_obj = xarray_obj.expand_dims("time")
        self._obj = xarray_obj

    @property
    def dekad(self):
        """Time values labeled as dekads."""
        return (
            self._obj.time.to_series()
            .apply(
                func=self._gen_labels,
                args=("d", 10.5),
            )
            .values
        )

    @property
    def pentad(self):
        """Time values labeled as pentads."""
        return (
            self._obj.time.to_series()
            .apply(
                func=self._gen_labels,
                args=("p", 5.19),
            )
            .values
        )

    @staticmethod
    def _gen_labels(x, lbl, c):
        return f"{x.year}{x.month:02}" + f"{lbl}{int(x.day//c+1)}"


@xarray.register_dataset_accessor("iteragg")
@xarray.register_dataarray_accessor("iteragg")
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


@xarray.register_dataset_accessor("anom")
@xarray.register_dataarray_accessor("anom")
class Anomalies(AccessorBase):
    """Class to calculate anomalies from reference."""

    def ratio(self, reference, offset=0):
        """Calculate anomaly as ratio."""
        return (self._obj + offset) / (reference + offset) * 100

    def diff(self, reference, offset=0):
        """Calculate anomaly as difference."""
        return (self._obj + offset) - (reference + offset)


@xarray.register_dataset_accessor("whit")
@xarray.register_dataarray_accessor("whit")
class WhittakerSmoother:
    """Class for applying different version of the Whittaker smoother."""

    def __init__(self, xarray_obj):
        """Construct with DataArray|Dataset."""
        if "time" not in xarray_obj.dims:
            raise ValueError(
                "'.whit' can only be applied to datasets / dataarrays "
                "with 'time' dimension!"
            )
        self._obj = xarray_obj

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
        if sg is None and s is None:
            raise ValueError("Need S or sgrid")

        lmda = 10 ** sg if sg is not None else s

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

    def whitint(self, labels_daily: np.ndarray, template: np.ndarray):
        """Compute temporal interpolation using the Whittaker filter."""
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


@xarray.register_dataset_accessor("algo")
@xarray.register_dataarray_accessor("algo")
class PixelAlgorithms:
    """Set of algorithms to be applied to pixel timeseries."""

    def __init__(self, xarray_obj):
        """Construct with DataArray|Dataset."""
        if "time" not in xarray_obj.dims:
            raise ValueError(
                "'.algo' can only be applied to datasets / dataarrays "
                "with 'time' dimension!"
            )
        self._obj = xarray_obj

    def spi(
        self,
        calibration_start=None,
        calibration_stop=None,
    ):
        """Calculate the SPI along the time dimension."""
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
        xsort = self._obj.sortby("time", ascending=False)
        xtemp = xsort.where(xsort == 1).cumsum("time", skipna=False)
        xtemp = xtemp.where(~xtemp.isnull(), 0).argmax("time")
        x_crbt = xtemp + xsort.isel(time=0)

        return x_crbt

    def lroo(self):
        """Longest run of ones along time dimension."""
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
