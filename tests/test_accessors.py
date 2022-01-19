"""Tests for xarray accessors"""
# pylint: disable=redefined-outer-name,unused-import,no-member,no-name-in-module,missing-function-docstring
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import seasmon_xr
from seasmon_xr.src.ws2d import ws2d


@pytest.fixture
def darr():
    np.random.seed(42)
    x = xr.DataArray(
        np.random.randint(1, 100, (5, 2, 2)), dims=("time", "y", "x"), name="band"
    )
    x["time"] = pd.date_range(start="2000-01-01", periods=5, freq="10D")

    return x


@pytest.fixture
def res_spi():
    return np.array(
        [
            [[217, 397, 644, -1947, 847], [1401, -979, 979, -920, -508]],
            [[-1673, 943, 1061, 116, -424], [988, 1178, 103, -1139, -1139]],
        ]
    )


def test_labels_dekad(darr):

    np.testing.assert_array_equal(
        darr.time.labeler.dekad,
        ["200001d1", "200001d2", "200001d3", "200001d3", "200002d1"],
    )


def test_labels_dekad_single(darr):

    np.testing.assert_array_equal(darr.isel(time=0).time.labeler.dekad, ["200001d1"])


def test_labels_pentad(darr):

    np.testing.assert_array_equal(
        darr.time.labeler.pentad,
        ["200001p1", "200001p3", "200001p5", "200001p6", "200002p2"],
    )


def test_labels_pentad_single(darr):

    np.testing.assert_array_equal(darr.isel(time=0).time.labeler.pentad, ["200001p1"])


def test_labels_exception(darr):
    with pytest.raises(TypeError):
        _ = darr.x.labeler.dekad


def test_algo_lroo(darr):
    _res = np.array([[3, 0], [4, 2]], dtype="uint8")

    darr_lroo = ((darr > 30) * 1).astype("uint8").algo.lroo()
    assert isinstance(darr_lroo, xr.DataArray)
    np.testing.assert_array_equal(darr_lroo, _res)


def test_algo_croo(darr):
    _res = np.array([[1, 0], [4, 0]], dtype="uint8")

    darr_croo = ((darr > 30) * 1).algo.croo()
    assert isinstance(darr_croo, xr.DataArray)
    np.testing.assert_array_equal(darr_croo, _res)


def test_anom_ratio(darr):
    _res = np.array([[85.0, 443.0], [18.0, 83.0]])

    np.testing.assert_array_equal(
        darr.isel(time=0).anom.ratio(darr.isel(time=1)).round(), _res
    )


def test_anom_diff(darr):
    _res = np.array([[-9, 72], [-68, -15]])

    np.testing.assert_array_equal(darr.isel(time=0).anom.diff(darr.isel(time=1)), _res)


def test_algo_autocorr(darr):
    _res = np.array(
        [[-0.7395127, -0.69477093], [-0.1768683, 0.6412078]], dtype="float32"
    )
    darr_autocorr = darr.algo.autocorr()
    assert isinstance(darr_autocorr, xr.DataArray)
    np.testing.assert_almost_equal(darr_autocorr, _res)


def test_algo_spi(darr, res_spi):
    _res = darr.algo.spi()
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)


def test_algo_spi_transp(darr, res_spi):
    _darr = darr.transpose(..., "time")
    _res = _darr.algo.spi()
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)


def test_algo_spi_attrs_default(darr):
    _res = darr.algo.spi()
    assert _res.attrs["spi_calibration_start"] == str(darr.time.dt.date[0].values)
    assert _res.attrs["spi_calibration_stop"] == str(darr.time.dt.date[-1].values)


def test_algo_spi_attrs_start(darr):
    _res = darr.algo.spi(calibration_start="2000-01-02")
    assert _res.attrs["spi_calibration_start"] == "2000-01-11"


def test_algo_spi_attrs_stop(darr):
    _res = darr.algo.spi(calibration_stop="2000-02-09")
    assert _res.attrs["spi_calibration_stop"] == "2000-01-31"


def test_algo_spi_decoupled_1(darr, res_spi):
    _res = darr.algo.spi(calibration_start="2000-01-01", calibration_stop="2000-02-10")

    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

    assert _res.attrs["spi_calibration_start"] == "2000-01-01"
    assert _res.attrs["spi_calibration_stop"] == "2000-02-10"


def test_algo_spi_decoupled_2(darr):
    res_spi = np.array(
        [
            [[406, 583, 826, -1704, 1028], [1182, -1019, 792, -964, -581]],
            [[-1628, 770, 878, 13, -480], [795, 1001, -173, -1550, -1550]],
        ]
    )

    _res = darr.algo.spi(calibration_start="2000-01-01", calibration_stop="2000-01-31")

    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

    assert _res.attrs["spi_calibration_start"] == "2000-01-01"
    assert _res.attrs["spi_calibration_stop"] == "2000-01-31"


def test_algo_spi_decoupled_3(darr):
    res_spi = np.array(
        [
            [[240, 401, 622, -1706, 804], [2213, -790, 1675, -717, -201]],
            [[-3454, 852, 1044, -504, -1390], [1230, 1427, 320, -925, -925]],
        ]
    )

    _res = darr.algo.spi(calibration_start="2000-01-11")

    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

    assert _res.attrs["spi_calibration_start"] == "2000-01-11"
    assert _res.attrs["spi_calibration_stop"] == "2000-02-10"


def test_algo_spi_decoupled_err_1(darr):
    with pytest.raises(ValueError):
        _res = darr.algo.spi(
            calibration_start="2000-03-01",
        )


def test_algo_spi_decoupled_err_2(darr):
    with pytest.raises(ValueError):
        _res = darr.algo.spi(
            calibration_stop="1999-01-01",
        )


def test_algo_spi_decoupled_err_3(darr):
    with pytest.raises(ValueError):
        _res = darr.algo.spi(
            calibration_start="2000-01-01",
            calibration_stop="2000-01-01",
        )


def test_algo_spi_decoupled_err_4(darr):
    with pytest.raises(ValueError):
        _res = darr.algo.spi(
            calibration_start="2000-02-01",
            calibration_stop="2000-01-01",
        )


def test_algo_exception(darr):
    with pytest.raises(ValueError):
        _ = darr.isel(time=0).algo


def test_agg_sum_1(darr):
    n = 0
    ii = -1
    for _x in darr.iteragg.sum(1):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1


def test_agg_sum_1_lim_begin(darr):
    n = 0
    begin = "2000-01-21"
    ii = darr.time.to_index().get_loc(begin)
    for _x in darr.iteragg.sum(1, begin=begin):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3


def test_agg_sum_1_lim_end(darr):
    n = 0
    end = "2000-01-21"
    ii = -1
    for _x in darr.iteragg.sum(1, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3
    assert (
        _x.time.dt.strftime("%Y-%m-%d").values == end
    )  # pylint: disable=undefined-loop-variable


def test_agg_sum_1_lim_begin_end(darr):
    n = 0
    begin = "2000-01-31"
    end = "2000-01-11"
    ii = darr.time.to_index().get_loc(begin)
    for _x in darr.iteragg.sum(1, begin=begin, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3
    assert (
        _x.time.dt.strftime("%Y-%m-%d").values == end
    )  # pylint: disable=undefined-loop-variable


def test_agg_sum_3(darr):
    n = 0
    for _x in darr.iteragg.sum(n=3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs["agg_start"], _x.attrs["agg_stop"])).sum(
                "time"
            ),
        )
        assert _x.attrs["agg_n"] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs["agg_stop"]
        n += 1
    assert n == 3


def test_agg_mean_1(darr):
    n = 0
    ii = -1
    for _x in darr.iteragg.mean(1):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 5


def test_agg_mean_1_lim_begin(darr):
    n = 0
    begin = "2000-01-21"
    ii = darr.time.to_index().get_loc(begin)
    for _x in darr.iteragg.mean(1, begin=begin):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3


def test_agg_mean_1_lim_end(darr):
    n = 0
    end = "2000-01-21"
    ii = -1
    for _x in darr.iteragg.mean(1, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3
    assert (
        _x.time.dt.strftime("%Y-%m-%d").values == end
    )  # pylint: disable=undefined-loop-variable


def test_agg_mean_1_lim_begin_end(darr):
    n = 0
    begin = "2000-01-31"
    end = "2000-01-11"
    ii = darr.time.to_index().get_loc(begin)
    for _x in darr.iteragg.mean(1, begin=begin, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3
    assert (
        _x.time.dt.strftime("%Y-%m-%d").values == end
    )  # pylint: disable=undefined-loop-variable


def test_agg_mean_3(darr):
    n = 0
    for _x in darr.iteragg.mean(3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs["agg_start"], _x.attrs["agg_stop"])).mean(
                "time"
            ),
        )
        assert _x.attrs["agg_n"] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs["agg_stop"]
        n += 1
    assert n == 3


def test_agg_full_3(darr):
    n = 0
    for _x in darr.iteragg.full(3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs["agg_start"], _x.attrs["agg_stop"])),
        )
        assert _x.time.size == 3
        assert _x.attrs["agg_n"] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs["agg_start"]
        assert str(_x.time.to_index()[-1]) == _x.attrs["agg_stop"]
        n += 1
    assert n == 3


def test_whit_whits_s(darr):
    _res = np.array(
        [
            [[54, 54, 55, 56, 60], [74, 60, 48, 35, 24]],
            [[49, 56, 59, 58, 56], [82, 60, 37, 14, -7]],
        ],
        dtype="int16",
    )
    _darr = darr.whit.whits(nodata=0, s=10)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:, 0, 0].astype("float64").data
    w = ((y != 0) * 1).astype("float64")
    z = np.rint(seasmon_xr.src.ws2d.ws2d(y, 10.0, w))

    np.testing.assert_array_equal(_darr[0, 0, :].data, z)


def test_whit_whits_sg(darr):
    sg = np.full((2, 2), -0.5)
    _res = np.array(
        [
            [[55, 59, 55, 38, 72], [82, 49, 50, 34, 27]],
            [[26, 71, 81, 61, 39], [80, 69, 33, 8, -3]],
        ],
        dtype="int16",
    )
    _darr = darr.whit.whits(nodata=0, sg=sg)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:, 0, 0].astype("float64").data
    w = ((y != 0) * 1).astype("float64")
    l = 10 ** -0.5
    z = np.rint(seasmon_xr.src.ws2d.ws2d(y, l, w))

    np.testing.assert_array_equal(_darr[0, 0, :].data, z)


def test_whit_whits_sg_zeros(darr):
    sg = np.full((2, 2), -np.inf)
    _res = darr.transpose(..., "time")
    _darr = darr.whit.whits(nodata=0, sg=sg)
    np.testing.assert_array_equal(_darr, _res)


def test_whit_whits_sg_p(darr):
    sg = np.full((2, 2), -0.5)
    _res = np.array(
        [
            [[56, 64, 70, 73, 85], [91, 77, 67, 49, 30]],
            [[60, 79, 83, 72, 55], [100, 80, 51, 23, 0]],
        ],
        dtype="int16",
    )
    _darr = darr.whit.whits(nodata=0, sg=sg, p=0.90)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:, 0, 0].astype("float64").data
    l = 10 ** -0.5
    z = np.rint(seasmon_xr.src.ws2dpgu(y, l, 0, 0.90))

    np.testing.assert_array_equal(_darr[0, 0, :].data, z)


def test_whit_whits_sg_p_zeros(darr):
    sg = np.full((2, 2), -np.inf)
    _res = darr.transpose(..., "time")
    _darr = darr.whit.whits(nodata=0, sg=sg, p=0.90)
    np.testing.assert_array_equal(_darr, _res)


def test_whit_whitsvc(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[55, 59, 55, 38, 72], [82, 49, 50, 34, 27]],
            [[26, 71, 81, 61, 39], [80, 69, 33, 8, -3]],
        ],
        dtype="int16",
    )
    _darr = darr.whit.whitsvc(nodata=0, srange=srange)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.full((2, 2), -0.5))

    y = darr[:, 0, 0].astype("float64").data

    z, l = seasmon_xr.src.ws2doptv(y, 0.0, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -0.5

def test_whit_whitsvc_unnamed(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[55, 59, 55, 38, 72], [82, 49, 50, 34, 27]],
            [[26, 71, 81, 61, 39], [80, 69, 33, 8, -3]],
        ],
        dtype="int16",
    )

    darr.name = None
    _darr = darr.whit.whitsvc(nodata=0, srange=srange)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.full((2, 2), -0.5))

    y = darr[:, 0, 0].astype("float64").data

    z, l = seasmon_xr.src.ws2doptv(y, 0.0, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -0.5


def test_whit_whitsvcp(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[54, 66, 71, 49, 86], [92, 60, 70, 43, 30]],
            [[60, 79, 83, 72, 55], [85, 84, 40, 10, 1]],
        ],
        dtype="int16",
    )
    _darr = darr.whit.whitsvc(nodata=0, srange=srange, p=0.90)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.array([[-1.5, -1.5], [-0.5, -1.5]]))

    y = darr[:, 0, 0].astype("float64").data

    z, l = seasmon_xr.src.ws2doptvp(y, 0.0, 0.90, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -1.5


def test_whit_whitsvcp_unnamed(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[54, 66, 71, 49, 86], [92, 60, 70, 43, 30]],
            [[60, 79, 83, 72, 55], [85, 84, 40, 10, 1]],
        ],
        dtype="int16",
    )

    darr.name = None
    _darr = darr.whit.whitsvc(nodata=0, srange=srange, p=0.90)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.array([[-1.5, -1.5], [-0.5, -1.5]]))

    y = darr[:, 0, 0].astype("float64").data

    z, l = seasmon_xr.src.ws2doptvp(y, 0.0, 0.90, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -1.5


def test_whit_whitint(darr):
    labels_daily = np.array(
        [
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
        ],
        dtype="int32",
    )

    template = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    temp = template.copy()
    res = darr.astype("int16").whit.whitint(labels_daily, temp)
    assert "newtime" in res.dims
    assert res.newtime.size == 5

    xx = darr[:, 0, 0].values

    w = template.copy()
    temp = template.copy()

    for ix, ii in enumerate(np.where(template != 0)[0]):
        template[ii] = xx[ix]
    template[-1] = xx[-1]

    z = ws2d(template, 0.00001, w)

    _df = pd.DataFrame({"data": z, "labels": labels_daily})
    xx_int = _df.groupby("labels").mean().round().astype("int16")

    np.testing.assert_array_equal(res[0, 0, :], xx_int.data.values)


def test_whit_exception(darr):
    with pytest.raises(ValueError):
        _ = darr.isel(time=0).whit
