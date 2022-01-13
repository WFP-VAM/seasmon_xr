"""Tests for pixel alogrithms"""
# pylint: disable=no-name-in-module,redefined-outer-name,no-value-for-parameter
# pyright: reportGeneralTypeIssues=false
import numpy as np
import pytest

from seasmon_xr.ops import (
    autocorr,
    autocorr_1d,
    autocorr_tyx,
    lroo,
    ws2dgu,
    ws2doptv,
    ws2doptvp,
    ws2doptvplc,
    ws2dpgu,
)
from seasmon_xr.ops.spi import brentq, gammafit, spifun


@pytest.fixture
def ts():
    """Testdata"""
    np.random.seed(42)
    x = np.random.gamma(1, size=10)
    return x


def test_lroo(ts):
    x_lroo = lroo(np.array((ts > 0.9) * 1, dtype="uint8"))
    assert x_lroo == 3


def pearson_reference(X, Y):
    return ((X - X.mean()) * (Y - Y.mean())).mean() / (X.std() * Y.std())


def autocorr_1d_reference(x, nodata=None):
    if nodata is not None:
        _x = x.astype("float64")
        _x[x == nodata] = np.nan
        x = _x

    X = x[:-1]
    Y = x[1:]
    if np.isnan(x).any():
        X, Y = X.copy(), Y.copy()
        X[np.isnan(X)] = np.nanmean(X)
        Y[np.isnan(Y)] = np.nanmean(Y)

    return pearson_reference(X, Y)


def test_autocorr(ts):
    ac = autocorr(ts.reshape(1, 1, -1))
    np.testing.assert_almost_equal(ac, 0.00398337)

    np.testing.assert_almost_equal(autocorr_1d(ts), 0.00398337)
    np.testing.assert_almost_equal(autocorr_1d_reference(ts), 0.00398337)
    np.testing.assert_almost_equal(autocorr_tyx(ts.reshape(-1, 1, 1)), 0.00398337)


def test_autocorr_nodata(ts_ndvi):
    nodata, ts = ts_ndvi
    rr = autocorr_1d(ts, nodata)
    rr_ref = autocorr_1d_reference(ts, nodata)
    assert rr == pytest.approx(rr_ref, rel=1e-3)


def test_brentq():
    x = brentq(xa=0.6446262296476516, xb=1.5041278691778537, s=0.5278852360624721)
    assert x == pytest.approx(1.083449238500003)


def test_gammafit(ts):
    parameters = gammafit(ts)
    assert parameters == pytest.approx((1.083449238500003, 0.9478709674697126))


def test_spi(ts):
    xspi = spifun(ts.reshape(1, 1, -1))
    assert xspi.shape == (1, 1, 10)
    np.testing.assert_array_equal(
        xspi[0, 0, :],
        [-382.0, 1654.0, 588.0, 207.0, -1097.0, -1098.0, -1677.0, 1094.0, 213.0, 514.0],
    )


def test_spi_nofit(ts):
    xspi = spifun(ts.reshape(1, 1, -1), a=1, b=2)
    assert xspi.shape == (1, 1, 10)
    np.testing.assert_array_equal(
        xspi[0, 0, :],
        [
            -809.0,
            765.0,
            -44.0,
            -341.0,
            -1396.0,
            -1396.0,
            -1889.0,
            343.0,
            -336.0,
            -101.0,
        ],
    )


def test_spi_selfit(ts):
    xspi = spifun(ts.reshape(1, 1, -1), cal_start=0, cal_stop=3)
    assert xspi.shape == (1, 1, 10)
    np.testing.assert_array_equal(
        xspi[0, 0, :],
        [
            -1211.0,
            1236.0,
            -32.0,
            -492.0,
            -2099.0,
            -2099.0,
            -2833.0,
            572.0,
            -484.0,
            -120.0,
        ],
    )


def test_spi_selfit_2(ts):
    cal_start = 2
    cal_stop = 8

    a, b = gammafit(ts[cal_start:cal_stop])
    xspi_ref = spifun(ts.reshape(1, 1, -1), a=a, b=b)
    xspi = spifun(ts.reshape(1, 1, -1), cal_start=cal_start, cal_stop=cal_stop)
    np.testing.assert_equal(xspi, xspi_ref)


def test_ws2dgu(ts):
    _ts = ts * 10
    z = ws2dgu(_ts, 10, 0)
    np.testing.assert_array_equal(z, [15, 14, 12, 9, 8, 7, 7, 9, 10, 12])


def test_ws2dpgu(ts):
    _ts = ts * 10
    z = ws2dpgu(_ts, 10, 0, 0.9)
    np.testing.assert_array_equal(z, [26, 24, 22, 20, 18, 17, 16, 15, 15, 14])


def test_ws2doptv(ts):
    _ts = ts * 10
    z, l = ws2doptv(_ts, 0, np.arange(-2, 2))
    np.testing.assert_array_equal(z, [10, 21, 16, 9, 3, 2, 5, 13, 12, 12])
    assert l == pytest.approx(0.31622776601683794)


def test_ws2doptvp(ts):
    _ts = ts * 10
    z, l = ws2doptvp(_ts, 0, 0.9, np.arange(-2, 2))
    np.testing.assert_array_equal(z, [13, 28, 19, 9, 3, 2, 7, 19, 15, 12])
    assert l == pytest.approx(0.03162277660168379)


def test_ws2doptvplc(ts):
    _ts = (ts * 10).astype("int16")
    z, l = ws2doptvplc(_ts, 0, 0.9, 0.9)
    np.testing.assert_array_equal(z, [12, 28, 19, 9, 3, 4, 13, 19, 14, 12])
    assert l == pytest.approx(0.03162277660168379)
