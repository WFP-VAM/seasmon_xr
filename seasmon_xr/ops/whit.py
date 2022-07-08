"""Numba accelerated Whittaker functions."""
# pyright: reportGeneralTypeIssues=false
# pylint: disable=C0103,C0301,E0401,R0912,R0913,R0914,R0915,too-many-lines
from math import log, pow, sqrt  # pylint: disable=W0622
from typing import Union

import numpy
import xarray
from numba import guvectorize, njit
from numba.core.types import float64, int16, int32, uint8, boolean


@njit
def ws2d(y, lmda, w):
    """
    Whittaker filter with differences of 2nd order.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        lmda (double): S value
        w (numpy.array): weights vector (1d, expected in float64)
    Returns:
        z (numpy.array): smoothed data array (1d)
    """
    n = y.shape[0]
    m = n - 1
    z = numpy.zeros(n)
    d = z.copy()
    c = z.copy()
    e = z.copy()

    d[0] = w[0] + lmda
    c[0] = (-2 * lmda) / d[0]
    e[0] = lmda / d[0]
    z[0] = w[0] * y[0]
    d[1] = w[1] + 5 * lmda - d[0] * (c[0] * c[0])
    c[1] = (-4 * lmda - d[0] * c[0] * e[0]) / d[1]
    e[1] = lmda / d[1]
    z[1] = w[1] * y[1] - c[0] * z[0]
    for i in range(2, m - 1):
        i1 = i - 1
        i2 = i - 2
        d[i] = w[i] + 6 * lmda - (c[i1] * c[i1]) * d[i1] - (e[i2] * e[i2]) * d[i2]
        c[i] = (-4 * lmda - d[i1] * c[i1] * e[i1]) / d[i]
        e[i] = lmda / d[i]
        z[i] = w[i] * y[i] - c[i1] * z[i1] - e[i2] * z[i2]
    i1 = m - 2
    i2 = m - 3
    d[m - 1] = w[m - 1] + 5 * lmda - (c[i1] * c[i1]) * d[i1] - (e[i2] * e[i2]) * d[i2]
    c[m - 1] = (-2 * lmda - d[i1] * c[i1] * e[i1]) / d[m - 1]
    z[m - 1] = w[m - 1] * y[m - 1] - c[i1] * z[i1] - e[i2] * z[i2]
    i1 = m - 1
    i2 = m - 2
    d[m] = w[m] + lmda - (c[i1] * c[i1]) * d[i1] - (e[i2] * e[i2]) * d[i2]
    z[m] = (w[m] * y[m] - c[i1] * z[i1] - e[i2] * z[i2]) / d[m]
    z[m - 1] = z[m - 1] / d[m - 1] - c[m - 1] * z[m]
    for i in range(m - 2, -1, -1):
        z[i] = z[i] / d[i] - c[i] * z[i + 1] - e[i] * z[i + 2]

    return z


@guvectorize(
    [(float64[:], float64, float64, int16[:])], "(n),(),() -> (n)", nopython=True
)
def ws2dgu(y, lmda, nodata, out):
    """
    Whittaker smoother with fixed lambda (S).

    Args:
        y: time-series numpy array
        l: smoothing parameter lambda (S)
        w: weights numpy array
        p: "Envelope" value
    Returns:
        Smoothed time-series array z
    """
    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)

    n = 0
    for ii in range(m):
        if y[ii] == nodata:
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1

    if n > 1:
        z = ws2d(y, lmda, w)
        numpy.round_(z, 0, out)
    else:
        out[:] = y[:]


@guvectorize(
    [(float64[:], float64, float64, float64, int16[:])],
    "(n),(),(),() -> (n)",
    nopython=True,
)
def ws2dpgu(y, lmda, nodata, p, out):
    """
    Whittaker smoother with asymmetric smoothing and fixed lambda (S).

    Args:
        y: time-series numpy array
        l: smoothing parameter lambda (S)
        w: weights numpy array
        p: "Envelope" value
    Returns:
        Smoothed time-series array z
    """
    w = numpy.zeros(y.shape, dtype=float64)
    m = y.shape[0]
    n = 0

    for ii in range(m):
        if y[ii] == nodata:
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1

    if n > 1:
        p1 = 1 - p
        z = numpy.zeros(m)
        znew = numpy.zeros(m)
        wa = numpy.zeros(m)
        ww = numpy.zeros(m)

        # Calculate weights

        for _ in range(10):
            for j in range(m):
                y_tmp = y[j]
                z_tmp = z[j]

                if y_tmp > z_tmp:
                    wa[j] = p
                else:
                    wa[j] = p1
                ww[j] = w[j] * wa[j]

            znew[:] = ws2d(y, lmda, ww)
            z_tmp = 0.0
            j = 0
            for j in range(m):
                z_tmp += abs(znew[j] - z[j])

            if z_tmp == 0.0:
                break

            z[:] = znew[:]

        z = ws2d(y, lmda, ww)
        numpy.round_(z, 0, out)

    else:
        out[:] = y[:]


@guvectorize(
    [(float64[:], float64, float64[:], int16[:], float64[:])],
    "(n),(),(m) -> (n),()",
    nopython=True,
)
def ws2doptv(y, nodata, llas, out, lopt):
    """
    Whittaker filter V-curve optimization of S.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        llas (numpy.array): 1d array of s values to use for optimization
    """
    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)
    n = 0
    for ii in range(m):
        if y[ii] == nodata:
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1
    if n > 1:
        m1 = m - 1
        m2 = m - 2
        nl = len(llas)
        nl1 = nl - 1
        i = 0
        k = 0

        fits = numpy.zeros(nl)
        pens = numpy.zeros(nl)
        z = numpy.zeros(m)
        diff1 = numpy.zeros(m1)
        lamids = numpy.zeros(nl1)
        v = numpy.zeros(nl1)

        # Compute v-curve
        for lix in range(nl):
            lmda = pow(10, llas[lix])
            z[:] = ws2d(y, lmda, w)
            for i in range(m):
                w_tmp = w[i]
                y_tmp = y[i]
                z_tmp = z[i]
                fits[lix] += pow(w_tmp * (y_tmp - z_tmp), 2)
            fits[lix] = log(fits[lix])

            for i in range(m1):
                z_tmp = z[i]
                z2 = z[i + 1]
                diff1[i] = z2 - z_tmp
            for i in range(m2):
                z_tmp = diff1[i]
                z2 = diff1[i + 1]
                pens[lix] += pow(z2 - z_tmp, 2)
            pens[lix] = log(pens[lix])

        # Construct v-curve
        llastep = llas[1] - llas[0]

        for i in range(nl1):
            l1 = llas[i]
            l2 = llas[i + 1]
            f1 = fits[i]
            f2 = fits[i + 1]
            p1 = pens[i]
            p2 = pens[i + 1]
            v[i] = sqrt(pow(f2 - f1, 2) + pow(p2 - p1, 2)) / (log(10) * llastep)
            lamids[i] = (l1 + l2) / 2

        vmin = v[k]
        for i in range(1, nl1):
            if v[i] < vmin:
                vmin = v[i]
                k = i

        lopt[0] = pow(10, lamids[k])
        z = ws2d(y, lopt[0], w)
        numpy.round_(z, 0, out)
    else:
        out[:] = y[:]
        lopt[0] = 0.0


@guvectorize(
    [(float64[:], float64, float64, float64[:], int16[:], float64[:])],
    "(n),(),(),(m) -> (n),()",
    nopython=True,
)
def ws2doptvp(y, nodata, p, llas, out, lopt):
    """
    Whittaker filter V-curve optimization of S and asymmetric weights.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        p (float): Envelope value for asymmetric weights
        llas (numpy.array): 1d array of s values to use for optimization
    """
    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)

    n = 0
    for ii in range(m):
        if y[ii] == nodata:
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1
    if n > 1:
        m1 = m - 1
        m2 = m - 2
        nl = len(llas)
        nl1 = nl - 1
        i = 0
        k = 0
        j = 0
        p1 = 1 - p

        fits = numpy.zeros(nl)
        pens = numpy.zeros(nl)
        z = numpy.zeros(m)
        znew = numpy.zeros(m)
        diff1 = numpy.zeros(m1)
        lamids = numpy.zeros(nl1)
        v = numpy.zeros(nl1)
        wa = numpy.zeros(m)
        ww = numpy.zeros(m)

        # Compute v-curve
        for lix in range(nl):
            lmda = pow(10, llas[lix])

            for i in range(10):
                for j in range(m):
                    y_tmp = y[j]
                    z_tmp = z[j]
                    if y_tmp > z_tmp:
                        wa[j] = p
                    else:
                        wa[j] = p1
                    ww[j] = w[j] * wa[j]

                znew[:] = ws2d(y, lmda, ww)
                z_tmp = 0.0
                j = 0
                for j in range(m):
                    z_tmp += abs(znew[j] - z[j])

                if z_tmp == 0.0:
                    break

                z[0:m] = znew[0:m]

            for i in range(m):
                w_tmp = w[i]
                y_tmp = y[i]
                z_tmp = z[i]
                fits[lix] += pow(w_tmp * (y_tmp - z_tmp), 2)
            fits[lix] = log(fits[lix])

            for i in range(m1):
                z_tmp = z[i]
                z2 = z[i + 1]
                diff1[i] = z2 - z_tmp
            for i in range(m2):
                z_tmp = diff1[i]
                z2 = diff1[i + 1]
                pens[lix] += pow(z2 - z_tmp, 2)
            pens[lix] = log(pens[lix])

        # Construct v-curve
        llastep = llas[1] - llas[0]

        for i in range(nl1):
            l1 = llas[i]
            l2 = llas[i + 1]
            fit1 = fits[i]
            fit2 = fits[i + 1]
            pen1 = pens[i]
            pen2 = pens[i + 1]
            v[i] = sqrt(pow(fit2 - fit1, 2) + pow(pen2 - pen1, 2)) / (log(10) * llastep)
            lamids[i] = (l1 + l2) / 2

        vmin = v[k]
        for i in range(1, nl1):
            if v[i] < vmin:
                vmin = v[i]
                k = i

        lopt[0] = pow(10, lamids[k])

        z[:] = 0.0

        for i in range(10):
            for j in range(m):
                y_tmp = y[j]
                z_tmp = z[j]

                if y_tmp > z_tmp:
                    wa[j] = p
                else:
                    wa[j] = p1
                ww[j] = w[j] * wa[j]

            znew[0:m] = ws2d(y, lopt[0], ww)
            z_tmp = 0.0
            j = 0
            for j in range(m):
                z_tmp += abs(znew[j] - z[j])

            if z_tmp == 0.0:
                break

            z[0:m] = znew[0:m]

        z = ws2d(y, lopt[0], ww)
        numpy.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0


@guvectorize(
    [(int16[:], float64, float64, float64, int16[:], float64[:])],
    "(n),(),(),() -> (n),()",
    nopython=True,
)
def ws2doptvplc(y, nodata, p, lc, out, lopt):
    """
    Whittaker filter V-curve optimization.

    Whittaker filter V-curve optimization of S, asymmetric weights and
    srange determined by autocorrelation.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        p (float): Envelope value for asymmetric weights
        lc (float): lag1 autocorrelation
    """
    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)

    n = 0
    for ii in range(m):
        if y[ii] == nodata:
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1

    if n > 1:
        if lc > 0.5:
            llas = numpy.arange(-2, 1.2, 0.2, dtype=float64)
        elif lc <= 0.5:
            llas = numpy.arange(0, 3.2, 0.2, dtype=float64)
        else:
            llas = numpy.arange(-1, 1.2, 0.2, dtype=float64)

        m1 = m - 1
        m2 = m - 2
        nl = len(llas)
        nl1 = nl - 1
        i = 0
        k = 0
        j = 0
        p1 = 1 - p

        fits = numpy.zeros(nl)
        pens = numpy.zeros(nl)
        z = numpy.zeros(m)
        znew = numpy.zeros(m)
        diff1 = numpy.zeros(m1)
        lamids = numpy.zeros(nl1)
        v = numpy.zeros(nl1)
        wa = numpy.zeros(m)
        ww = numpy.zeros(m)

        # Compute v-curve
        for lix in range(nl):
            lmda = pow(10, llas[lix])

            for i in range(10):
                for j in range(m):
                    y_tmp = y[j]
                    z_tmp = z[j]
                    if y_tmp > z_tmp:
                        wa[j] = p
                    else:
                        wa[j] = p1
                    ww[j] = w[j] * wa[j]

                znew[:] = ws2d(y, lmda, ww)
                z_tmp = 0.0
                j = 0
                for j in range(m):
                    z_tmp += abs(znew[j] - z[j])

                if z_tmp == 0.0:
                    break

                z[0:m] = znew[0:m]

            for i in range(m):
                w_tmp = w[i]
                y_tmp = y[i]
                z_tmp = z[i]
                fits[lix] += pow(w_tmp * (y_tmp - z_tmp), 2)
            fits[lix] = log(fits[lix])

            for i in range(m1):
                z_tmp = z[i]
                z2 = z[i + 1]
                diff1[i] = z2 - z_tmp
            for i in range(m2):
                z_tmp = diff1[i]
                z2 = diff1[i + 1]
                pens[lix] += pow(z2 - z_tmp, 2)
            pens[lix] = log(pens[lix])

        # Construct v-curve
        llastep = llas[1] - llas[0]

        for i in range(nl1):
            l1 = llas[i]
            l2 = llas[i + 1]
            fit1 = fits[i]
            fit2 = fits[i + 1]
            pen1 = pens[i]
            pen2 = pens[i + 1]
            v[i] = sqrt(pow(fit2 - fit1, 2) + pow(pen2 - pen1, 2)) / (log(10) * llastep)
            lamids[i] = (l1 + l2) / 2

        vmin = v[k]
        for i in range(1, nl1):
            if v[i] < vmin:
                vmin = v[i]
                k = i

        lopt[0] = pow(10, lamids[k])

        z[:] = 0.0

        for i in range(10):
            for j in range(m):
                y_tmp = y[j]
                z_tmp = z[j]

                if y_tmp > z_tmp:
                    wa[j] = p
                else:
                    wa[j] = p1
                ww[j] = w[j] * wa[j]

            znew[0:m] = ws2d(y, lopt[0], ww)
            z_tmp = 0.0
            j = 0
            for j in range(m):
                z_tmp += abs(znew[j] - z[j])

            if z_tmp == 0.0:
                break

            z[0:m] = znew[0:m]

        z = ws2d(y, lopt[0], ww)
        numpy.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0


@guvectorize(
    [(float64[:], float64, float64[:], boolean, int16[:], float64[:])],
    "(n),(),(m),() -> (n),()",
    nopython=True,
)
def ws2dwcv(y, nodata, llas, robust, out, lopt):
    """
    Whittaker filter GCV optimization of S.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        llas (numpy.array): 1d array of s values to use for optimization
        robust (boolean): performs a robust fitting by computing robust weights if True
    """
    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)

    n = 0
    for ii in range(m):
        if (y[ii] == nodata) or numpy.isnan(y[ii]) or numpy.isinf(y[ii]):
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1

    # Eigenvalues
    d_eigs = -2 + 2 * numpy.cos(numpy.arange(m) * numpy.pi / m)
    d_eigs[0] = 1e-15

    if n > 1:

        z = numpy.zeros(m)
        r_weights = numpy.ones(m)

        # Setting number of robust iterations to perform
        if not robust:
            r_its = 1
        else:
            r_its = 4

        # Initialising lists for writing to
        robust_gcv = []

        gcv_temp = [1e15, 0]
        for it in range(r_its):
            if it > 1:
                Sopt_Rog_val = robust_gcv[1][1]
            else:
                Sopt_Rog_val = 0.0

            if not Sopt_Rog_val:
                smoothing = 10**llas
            else:
                smoothing = numpy.array([Sopt_Rog_val])

            w_temp = w * r_weights
            for lmda in smoothing:

                NDVI_smoothed = ws2d(y, lmda, w_temp)

                gamma = w_temp / (w_temp + lmda * ((-1 * d_eigs) ** 2))
                tr_H = gamma.sum()
                wsse = (((w_temp**0.5) * (y - NDVI_smoothed)) ** 2).sum()
                denominator = w_temp.sum() * (1 - (tr_H / (w_temp.sum()))) ** 2
                gcv_score = wsse / denominator

                gcv, NDVIhat = ([gcv_score, lmda], NDVI_smoothed)

                if gcv[0] < gcv_temp[0]:
                    gcv_temp = gcv
                    tempNDVI_arr = NDVIhat

            best_gcv = gcv_temp
            s = best_gcv[1]

            if robust:
                gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
                r_arr = y - tempNDVI_arr

                MAD = numpy.median(
                    numpy.abs(
                        r_arr[r_weights != 0] - numpy.median(r_arr[r_weights != 0])
                    )
                )
                u_arr = r_arr / (1.4826 * MAD * numpy.sqrt(1 - gamma.sum() / n))

                r_weights = (1 - (u_arr / 4.685) ** 2) ** 2
                r_weights[(numpy.abs(u_arr / 4.685) > 1)] = 0

                r_weights[r_arr > 0] = 1

            robust_weights = w * r_weights

            robust_gcv.append(best_gcv)

        robust_gcv = numpy.array(robust_gcv)

        if robust:
            lopt[0] = robust_gcv[1, 1]
        else:
            lopt[0] = robust_gcv[0, 1]

        z[:] = 0.0
        z = ws2d(y, lopt[0], robust_weights)
        numpy.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0


@guvectorize(
    [(float64[:], float64, float64, float64[:], boolean, int16[:], float64[:])],
    "(n),(),(),(m),() -> (n),()",
    nopython=True,
)
def ws2dwcvp(y, nodata, p, llas, robust, out, lopt):
    """
    Whittaker filter GCV optimization of S and asymmetric weights.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        p (float): Envelope value for asymmetric weights
        llas (numpy.array): 1d array of s values to use for optimization
        robust (boolean): performs a robust fitting by computing robust weights if True
    """
    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)

    n = 0
    for ii in range(m):
        if (y[ii] == nodata) or numpy.isnan(y[ii]) or numpy.isinf(y[ii]):
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1

    # Eigenvalues
    d_eigs = -2 + 2 * numpy.cos(numpy.arange(m) * numpy.pi / m)
    d_eigs[0] = 1e-15

    if n > 1:

        z = numpy.zeros(m)
        znew = numpy.zeros(m)
        wa = numpy.zeros(m)
        ww = numpy.zeros(m)
        r_weights = numpy.ones(m)

        # Setting number of robust iterations to perform
        if not robust:
            r_its = 1
        else:
            r_its = 4

        # Initialising lists for writing to
        robust_gcv = []

        gcv_temp = [1e15, 0]
        for it in range(r_its):
            if it > 1:
                Sopt_Rog_val = robust_gcv[1][1]
            else:
                Sopt_Rog_val = 0.0

            if not Sopt_Rog_val:
                smoothing = 10**llas
            else:
                smoothing = numpy.array([Sopt_Rog_val])

            w_temp = w * r_weights
            for lmda in smoothing:

                NDVI_smoothed = ws2d(y, lmda, w_temp)

                gamma = w_temp / (w_temp + lmda * ((-1 * d_eigs) ** 2))
                tr_H = gamma.sum()
                wsse = (((w_temp**0.5) * (y - NDVI_smoothed)) ** 2).sum()
                denominator = w_temp.sum() * (1 - (tr_H / (w_temp.sum()))) ** 2
                gcv_score = wsse / denominator

                gcv, NDVIhat = ([gcv_score, lmda], NDVI_smoothed)

                if gcv[0] < gcv_temp[0]:
                    gcv_temp = gcv
                    tempNDVI_arr = NDVIhat

            best_gcv = gcv_temp
            s = best_gcv[1]

            if robust:
                gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
                r_arr = y - tempNDVI_arr

                MAD = numpy.median(
                    numpy.abs(
                        r_arr[r_weights != 0] - numpy.median(r_arr[r_weights != 0])
                    )
                )
                u_arr = r_arr / (1.4826 * MAD * numpy.sqrt(1 - gamma.sum() / n))

                r_weights = (1 - (u_arr / 4.685) ** 2) ** 2
                r_weights[(numpy.abs(u_arr / 4.685) > 1)] = 0

                r_weights[r_arr > 0] = 1

            robust_weights = w * r_weights

            robust_gcv.append(best_gcv)

        robust_gcv = numpy.array(robust_gcv)

        if robust:
            lopt[0] = robust_gcv[1, 1]
        else:
            lopt[0] = robust_gcv[0, 1]

        z[:] = 0.0

        for _ in range(10):
            for j in range(m):
                y_tmp = y[j]
                z_tmp = z[j]

                if y_tmp > z_tmp:
                    wa[j] = p
                else:
                    wa[j] = 1 - p
                ww[j] = robust_weights[j] * wa[j]

            znew[0:m] = ws2d(y, lopt[0], ww)
            z_tmp = 0.0
            j = 0
            for j in range(m):
                z_tmp += abs(znew[j] - z[j])

            if z_tmp == 0.0:
                break

            z[0:m] = znew[0:m]

        z = ws2d(y, lopt[0], ww)
        numpy.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0


def whits(
    ds: xarray.Dataset,
    dim: str,
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
        dim: dimension to use for filtering
        nodata: nodata value
        sg: sgrid,
        s: S value
        p: Envelope value for asymmetric weights

    Returns:
        ds_out: xarray.Dataset with smoothed data
    """
    if sg is None and s is None:
        raise ValueError("Need S or sgrid")

    lmda = sg if sg is not None else s

    if p is not None:

        xout = xarray.apply_ufunc(
            ws2dpgu,
            ds[dim],
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
            ws2dgu,
            ds[dim],
            lmda,
            nodata,
            input_core_dims=[["time"], [], []],
            output_core_dims=[["time"]],
            dask="parallelized",
            keep_attrs=True,
        )

    return xout


def whitsvc(
    ds: xarray.Dataset,
    dim: str,
    nodata: Union[int, float],
    lc: xarray.DataArray = None,
    srange: numpy.ndarray = None,
    p: float = None,
) -> xarray.Dataset:
    """
    Apply whittaker with V-curve optimization of S.

    Args:
        ds: input dataset,
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
            raise ValueError("If lc is set, a p value needs to be specified as well.")

        ds_out, sgrid = xarray.apply_ufunc(
            ws2doptvplc,
            ds[dim],
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
                ws2doptvp,
                ds[dim],
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
                ws2doptv,
                ds[dim],
                nodata,
                srange,
                input_core_dims=[["time"], [], ["dim0"]],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

    ds_out = ds_out.to_dataset()
    ds_out["sgrid"] = numpy.log10(sgrid).astype("float32")

    return ds_out


def whitswcv(
    ds: xarray.Dataset,
    dim: str,
    nodata: Union[int, float],
    srange: numpy.ndarray = None,
    p: float = None,
    robust: boolean = True,
) -> xarray.Dataset:
    """
    Apply whittaker with GCV optimization of S.

    Args:
        ds: input dataset,
        dim: dimension to use for filtering
        nodata: nodata value
        lc: lag1 autocorrelation DataArray,
        srange: values of S for V-curve optimization (mandatory if no autocorrelation raster)
        p: Envelope value for asymmetric weights
        robust (boolean): perform a robust fitting by computing robust weights if True

    Returns:
        ds_out: xarray.Dataset with smoothed data and sgrid
    """
    if p:
        ds_out, sgrid = xarray.apply_ufunc(
            ws2dwcvp,
            ds[dim],
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

        ds_out, sgrid = xarray.apply_ufunc(
            ws2dwcv,
            ds[dim],
            nodata,
            srange,
            robust,
            input_core_dims=[["time"], [], ["dim0"], []],
            output_core_dims=[["time"], []],
            dask="parallelized",
            keep_attrs=True,
        )

    ds_out = ds_out.to_dataset()
    ds_out["sgrid"] = numpy.log10(sgrid).astype("float32")

    return ds_out


def whitint(
    ds: xarray.Dataset, dim: str, labels_daily: numpy.ndarray, template: numpy.ndarray
):
    """Perform temporal interpolation using the Whittaker filter."""
    template_out = numpy.zeros(numpy.unique(labels_daily).size, dtype="u1")

    ds_out = xarray.apply_ufunc(
        tinterpolate,
        ds[dim],
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


@njit
def autocorr(x):
    """
    Calculate Lag-1 autocorrelation.

    Adapted from https://stackoverflow.com/a/29194624/5997555
    Args:
        x: 3d data array
    Returns:
        Lag-1 autocorrelation array
    """
    r, c, t = x.shape
    z = numpy.zeros((r, c), dtype="float32")

    M = t - 1

    for rr in range(r):
        for cc in range(c):

            data1 = x[rr, cc, 1:]
            data2 = x[rr, cc, :-1]

            sum1 = 0.0
            sum2 = 0.0
            for i in range(M):
                sum1 += data1[i]
                sum2 += data2[i]

            mean1 = sum1 / M
            mean2 = sum2 / M

            var_sum1 = 0.0
            var_sum2 = 0.0
            cross_sum = 0.0
            for i in range(M):
                var_sum1 += (data1[i] - mean1) ** 2
                var_sum2 += (data2[i] - mean2) ** 2
                cross_sum += data1[i] * data2[i]

            std1 = (var_sum1 / M) ** 0.5
            std2 = (var_sum2 / M) ** 0.5
            cross_mean = cross_sum / M

            if std1 != 0 and std2 != 0:

                lc = (cross_mean - mean1 * mean2) / (std1 * std2)
            else:
                lc = 0.0
            z[rr, cc] = lc
    return z


def lag1corr(ds: xarray.Dataset, dim: str):
    """
    Xarray wrapper for autocorr.

    Args:
        ds: input dataset,
        dim: dimension to use for calculation

    Returns:
        xarray.DataArray with lag1 autocorrelation
    """
    return xarray.apply_ufunc(
        autocorr,
        ds[dim],
        input_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=["float32"],
    )


@guvectorize(
    [(int16[:], float64[:], int32[:], uint8[:], int16[:])],
    "(n),(m),(m),(l) -> (l)",
    nopython=True,
)
def tinterpolate(x, template, labels, template_out, out):  # pylint: disable=W0613
    """
    Temporal interpolation of smoothed data.

    Args:
        x: smoothed data
        template: zeros array in daily length with 1s marking smoothed data
        labels: array of labels for grouping of length equal to template
        template_out: helper array to determine the length of output array
    """
    temp = template.copy()
    w = template.copy()
    ii = 0
    jj = 0
    for tt in temp:
        if tt != 0:
            temp[ii] = x[jj]
            jj += 1
        ii += 1
    temp[-1] = x[-1]
    z = ws2d(temp, 0.00001, w)
    ii = 1
    jj = 1
    kk = 0
    v = z[0]

    for ll in labels[1:]:
        if ll == labels[ii - 1]:
            v += z[ii]
            jj += 1
        else:
            out[kk] = round(v / jj)
            kk += 1
            jj = 1
            v = z[ii]

        ii += 1

    out[kk] = round(v / jj)
