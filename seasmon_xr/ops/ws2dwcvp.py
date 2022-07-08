"""Whittaker filter V-curve optimization of S and asymmetric weights."""
# pyright: reportGeneralTypeIssues=false
import numpy
from numba import guvectorize, jit
from numba.core.types import float64, int16, boolean

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, float64[:], boolean, int16[:], float64[:])],
        "(n),(),(),(m),() -> (n),()",
        nopython=True,
    )
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

    if n > 5:

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


@jit(nopython=True)
def _ws2dwcvp(y, w, p, llas, robust):
    """
    Whittaker filter GCV optimization of S and asymmetric weights.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        w (numpy.array): weights same size as y
        p (float): Envelope value for asymmetric weights
        llas (numpy.array): 1d array of s values to use for optimization
        robust (boolean): performs a robust fitting by computing robust weights if True
    """
    m = y.shape[0]
    n = w.sum()

    # Eigenvalues
    d_eigs = -2 + 2 * numpy.cos(numpy.arange(m) * numpy.pi / m)
    d_eigs[0] = 1e-15

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
                numpy.abs(r_arr[r_weights != 0] - numpy.median(r_arr[r_weights != 0]))
            )
            u_arr = r_arr / (1.4826 * MAD * numpy.sqrt(1 - gamma.sum() / n))

            r_weights = (1 - (u_arr / 4.685) ** 2) ** 2
            r_weights[(numpy.abs(u_arr / 4.685) > 1)] = 0

            r_weights[r_arr > 0] = 1

        robust_weights = w * r_weights

        robust_gcv.append(best_gcv)

    robust_gcv = numpy.array(robust_gcv)

    if robust:
        lopt = robust_gcv[1, 1]
    else:
        lopt = robust_gcv[0, 1]

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

        znew[0:m] = ws2d(y, lopt, ww)
        z_tmp = 0.0
        j = 0
        for j in range(m):
            z_tmp += abs(znew[j] - z[j])

        if z_tmp == 0.0:
            break

        z[0:m] = znew[0:m]

    z = ws2d(y, lopt, ww)
    z = numpy.round_(z, 0)

    return z, lopt
