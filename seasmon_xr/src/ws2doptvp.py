"""Whittaker filter V-curve optimization of S and asymmetric weights."""
# pyright: reportGeneralTypeIssues=false
from math import log, sqrt

import numpy
from numba import guvectorize, jit
from numba.core.types import float64, int16

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, float64[:], int16[:], float64[:])],
        "(n),(),(),(m) -> (n),()",
        nopython=True,
    )
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


@jit(nopython=True)
def _ws2doptvp(y, w, p, llas):
    """
    Whittaker filter V-curve optimization of S and asymmetric weights.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        w (numpy.array): weights same size as y
        p (float): Envelope value for asymmetric weights
        llas (numpy.array): 1d array of s values to use for optimization
    """
    m = y.shape[0]

    m1 = m - 1
    m2 = m - 2
    nl = len(llas)
    nl1 = nl - 1
    i = 0
    k = 0
    j = 0
    p1 = 1 - p

    fits = numpy.zeros(nl, dtype=float64)
    pens = numpy.zeros(nl, dtype=float64)
    z = numpy.zeros(m, dtype=float64)
    znew = numpy.zeros(m, dtype=float64)
    diff1 = numpy.zeros(m1, dtype=float64)
    lamids = numpy.zeros(nl1, dtype=float64)
    v = numpy.zeros(nl1, dtype=float64)
    wa = numpy.zeros(m, dtype=float64)
    ww = numpy.zeros(m, dtype=float64)

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

    lopt = pow(10, lamids[k])

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

        znew[0:m] = ws2d(y, lopt, ww)
        z_tmp = 0.0
        j = 0
        for j in range(m):
            z_tmp += abs(znew[j] - z[j])

        if z_tmp == 0.0:
            break

        z[0:m] = znew[0:m]

    z = ws2d(y, lopt, ww)
    return z, lopt
