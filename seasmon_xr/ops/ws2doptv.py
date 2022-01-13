"""Whittaker filter V-curve optimization os S."""
from math import log, sqrt

import numpy
from numba import guvectorize
from numba.core.types import float64, int16

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64[:], int16[:], float64[:])],
        "(n),(),(m) -> (n),()",
        nopython=True,
    )
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
    w = numpy.zeros(y.shape)
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
