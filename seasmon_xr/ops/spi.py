"""Tooling to calculate Standardized Precipitation Index (SPI)."""
from math import log, sqrt

import numba
import numba.core.types as nt
import numba_scipy  # pylint: disable=unused-import
import numpy as np
import scipy.special as sc


@numba.njit
def brentq(xa, xb, s):
    """
    Root finding optimization using Brent's method.

    adapted from:

    https://github.com/scipy/scipy/blob/f2ef65dc7f00672496d7de6154744fee55ef95e9/scipy/optimize/Zeros/brentq.c#L37
    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.
    """
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    maxiter = 100
    xtol = 2e-12
    rtol = 8.881784197001252e-16

    func = lambda a: log(a) - sc.digamma(a) - s

    fpre = func(xpre)
    fcur = func(xcur)

    if (fpre * fcur) > 0:
        return 0.0

    if fpre == 0:
        return xpre

    if fcur == 0:
        return xcur

    iterations = 0

    for _ in range(maxiter):
        iterations += 1

        if (fpre * fcur) < 0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if abs(fblk) < abs(fcur):
            xpre, xcur = xcur, xblk
            xblk = xpre

            fpre, fcur = fcur, fblk
            fblk = fpre

        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2

        if fcur == 0 or (abs(sbis) < delta):
            return xcur

        if (abs(spre) > delta) and (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = (
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                )

            if (2 * abs(stry)) < min(abs(spre), 3 * abs(sbis) - delta):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis

        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur

        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = func(xcur)

    return xcur


@numba.njit
def gammafit(x):
    """
    Calculate gamma distribution parameters for timeseries.

    Adapted from:
    https://github.com/scipy/scipy/blob/f2ef65dc7f00672496d7de6154744fee55ef95e9/scipy/stats/_continuous_distns.py#L2554
    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.
    """
    n = 0
    xts = 0
    logs = 0

    for xx in x:
        if xx > 0:
            xts += xx
            logs += log(xx)
            n += 1

    if n == 0:
        return (0, 0)

    xtsbar = xts / n
    s = log(xtsbar) - (logs / n)

    if s == 0:
        return (0, 0)

    a_est = (3 - s + sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    xa = a_est * (1 - 0.4)
    xb = a_est * (1 + 0.4)
    a = brentq(xa, xb, s)
    if a == 0:
        return (0, 0)

    b = xtsbar / a

    return (a, b)


@numba.njit
def spifun(x, a=None, b=None, cal_start=None, cal_stop=None):
    """Calculate SPI with gamma distribution for 3d array."""
    y = np.full_like(x, -9999)
    r, c, t = x.shape

    if cal_start is None:
        cal_start = 0

    if cal_stop is None:
        cal_stop = t

    cal_ix = np.arange(cal_start, cal_stop)

    for ri in range(r):
        for ci in range(c):

            xt = x[ri, ci, :]
            valid_ix = []

            for tix in range(t):
                if xt[tix] > 0:
                    valid_ix.append(tix)

            n_valid = len(valid_ix)

            p_zero = 1 - (n_valid / t)

            if p_zero > 0.9:
                y[ri, ci, :] = -9999
                continue

            if a is None or b is None:
                alpha, beta = gammafit(xt[cal_ix])
            else:
                alpha, beta = (a, b)

            if alpha == 0 or beta == 0:
                y[ri, ci, :] = -9999
                continue

            spi = np.full(t, p_zero, dtype=nt.float64)  # type: ignore

            for tix in valid_ix:
                spi[tix] = p_zero + (
                    (1 - p_zero)
                    * sc.gammainc(alpha, xt[tix] / beta)  # pylint: disable=no-member
                )

            for tix in range(t):
                spi[tix] = sc.ndtri(spi[tix]) * 1000

            np.round_(spi, 0, spi)

            y[ri, ci, :] = spi[:]

    return y
