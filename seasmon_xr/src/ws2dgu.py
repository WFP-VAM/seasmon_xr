"""Whittaker smoother with fixed lambda (S)."""
import numpy
from numba import guvectorize
from numba.core.types import float64, int16

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, int16[:])], "(n),(),() -> (n)", nopython=True
    )
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
    if lmda != 0.0:

        m = y.shape[0]
        w = numpy.zeros(y.shape, dtype=float64)  # type: ignore

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
    else:
        out[:] = y[:]
