"""Whittaker smoother with asymmetric smoothing and fixed lambda (S)."""
import numpy
from numba import guvectorize
from numba.core.types import float64, int16

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, float64, int16[:])],
        "(n),(),(),() -> (n)",
        nopython=True,
    )
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
    if lmda != 0.0:

        w = numpy.zeros(y.shape, dtype=float64)  # type: ignore
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
    else:
        out[:] = y[:]
