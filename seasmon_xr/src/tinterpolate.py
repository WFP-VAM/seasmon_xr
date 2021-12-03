"""Interpolation numba functions."""
from numba import guvectorize
from numba.core.types import float64, int16, int32, uint8

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(int16[:], float64[:], int32[:], uint8[:], int16[:])],
        "(n),(m),(m),(l) -> (l)",
        nopython=True,
    )
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
