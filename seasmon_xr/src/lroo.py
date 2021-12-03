"""Calculate the longest run of ones inside a 1d array."""
import numpy
from numba import guvectorize

from ._helper import lazycompile


@lazycompile(guvectorize("(uint8[:], uint8[:])", "(n) -> ()", nopython=True))
def lroo(data, out):
    """
    Calculate the longest run of ones.

    Intended to be used with xarray and apply_ufunc
    """
    cr = 1
    mr = 0
    dots = numpy.where(data.flatten() == 1)[0]

    for ix in range(1, dots.size):

        d = dots[ix] - dots[ix - 1]
        if d == 1:
            cr += 1
            if cr > mr:
                mr = cr
        else:
            cr = 1

    if mr > 1:
        out[0] = mr
    else:
        out[0] = 0
