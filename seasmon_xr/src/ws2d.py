"""Whittaker filter with differences of 2nd order for a 1d array."""
from numba import njit
from numpy import zeros


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
    z = zeros(n)
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
