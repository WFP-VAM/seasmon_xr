"""Lag-1 autocorrelations."""
from numba import njit
from numba.core.types import float32, float64, int64
from numpy import isnan, zeros

from ._helper import lazycompile

# pyright: reportGeneralTypeIssues=false


@njit
def autocorr_1d_float(data):
    """
    Lag 1 autocorrelation on float data with nan values.

    X = data[:-1]
    Y = data[1:]

    mean( (X - X.mean()) * (Y - Y.mean()) )
    ---------------------------------------
              X.std()*Y.std()

    Args:
        data ([type]): Integer data with missing values
    """
    assert data.ndim == 1
    xx = data[:-1]
    yy = data[1:]

    N = xx.shape[0]

    # ((X-X.mean())*(Y-Y.mean())).mean()
    Sxy = float64(0)  # Sum(Xi*Yi) when Xi and Yi are both valid
    Sx_ = float64(0)  # Sum(Xi)   when Xi is valid and Yi is valid
    Sy_ = float64(0)  # Sum(Yi)   when Xi is valid and Yi is valid
    nxy = float64(0)  # number of valid Xi,Yi tuples (both non-nodata)

    # var(X)
    Sx = float64(0)  # Sum(Xi)    when Xi is valid
    Sxx = float64(0)  # Sum(Xi*Xi) when Xi is valid
    nx = float64(0)  # number of valid Xi

    # var(Y)
    Sy = float64(0)  # Sum(Yi)    when Yi is valid
    Syy = float64(0)  # Sum(Yi*Yi) when Yi is valid
    ny = float64(0)  # number of valid Yi

    for i in range(N):
        x = float64(xx[i])
        y = float64(yy[i])

        x_ok = not isnan(x)
        y_ok = not isnan(y)

        if x_ok:
            Sx += x
            Sxx += x * x
            nx += 1

        if y_ok:
            Sy += y
            Syy += y * y
            ny += 1

        if x_ok and y_ok:
            Sx_ += x
            Sy_ += y
            Sxy += x * y
            nxy += 1

    result = float64(0.0)  # or should this be nan?
    if nxy == 0:
        return result

    A = nxy * Sxy - Sx_ * Sy_

    # var(X[np.isfinite(X)]) Vairance of X excluding missing values
    var_X = nx * Sxx - Sx * Sx
    var_Y = ny * Syy - Sy * Sy

    # var(X) where missing values were replaced with mean,
    #   i.e. X[X==nodata] = mean(X[X!=nodata])
    var_X = var_X * nx / N
    var_Y = var_Y * ny / N

    if var_X < 1e-8 or var_Y < 1e-8:
        return result

    result = A * (var_X**-0.5) * (var_Y**-0.5)
    return result


@njit
def autocorr_1d_int(data, nodata):
    """
    Lag 1 autocorrelation on integer data with nodata marker.

    X = data[:-1]
    Y = data[1:]

    mean( (X - X.mean()) * (Y - Y.mean()) )
    ---------------------------------------
              std(X)*std(Y)

    Args:
        data ([type]): Integer data with missing values
        nodata ([type]): Missing value marker
    """
    assert data.ndim == 1
    assert data.dtype.kind in ("i", "u")
    xx = data[:-1]
    yy = data[1:]

    N = xx.shape[0]

    #   ((X - X.mean())*(Y - Y.mean())).mean()
    # =  Sxy*nxy - Sx_*Sy_
    Sx_ = int64(0)  # Sum(Xi)    when Xi is valid and Yi is valid
    Sy_ = int64(0)  # Sum(Yi)    when Xi is valid and Yi is valid
    Sxy = int64(0)  # Sum(Xi*Yi) when Xi and Yi are both valid
    nxy = int64(0)  # number of valid Xi,Yi tuples (both non-nodata)

    # var(X) = Sxx*nx - Sx*Sx
    Sx = int64(0)  # Sum(Xi)    when Xi is valid
    Sxx = int64(0)  # Sum(Xi*Xi) when Xi is valid
    nx = int64(0)  # number of valid Xi

    # var(Y) = Syy*ny - Sy*Sy
    Sy = int64(0)  # Sum(Yi)    when Yi is valid
    Syy = int64(0)  # Sum(Yi*Yi) when Yi is valid
    ny = int64(0)  # number of valid Yi

    for i in range(N):
        x = xx[i]
        y = yy[i]

        if x != nodata:
            Sx += x
            Sxx += x * x
            nx += 1

        if y != nodata:
            Sy += y
            Syy += y * y
            ny += 1

        if x != nodata and y != nodata:  # pylint: disable=consider-using-in
            Sx_ += x
            Sy_ += y
            Sxy += x * y
            nxy += 1

    result = float64(0.0)  # or should this be nan?
    if nxy == 0:
        return result

    A = nxy * float64(Sxy) - float64(Sx_) * float64(Sy_)

    # var(X[np.isfinite(X)]) Vairance of X excluding missing values
    var_X = nx * float64(Sxx) - float64(Sx) * float64(Sx)
    var_Y = ny * float64(Syy) - float64(Sy) * float64(Sy)

    # var(X) where missing values were replaced with mean,
    #   i.e. X[X==nodata] = mean(X[X!=nodata])
    var_X = var_X * nx / N
    var_Y = var_Y * ny / N

    if var_X < 1e-8 or var_Y < 1e-8:
        return result

    result = A * (var_X**-0.5) * (var_Y**-0.5)
    return result


@njit
def autocorr_1d(data, nodata=None):
    """Calculate Lag-1 autocorrelation on 1-d inputs."""
    result = float64(0)
    if nodata is None:
        result = autocorr_1d_float(data)
    else:
        result = autocorr_1d_int(data, nodata)
    return result


@lazycompile(njit)
def autocorr(x, nodata=None):
    """
    Calculate Lag-1 autocorrelation.

    Args:
        x: 3d data array (Y,X,T)
    Returns:
        Lag-1 autocorrelation array (Y,X)
    """
    r, c, _ = x.shape
    z = zeros((r, c), dtype="float32")

    for rr in range(r):
        for cc in range(c):
            data = x[rr, cc, :]
            z[rr, cc] = autocorr_1d(data, nodata)

    return z


@lazycompile(njit)
def autocorr_tyx(tyx, nodata=None):
    """
    Calculate Lag-1 autocorrelation.

    Args:
        tyx: 3d data array in T,Y,X order
    Returns:
        Lag-1 autocorrelation array Y,X order
    """
    _, nr, nc = tyx.shape
    z = zeros((nr, nc), dtype=float32)

    for rr in range(nr):
        for cc in range(nc):
            data = tyx[:, rr, cc]
            z[rr, cc] = autocorr_1d(data, nodata)

    return z
