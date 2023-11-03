'''
Module based on numpy polyval and polyfit functions.
Used to returns coefficients of polynomial of a specific degree 

'''

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np


__all__ = ['polyval2d', 'polygrid2d', 'polyfit2d']


class RankWarning(UserWarning):
    pass


def _polyvander2d(x, y, deg):
    """Return the pseudo-Vandermonde matrix for a given degree.

[[1, x[0], y[0], ... , x[0]*y[0]^(n-1), y[0]^n]
[1, x[1], y[1], ... , x[1]*y[1]^(n-1), y[1]^n]
... ...
[1, x[M], y[M], ... , x[M]*y[M]^(n-1), y[M]^n]]

where `n` is `deg`.

Parameters
----------
x, y : array_like, shape (M,)
Array of points. `x` and `y` must have the same shape.
deg : int
Degree of the resulting matrix.

Returns
-------
v : ndarray
The Vandermonde matrix. The shape of the matrix is ``x.shape +
((deg+1)*(deg+2) // 2, ))``.
"""
    x = np.array(x, copy=False, ndmin=1) + 0.0
    y = np.array(y, copy=False, ndmin=1) + 0.0
    if x.ndim != 1:
        raise ValueError("x must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("y must be 1-dim.")

    dims = ((deg+1)*(deg+2) // 2, ) + x.shape
    v = np.empty(dims, dtype=x.dtype)
    v[0] = x * 0 + 1.0
    i = 1
    for j in range(1, deg+1):
        v[i:i+j] = x * v[i-j:i]
        v[i+j] = y * v[i-1]
        i += j + 1
    return np.rollaxis(v, 0, v.ndim)


def polyfit2d(x, y, z, deg=1, rcond=None, full_output=False):
    """Return the coefficients of a polynomial of degree `deg`.

The coefficients are determined by the least square fit to given data values
`z` at given points ``(x, y)``. The fitting assumes the polynomial in a
form::

.. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

with a constraint of ``i + j <= n`` where `n` is `deg`.

Parameters
----------
x, y : array_like, shape (M,)
x- and y-oordinates of the M data points ``(x[i], y[i])``.
z : array_like, shape (M,)
z-coordinates of the M data points.
deg : int, optional
Degree of the polynomial to be fit.
rcond : float, optional
Relative condition of the fit. Singular values smaller than
`rcond`, relative to the largest singular value, will be
ignored. The default value is ``len(x)*eps``, where `eps` is
the relative precision of the platform's float type, about 2e-16
in most cases.
full_output : {True, False}, optional
Just the coefficients are returned if False, and diagnostic
information from the SVD is also returned if True.

Returns
-------
coef : ndarray
Array of coefficients.
[residuals, rank, singular_values, rcond] : if `full_output` = True
Sum of the squared residuals of the least-squares fit; the
effective rank of the scaled pseudo-Vandermonde matrix; its
singular values, and the specified value of `rcond`. For more
details, see `numpy.linalg.lstsq`.

Warns
-----
RankWarning
The rank of the coefficient matrix in the least-squares fit is
deficient. The warning is only raised if `full_output` = False.

See Also
--------
polyval2d, polygrid2d
"""
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    z = np.asarray(z) + 0.0

    deg = int(deg)
    if deg < 1:
        raise ValueError("deg must be larger than 1.")

    # Check inputs.
    if x.ndim != 1:
        raise ValueError("x must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("y must be 1-dim.")
    if z.ndim != 1:
        raise ValueError("z must be 1-dim.")
    if x.size != y.size or x.size != z.size:
        raise ValueError("x, y, and z must have the same size.")

    # Set up the matrices for the problem in transposed form.
    lhs = _polyvander2d(x, y, deg).T
    rhs = z.T

    # Set rcond.
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps

    # Determine the norms of the design maxtirx columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c1, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
    c1 = (c1.T / scl).T

    # Warn on rank reduction.
    if rank != lhs.shape[0] and not full_output:
        msg = "The fit may be poorly conditioned."
        warnings.warn(msg, RankWarning)

    # Allocate the coefficients in a 2-dim array.
    inds = []
    for m in range(deg + 1):
        for j in range(m + 1):
            for i in range(m + 1):
                if i + j != m:
                    continue
                inds.append((i, j))
    c2 = np.zeros((deg + 1, deg + 1))
    indices = list(zip(*inds))
    c2[tuple(indices)] = c1

    if full_output:
        return c2, [resids, rank, s, rcond]
    else:
        return c2


def polygrid2d(x, y, c):
    """
Evaluate a 2-D polynomial on the Cartesian product of x and y.

This function returns the values:

.. math:: p(a,b) = \sum_{i,j} c_{i,j} * a^i * b^j

where the points `(a, b)` consist of all pairs formed by taking
`a` from `x` and `b` from `y`. The resulting points form a grid with
`x` in the first dimension and `y` in the second.

The parameters `x` and `y` are converted to arrays only if they are
tuples or a lists, otherwise they are treated as a scalars. In either
case, either `x` and `y` or their elements must support multiplication
and addition both with themselves and with the elements of `c`.

If `c` has fewer than two dimensions, ones are implicitly appended to
its shape to make it 2-D. The shape of the result will be c.shape[2:] +
x.shape + y.shape.

Parameters
----------
x, y : array_like, compatible objects
The two dimensional series is evaluated at the points in the
Cartesian product of `x` and `y`. If `x` or `y` is a list or
tuple, it is first converted to an ndarray, otherwise it is left
unchanged and, if it isn't an ndarray, it is treated as a scalar.
c : array_like
Array of coefficients ordered so that the coefficients for terms of
degree i,j are contained in ``c[i,j]``. If `c` has dimension
greater than two the remaining indices enumerate multiple sets of
coefficients.

Returns
-------
values : ndarray, compatible object
The values of the two dimensional polynomial at points in the Cartesian
product of `x` and `y`.

See Also
--------
polyval2d, polyfit2d
numpy.polynomial.polynomial.polygrid2d
"""
    from numpy.polynomial.polynomial import polygrid2d

    c = np.asarray(c)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("c must be a squard 2-dim array.")
    return polygrid2d(x, y, c)


def polyval(x, c, tensor=True):
    c = np.array(c, ndmin=1, copy=0)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    c0 = c[-1] + x*0
    for i in range(2, len(c) + 1) :
        c0 = c[-i] + c0*x
    return c0


def polyval2d(x, y, c):
    """Evaluate a 2-D polynomial at points (x, y).

This function returns the value

.. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

Parameters
----------
x, y : array_like, compatible objects
The two dimensional series is evaluated at the points (x, y), where x
and y must have the same shape. If x or y is a list or tuple, it is
first converted to an ndarray, otherwise it is left unchanged and, if it
is not an ndarray, it is treated as a scalar.
c : array_like
Array of coefficients ordered so that the coefficient of the term of
multi-degree i,j is contained in `c[i,j]`.

Returnes
--------
values : ndarray, compatible object
The values of the two dimensional polynomial at points formed with pairs
of corresponding values from x and y.

See Also
--------
polygrid2d, polyfit2d
numpy.polynomial.polynomial.polyval2d
"""

    try:
        x, y = np.array((x, y), copy=0)
    except:
        raise ValueError('x, y are incompatible')

    c = polyval(x, c)
    c = polyval(y, c, tensor=False)
    return c