"""Convenience functions for the dsigma pipeline."""

import numpy as np
from scipy.interpolate import make_interp_spline

from astropy import units as u

__all__ = ['interpolate_over_redshift', 'in_degrees', 'spherical_to_cartesian']


def interpolate_over_redshift(f, z, *args, **kwargs):
    """Interpolate a function over redshift.

    For many cosmological calculations such as comoving distances performing
    the precise calculation for all objects in the catalog would be very
    expensive. Instead, we can perform the calculation on a grid in redshift
    and then interpolate. For most calculations, this is extremely accurate
    and much faster.

    Parameters
    ----------
    f : callable
        Function to evaluate over redshift.
    z : numpy.ndarray
        Redshifts to evaluate.
    *args
        Additional arguments passed to `f`.
    **kwargs
        Extra keyword arguments passed to `f`.

    Returns
    -------
    f_of_z : numpy.ndarray or astropy.units.quantity.Quantity
        Interpolated values.

    """
    z_min, z_max = np.amin(z), np.amax(z)

    if z_min == z_max:
        return np.repeat(f(z[0], *args, **kwargs), len(z))

    a_support = np.linspace(1.0 / (1 + z_max), 1.0 / (1 + z_min), 10000)
    y_support = f(1 / a_support - 1, *args, **kwargs)

    # Interpolating the sorted array is faster.
    idx = np.argsort(z)
    y = make_interp_spline(a_support, y_support)(1.0 / (1 + z[idx]))
    y[idx] = y

    if isinstance(y_support, u.Quantity):
        y = y * y_support.unit

    return y


def in_degrees(angle):
    """Add a degree unit to an angle if it doesn't have a unit, yet.

    Parameters
    ----------
    angle : astropy.units.quantity.Quantity
        Angle with or without units.

    Returns
    -------
    angle : astropy.units.quantity.Quantity
        Angle with units.

    """
    if angle.unit == u.Unit(''):
        angle = u.Quantity(angle.value, u.deg, copy=False)
    return angle.to(u.deg)


def spherical_to_cartesian(ra, dec):
    """Convert spherical coordinates to Cartesian coordinates on a unit sphere.

    Parameters
    ----------
    ra : astropy.units.quantity.Quantity
        Right ascension.
    dec : astropy.units.quantity.Quantity
        Declination.

    Returns
    -------
    x, y, z : float or numpy.ndarray
        Cartesian coordinates.

    """
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    return x, y, z
