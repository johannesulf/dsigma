"""Convenience functions for the dsigma pipeline."""

import numpy as np
from scipy.interpolate import make_interp_spline

from astropy.units.quantity import Quantity

__all__ = ['interpolate_over_redshift', 'cartesian_to_spherical',
           'spherical_to_cartesian']


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

    Raises
    ------
    ValueError
        If redshifts are negative.

    """
    if np.any(z < 0):
        msg = "Redshifts cannot be negative."
        raise ValueError(msg)

    z_unique, unique_inverse = np.unique(z, return_inverse=True)
    if len(z_unique) <= 10000:
        y = f(z_unique, *args, **kwargs)[unique_inverse]
    else:
        a_support = np.linspace(1.0 / (1 + np.amax(z)), 1.0 / (1 + np.amin(z)),
                                10000)
        y_support = f(1 / a_support - 1, *args, **kwargs)
        y = make_interp_spline(a_support, y_support)(1.0 / (1 + z))
    
        if isinstance(y_support, Quantity):
            y = y * y_support.unit

    return y


def spherical_to_cartesian(ra, dec):
    """Convert spherical coordinates to cartesian coordinates on a unit sphere.

    Parameters
    ----------
    ra : float or numpy.ndarray
        Right ascension.
    dec : float or numpy.ndarray
        Declination.

    Returns
    -------
    x, y, z : float or numpy.ndarray
        Cartesian coordinates.

    """
    x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    z = np.sin(np.deg2rad(dec))
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : float or numpy.ndarray
        x-coordinate.
    y : float or numpy.ndarray
        y-coordinate.
    z : float or numpy.ndarray
        z-coordinate.

    Returns
    -------
    ra, dec : float or numpy.ndarray
        Spherical coordinates.

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    ra = np.arctan2(y, x)
    ra = np.where(ra < 0, ra + 2 * np.pi, ra)
    dec = np.arcsin(z / r)
    return np.rad2deg(ra), np.rad2deg(dec)
