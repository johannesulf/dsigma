"""Convenience functions for the dsigma pipeline."""

import numpy as np

__all__ = ['cartesian_to_spherical', 'spherical_to_cartesian']


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
