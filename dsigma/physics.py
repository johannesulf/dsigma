"""Physics functions for the dsigma pipeline."""

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


__all__ = ['mpc_per_degree', 'projection_angle', 'projection_angle_sin_cos',
           'critical_surface_density']

_sigma_crit_factor = (c.c**2 / (4 * np.pi * c.G)).to(u.Msun / u.pc).value


def mpc_per_degree(cosmology, z, comoving=False):
    """Estimate the angular scale in Mpc/degree at certain redshift.

    Parameters
    ----------
    cosmo : cosmology.Cosmology object
        Cosmology object from the `cosmology` package by Erin Sheldon.
    z : float or numpy array
        Redshift of the object.
    comoving : boolen
        Use comoving distance instead of physical distance when True.
        Default: False

    Returns
    -------
    float or numpy array
        Physical scale in unit of Mpc/degree.
    """

    if comoving:
        return (cosmology.comoving_transverse_distance(z).to(u.Mpc).value *
                np.deg2rad(1))

    return (cosmology.angular_diameter_distance(z).to(u.Mpc).value *
            np.deg2rad(1))


def projection_angle(ra_l, dec_l, ra_s, dec_s):
    """Calculate projection angle between lens and sources.

    Parameters
    ----------
    ra_l, dec_l : float or numpy array
        Coordinates of the lens galaxies in degrees.
    ra_s, dec_s : float or numpy array
        Coordinates of the source galaxies in degrees.

    Returns
    -------
    cos_2phi, sin_2phi : float or numpy array
        The :math:`\cos` and :math:`\sin` of :math:`2 \phi`, where
        :math:`\phi` is the angle measured from right ascension direction to a
        line connecting the lens and source galaxies.
    """

    # Convert everything into radians.
    ra_l, dec_l = np.deg2rad(ra_l), np.deg2rad(dec_l)
    ra_s, dec_s = np.deg2rad(ra_s), np.deg2rad(dec_s)

    # Calculate the tan(phi).
    mask = np.cos(dec_s) * np.sin(ra_s - ra_l) != 0

    if hasattr(mask, "__len__"):
        tan_phi = (
            (np.cos(dec_l) * np.sin(dec_s) - np.sin(dec_l) * np.cos(dec_s) *
             np.cos(ra_s - ra_l))[mask] /
            (np.cos(dec_s) * np.sin(ra_s - ra_l))[mask])

        cos_2phi = np.repeat(-1.0, len(mask))
        sin_2phi = np.repeat(0.0, len(mask))

        cos_2phi[mask] = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
        sin_2phi[mask] = 2.0 * tan_phi / (1.0 + tan_phi * tan_phi)
    elif mask:
        tan_phi = (
            (np.cos(dec_l) * np.sin(dec_s) - np.sin(dec_l) * np.cos(dec_s) *
             np.cos(ra_s - ra_l)) / (np.cos(dec_s) * np.sin(ra_s - ra_l)))

        cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
        sin_2phi = (2.0 * tan_phi / (1.0 + tan_phi * tan_phi))
    else:
        cos_2phi = -1
        sin_2phi = 0

    return cos_2phi, sin_2phi


def projection_angle_sin_cos(sin_ra_l, cos_ra_l, sin_dec_l, cos_dec_l,
                             sin_ra_s, cos_ra_s, sin_dec_s, cos_dec_s):
    """Calculate projection angle between lens and sources. This function is
    similar to projection_angle but is much faster if sin and cos of the angles
    have been computed before.

    Parameters
    ----------
    sin_ra_l, cos_ra_l, sin_dec_l, cos_dec_l : float or numpy array
        Coordinates of the lens galaxies.
    sin_ra_s, cos_ra_s, sin_dec_s, cos_dec_s : float or numpy array
        Coordinates of the source galaxies.

    Returns
    -------
    cos_2phi, sin_2phi : float or numpy array
        The :math:`\cos` and :math:`\sin` of :math:`2 \phi`, where
        :math:`\phi` is the angle measured from right ascension direction to a
        line connecting the lens and source galaxies.
    """

    # Use trigonometric identities.
    sin_ra_s_minus_ra_l = sin_ra_s * cos_ra_l - cos_ra_s * sin_ra_l
    cos_ra_s_minus_ra_l = cos_ra_s * cos_ra_l + sin_ra_s * sin_ra_l


    # Calculate the tan(phi).
    mask = cos_dec_s * sin_ra_s_minus_ra_l != 0

    if hasattr(mask, "__len__"):
        tan_phi = (
            (cos_dec_l * sin_dec_s - sin_dec_l * cos_dec_s *
             cos_ra_s_minus_ra_l)[mask] /
            (cos_dec_s * sin_ra_s_minus_ra_l)[mask])

        cos_2phi = np.repeat(-1.0, len(mask))
        sin_2phi = np.repeat(0.0, len(mask))

        cos_2phi[mask] = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
        sin_2phi[mask] = 2.0 * tan_phi / (1.0 + tan_phi * tan_phi)
    elif mask:
        tan_phi = (
            (cos_dec_l * sin_dec_s - sin_dec_l * cos_dec_s *
             cos_ra_s_minus_ra_l) / (cos_dec_s * sin_ra_s_minus_ra_l))

        cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
        sin_2phi = (2.0 * tan_phi / (1.0 + tan_phi * tan_phi))
    else:
        cos_2phi = -1
        sin_2phi = 0

    return cos_2phi, sin_2phi


def critical_surface_density(z_l, z_s,
                             cosmology=FlatLambdaCDM(H0=100, Om0=0.3),
                             comoving=False, d_l=None, d_s=None):
    """The critical surface density for a given lens and source redshift.

    Parameters
    ----------
    z_l : float or numpy array
        Redshift of lens.
    z_s : float or numpy array
        Redshift of source.
    cosmos : cosmology.Cosmology object
        Cosmology object from `cosmology` package by Erin Sheldon.
    comoving : boolean, optional
        Flag for using comoving instead of physical unit.
    d_l : float or numpy array
        Comoving transverse distance to the lens. If not given, it is
        calculated from the redshift provided.
    d_s : float or numpy array
        Comoving transverse distance to the source. If not given, it is
        calculated from the redshift provided.

    Returns
    -------
    float or numpy array
        Critical surface density for each lens-source pair.

    """

    if d_l is None:
        d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    if d_s is None:
        d_s = cosmology.comoving_transverse_distance(z_s).to(u.Mpc).value

    dist_term = (1e-6 * (d_s / (1 + z_s)) / (d_l / (1 + z_l)) /
                 (np.where(d_s > d_l, d_s - d_l, 1) / (1 + z_s)))
    dist_term[d_s <= d_l] = np.inf

    if comoving:
        dist_term /= (1.0 + z_l)**2

    return _sigma_crit_factor * dist_term
