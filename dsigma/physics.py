"""Physics functions for the dsigma pipeline."""

import numpy as np
from astropy import constants as c
from astropy import units as u
from scipy.special import jv, jn_zeros
from astropy.cosmology import FlatLambdaCDM


__all__ = ['mpc_per_degree', 'projection_angle', 'critical_surface_density',
           'effective_critical_surface_density',
           'lens_magnification_shear_bias']

_sigma_crit_factor = (c.c**2 / (4 * np.pi * c.G)).to(u.Msun / u.pc).value


def mpc_per_degree(z, cosmology=FlatLambdaCDM(H0=100, Om0=0.3),
                   comoving=False):
    """Estimate the angular scale in Mpc/degree at certain redshift.

    Parameters
    ----------
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations.
    z : float or numpy.ndarray
        Redshift of the object.
    comoving : boolen
        Use comoving distance instead of physical distance when True.
        Default: False

    Returns
    -------
    float or numpy.ndarray
        Physical scale in unit of Mpc/degree.

    """
    if comoving:
        return (cosmology.comoving_transverse_distance(z).to(u.Mpc).value *
                np.deg2rad(1))

    return (cosmology.angular_diameter_distance(z).to(u.Mpc).value *
            np.deg2rad(1))


def projection_angle(ra_l, dec_l, ra_s, dec_s):
    r"""Calculate projection angle between lens and sources.

    Parameters
    ----------
    ra_l, dec_l : float or numpy.ndarray
        Coordinates of the lens galaxies in degrees.
    ra_s, dec_s : float or numpy.ndarray
        Coordinates of the source galaxies in degrees.

    Returns
    -------
    cos_2phi, sin_2phi : float or numpy.ndarray
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


def critical_surface_density(z_l, z_s, cosmology=None, comoving=True, d_l=None,
                             d_s=None):
    """Compute the critical surface density.

    Parameters
    ----------
    z_l : float or numpy.ndarray
        Redshift of lens.
    z_s : float or numpy.ndarray
        Redshift of source.
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations. Only used if comoving distances
        are not passed.
    comoving : bool, optional
        Flag for using comoving instead of physical units.
    d_l : float or numpy.ndarray
        Comoving transverse distance to the lens. If not given, it is
        calculated from the redshift provided.
    d_s : float or numpy.ndarray
        Comoving transverse distance to the source. If not given, it is
        calculated from the redshift provided.

    Returns
    -------
    sigma_crit : float or numpy.ndarray
        Critical surface density for each lens-source pair.

    """
    if d_l is None:
        d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    if d_s is None:
        d_s = cosmology.comoving_transverse_distance(z_s).to(u.Mpc).value

    dist_term = (1e-6 * (d_s / (1 + z_s)) / (d_l / (1 + z_l)) /
                 (np.where(d_s > d_l, d_s - d_l, 1) / (1 + z_s)))

    if np.isscalar(dist_term):
        if d_s <= d_l:
            dist_term = np.inf
    else:
        dist_term[d_s <= d_l] = np.inf

    if comoving:
        dist_term /= (1.0 + z_l)**2

    return _sigma_crit_factor * dist_term


def effective_critical_surface_density(z_l, z_s, n_s, cosmology,
                                       comoving=True):
    """Compute the effective critical surface density.

    Parameters
    ----------
    z_l : float or numpy.ndarray
        Redshift of lens.
    z_s : numpy.ndarray
        Redshifts of sources.
    n_s : numpy.ndarray
        Fraction of source galaxies in each redshift bin. Does not need to be
        normalized.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.
    comoving : boolean, optional
        Flag for using comoving instead of physical unit.

    Returns
    -------
    sigma_crit_eff : float or numpy.ndarray
        Effective critical surface density for the lens redshift given the
        source redshift distribution.

    """
    d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(z_s).to(u.Mpc).value

    if not np.isscalar(z_l):
        z_l = np.repeat(z_l, len(z_s)).reshape((len(z_l), len(z_s)))
        d_l = np.repeat(d_l, len(z_s)).reshape(z_l.shape)
        z_s = np.tile(z_s, len(z_l)).reshape(z_l.shape)
        d_s = np.tile(d_s, len(z_l)).reshape(z_l.shape)
        n_s = np.tile(n_s, len(z_l)).reshape(z_l.shape)

    sigma_crit = critical_surface_density(z_l, z_s, cosmology=cosmology,
                                          comoving=comoving, d_l=d_l, d_s=d_s)

    if not np.isscalar(z_l):
        sigma_crit_eff = np.repeat(np.inf, len(z_l))
        mask = np.average(sigma_crit**-1, axis=-1, weights=n_s) == 0
        sigma_crit_eff[~mask] = np.average(sigma_crit**-1, axis=-1,
                                           weights=n_s)[~mask]**-1
        return sigma_crit_eff
    else:
        if np.average(sigma_crit**-1, weights=n_s) > 0:
            return np.average(sigma_crit**-1, weights=n_s)**-1
        else:
            return np.inf


def lens_magnification_shear_bias(theta, alpha_l, z_l, z_s, camb_results,
                                  n_z=10, n_ell=200, bessel_function_zeros=100,
                                  k_max=1e3):
    r"""Compute the lens magnification bias to the mean tangential shear.

    This function is based on equations (13) and (14) in Unruh et al. (2020).

    Parameters
    ----------
    theta : float or astropy.units.quantity.Quantity
        Angular separation :math:`\theta` from the lens sample. If not quantity
        is given, the separation is assumed to be in radians.
    alpha_l : float
        Local slope of the flux distribution of lenses near the flux limit.
    z_l : float
        Redshift of lens.
    z_s : float
        Redshift of source.
    camb_results : camb.results.CAMBdata
        CAMB results object that contains information on cosmology and the
        matter power spectrum.
    n_z : int, optional
        Number of redshift bins used in the integral. Larger numbers will be
        more accurate.
    n_ell : int, optional
        Number of :math:`\ell` bins used in the integral. Larger numbers will
        be more accurate.
    bessel_function_zeros : int, optional
        The calculation involves an integral over the second order Bessel
        function :math:`J_2 (\ell \theta)` from :math:`\ell = 0` to
        :math:`\ell = \infty`. In practice, this function replaces the upper
        bound with the bessel_function_zeros-th zero point of the Bessel
        function. Larger number should lead to more accurate results. However,
        in practice, this also requires larger `n_ell`. Particularly, `n_ell`
        should never fall below `bessel_function_zeros`.
    k_max : float, optional
        The maximum wavenumber beyond which the power spectrum is assumed to be
        0.

    Returns
    -------
    et_lm : float
        Bias in the mean tangential shear due to lens magnification effects.

    """
    camb_interp = camb_results.get_matter_power_interpolator(
        hubble_units=False, k_hunit=False)

    if not isinstance(theta, u.quantity.Quantity):
        theta = theta * u.rad

    theta = theta.to(u.rad).value

    ell_min = 0
    ell_max = np.amax(jn_zeros(2, bessel_function_zeros)) / theta
    z_min = 0
    z_max = min(z_l, z_s)

    z, w_z = np.polynomial.legendre.leggauss(n_z)
    z = (z_max - z_min) / 2.0 * z + (z_max + z_min) / 2.0
    w_z = w_z * (z_max - z_min) / 2.0

    ell, w_ell = np.polynomial.legendre.leggauss(n_ell)
    ell = (ell_max - ell_min) / 2.0 * ell + (ell_max + ell_min) / 2.0
    w_ell = w_ell * (ell_max - ell_min) / 2.0

    int_z = np.array([
        (1 + z_i)**2 / (2 * np.pi) *
        camb_results.hubble_parameter(0) /
        camb_results.hubble_parameter(z_i) *
        camb_results.angular_diameter_distance2(z_i, z_l) *
        camb_results.angular_diameter_distance2(z_i, z_s) /
        camb_results.angular_diameter_distance(z_l) /
        camb_results.angular_diameter_distance(z_s) for z_i in z])

    d_ang = np.array([
        camb_results.angular_diameter_distance(z_i) for z_i in z])
    z = np.tile(z, n_ell)
    int_z = np.tile(int_z, n_ell)
    d_ang = np.tile(d_ang, n_ell)
    w_z = np.tile(w_z, n_ell)

    int_ell = ell * jv(2, ell * theta)
    ell = np.repeat(ell, n_z)
    int_ell = np.repeat(int_ell, n_z)
    w_ell = np.repeat(w_ell, n_z)

    k = (ell + 0.5) / ((1 + z) * d_ang)

    int_z_ell = np.array([camb_interp.P(z[i], k[i]) for i in range(len(k))])
    int_z_ell = np.where(k > k_max, 0, int_z_ell)

    gamma = np.sum(int_z * int_ell * int_z_ell * w_z * w_ell)
    gamma = ((gamma * u.Mpc**3) * 9 * camb_results.Params.H0**3 * u.km**3 /
             u.s**3 / u.Mpc**3 *
             (camb_results.Params.omch2 + camb_results.Params.ombh2)**2 /
             (camb_results.Params.H0 / 100)**4 / 4 / c.c**3)

    return 2 * (alpha_l - 1) * gamma.to(u.dimensionless_unscaled).value
