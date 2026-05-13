"""Physics functions for the dsigma pipeline."""

from functools import partial

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import units as cu
from scipy.special import jn_zeros, jv

from . import default_cosmology

__all__ = ['critical_surface_density', 'effective_critical_surface_density',
           'lens_magnification_shear_bias', 'mpc_per_degree']


def mpc_per_degree(z, cosmology=None, comoving=False):
    r"""Calculate the conversion factor between angular and physical scales.

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift of the object.
    cosmology : astropy.cosmology or None, optional
        Cosmology to use for calculation. If ``None``, use
        ``dsigma.default_cosmology``. Default is ``None``.
    comoving : bool, optional
        Use comoving distance instead of physical distance when True.
        Default is ``False``.

    Returns
    -------
    factor : astropy.units.quantity.Quantity
        Conversion factor.

    """
    cosmology = default_cosmology if cosmology is None else cosmology

    if comoving:
        d = cosmology.comoving_transverse_distance(z)
    else:
        d = cosmology.angular_diameter_distance(z)

    d = d.to(u.Mpc / cu.littleh, cu.with_H0(cosmology.H0))

    return (d / u.rad).to(u.Mpc / cu.littleh / u.deg)


def critical_surface_density(
        z_l, z_s, cosmology=None, comoving=True, d_l=None, d_s=None):
    """Compute the critical surface density.

    Parameters
    ----------
    z_l : float or numpy.ndarray
        Redshift of lens.
    z_s : float or numpy.ndarray
        Redshift of source.
    cosmology : astropy.cosmology or None, optional
        Cosmology to assume for calculations. Only used if comoving distances
        are not passed. If ``None``, use ``dsigma.default_cosmology``. Default
        is ``None``.
    comoving : bool, optional
        Flag for using comoving instead of physical units.
    d_l : astropy.units.quantity.Quantity
        Comoving transverse distance to the lens. If not given, it is
        calculated from the redshift provided.
    d_s : astropy.units.quantity.Quantity
        Comoving transverse distance to the source. If not given, it is
        calculated from the redshift provided.

    Returns
    -------
    sigma_crit : astropy.units.quantity.Quantity
        Critical surface density for each lens-source pair.

    """
    cosmology = default_cosmology if cosmology is None else cosmology

    if d_l is None:
        d_l = cosmology.comoving_transverse_distance(z_l)
    if d_s is None:
        d_s = cosmology.comoving_transverse_distance(z_s)

    with np.errstate(divide='ignore'):
        sigma_crit = c.c**2 / (4 * np.pi * c.G) * (
            (d_s / (1 + z_s)) / (d_l / (1 + z_l)) / ((d_s - d_l) / (1 + z_s)))
    sigma_crit = np.where(d_s <= d_l, np.inf * sigma_crit.unit, sigma_crit)

    if comoving:
        sigma_crit /= (1.0 + z_l)**2

    return sigma_crit.to(
        cu.littleh * u.Msun / u.pc**2, cu.with_H0(cosmology.H0))


def effective_critical_surface_density(
        z_l, z_s, n_s, cosmology=None, comoving=True):
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
    cosmology : astropy.cosmology or None, optional
        Cosmology to assume for calculations. If ``None``, use
        ``dsigma.default_cosmology``. Default is ``None``.
    comoving : boolean, optional
        Flag for using comoving instead of physical unit.

    Returns
    -------
    sigma_crit_eff : astropy.units.quantity.Quantity
        Effective critical surface density for the lens redshift given the
        source redshift distribution. Has the same length as `z_l`.

    """
    cosmology = default_cosmology if cosmology is None else cosmology
    d_l = cosmology.comoving_transverse_distance(z_l)
    d_s = cosmology.comoving_transverse_distance(z_s)

    if hasattr(z_l, '__len__'):
        z_l = np.repeat(z_l, len(z_s)).reshape((len(z_l), len(z_s)))
        d_l = np.repeat(d_l, len(z_s)).reshape(z_l.shape)
        z_s = np.tile(z_s, len(z_l)).reshape(z_l.shape)
        d_s = np.tile(d_s, len(z_l)).reshape(z_l.shape)
        n_s = np.tile(n_s, len(z_l)).reshape(z_l.shape)

    sigma_crit_eff_inv = 1.0 / critical_surface_density(
        z_l=z_l, d_l=d_l, z_s=z_s, d_s=d_s, comoving=comoving)
    sigma_crit_eff_inv = np.average(sigma_crit_eff_inv, weights=n_s, axis=-1)

    with np.errstate(divide='ignore'):
        return 1.0 / sigma_crit_eff_inv


def _to_camb(cosmology, sigma_8, n_s, z):
    r"""Convert an astropy cosmology object into a CAMB result object.

    Parameters
    ----------
    cosmology : astropy.cosmology.FlatLambdaCDM
        Astropy cosmology.
    sigma_8 : float
        Scale of fluctations at :math:`8 h^{-1} \, \mathrm{Mpc}`.
    n_s : float
        Primordial power spectrum index. Default is 0.96.
    z : numpy.ndarray
        Redshifts for which to compute the power spectrum.

    Raises
    ------
    ValueError
        If cosmology is not instance of ``astropy.cosmology.FlatLambdaCDM``.

    Returns
    -------
    results : camb.results.CAMBdata
        CAMB results object that contains information about the matter power
        spectrum.

    """
    if not isinstance(cosmology, FlatLambdaCDM):
        msg = "Cosmology must be instance of astropy.cosmology.FlatLambdaCDM."
        raise ValueError(msg)

    import camb

    h = cosmology.H0.to(u.km / u.s / u.Mpc).value / 100
    a_s = 2e-9  # initial guess
    m_nu = cosmology.m_nu.to(u.eV).value

    # TODO: Somebody should check this, especially in regards to neutrinos.
    pars = camb.set_params(
        H0=100 * h, omch2=(cosmology.Om0 - cosmology.Ob0) * h**2,
        ombh2=cosmology.Ob0 * h**2, omnuh2=cosmology.Onu0 * h**2,
        TCMB=cosmology.Tcmb0.to(u.K).value,
        num_nu_massless=cosmology.Neff - np.sum(m_nu > 0),
        num_nu_massive=np.sum(m_nu > 0),
        nu_mass_eigenstates=len(np.unique(m_nu[m_nu > 0])),
        nu_mass_numbers=np.unique(m_nu[m_nu > 0], return_counts=True)[1],
        nu_mass_degeneracies=np.unique(m_nu[m_nu > 0]),
        ns=n_s, As=a_s, NonLinear='NonLinear_pk', kmax=1000.0, redshifts=z)

    results = camb.get_results(pars)

    # Iterate to get the sigma_8 value correct.
    while np.abs(np.log(sigma_8 / results.get_sigma8_0())) > 1e-9:
        a_s *= (sigma_8 / results.get_sigma8_0())**2
        pars.InitPower.set_params(ns=n_s, As=a_s)
        results = camb.get_results(pars)

    return results


def _gaussian_quadrature_2d(f, n_x, x_min, x_max, n_y, y_min, y_max):
    """Integrate a two-dimensional function using Gaussian quadrature.

    Parameters
    ----------
    f : callable
        Function to integrate.
    n_x : int
        Number of points in the x-dimension.
    x_min : float
        Lower integration limit for x.
    x_max : float
        Upper integration limit for x.
    n_y : int
        Number of points in the y-dimension.
    y_min : float
        Lower integration limit for y.
    y_max : float
        Upper integration limit for y.

    Returns
    -------
    integral : float
        Computed integral.

    """
    x, w_x = np.polynomial.legendre.leggauss(n_x)
    x = (x_max - x_min) / 2.0 * x + (x_max + x_min) / 2.0
    w_x = w_x * (x_max - x_min) / 2.0

    y, w_y = np.polynomial.legendre.leggauss(n_y)
    y = (y_max - y_min) / 2.0 * y + (y_max + y_min) / 2.0
    w_y = w_y * (y_max - y_min) / 2.0

    x, y = np.meshgrid(x, y)
    z = f(x.ravel(), y.ravel())
    w_z = np.outer(w_x, w_y).T.ravel()

    return np.sum(z * w_z)


def lens_magnification_shear_bias(
        theta, alpha_l, z_l, z_s, cosmology=None, sigma_8=0.82, n_s=0.96,
        n_z=10, n_ell=1000, bessel_function_zeros=100, k_max=1e3):
    r"""Compute the lens magnification bias to the mean tangential shear.

    This function is based on equations (13) and (14) in Unruh et al. (2020).

    Parameters
    ----------
    theta : float, numpy.ndarray or astropy.units.quantity.Quantity
        Angular separation :math:`\theta` from the lens sample. If it has no
        unit, assume the value is given in radians.
    alpha_l : float
        Local slope of the flux distribution of lenses near the flux limit.
    z_l : float
        Redshift of lens.
    z_s : float
        Redshift of source.
    cosmology : astropy.cosmology or None, optional
        Cosmology to assume for calculations. If ``None``, use
        ``dsigma.default_cosmology``. Default is ``None``.
    sigma_8 : float, optional
        Scale of fluctations at :math:`8 h^{-1} \, \mathrm{Mpc}`. Default is
        0.82.
    n_s : float, optional
        Primordial power spectrum index. Default is 0.96.
    n_z : int, optional
        Number of redshift bins used in the integral. Larger numbers will be
        more accurate. Default is 10.
    n_ell : int, optional
        Number of :math:`\ell` bins used in the integral. Larger numbers will
        be more accurate. Default is 200.
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
    gt : float or numpy.ndarray
        Bias in the mean tangential shear due to lens magnification effects.

    """
    cosmology = default_cosmology if cosmology is None else cosmology

    if not isinstance(theta, u.Quantity):
        theta *= u.rad
    theta = theta.to(u.rad).value

    r = _to_camb(cosmology, sigma_8, n_s, np.linspace(z_l, 0, 10))
    p = r.get_matter_power_interpolator(hubble_units=False, k_hunit=False).P
    d_l = cosmology.angular_diameter_distance(z_l)
    d_s = cosmology.angular_diameter_distance(z_s)

    def f(theta, z, ell):
        z_u, idx = np.unique(z, return_inverse=True)
        k = (ell + 0.5) / ((1 + z) * cosmology.angular_diameter_distance(
            z_u).to(u.Mpc).value[idx])
        return ((1 + z)**2 * ell * jv(2, ell * theta) *
                (cosmology.H0 / cosmology.H(z_u)[idx]) *
                cosmology.angular_diameter_distance_z1z2(z_u, z_l)[idx] / d_l *
                cosmology.angular_diameter_distance_z1z2(z_u, z_s)[idx] / d_s *
                np.where(k > k_max, 0, np.array(
                    [p(z_i, k_i) for z_i, k_i in zip(z, k)])))

    ell_max = np.amax(jn_zeros(2, bessel_function_zeros)) / theta

    if not hasattr(theta, '__len__'):
        integral = _gaussian_quadrature_2d(
            partial(f, theta), n_z, 0, z_l, n_ell, 0, ell_max)
    else:
        integral = np.array([_gaussian_quadrature_2d(
            partial(f, theta[i]), n_z, 0, z_l, n_ell, 0, ell_max[i]) for i in
            range(len(theta))])

    integral = integral * u.Mpc**3  # units for the P(k) used earlier

    return (2 * (alpha_l - 1) * 9 * cosmology.H0**3 * cosmology.Om0**2 /
            (8 * np.pi * c.c**3) * integral).to(u.dimensionless_unscaled).value
