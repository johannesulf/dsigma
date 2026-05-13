import numpy as np
from astropy import units as u
from astropy.cosmology import units as cu

from dsigma import default_cosmology, physics


def test_critical_surface_density():
    # Perform some plausibility checks for the critical surface density.

    z_l = 0.3
    z_s = 0.5

    # result should have correct units
    sigma_crit = physics.critical_surface_density(z_l, z_s)
    assert isinstance(sigma_crit, u.Quantity)
    sigma_crit.to(cu.littleh * u.Msun / u.pc**2)

    # comoving should be included correctly
    sigma_crit_com = physics.critical_surface_density(z_l, z_s)
    sigma_crit_phy = physics.critical_surface_density(z_l, z_s, comoving=False)

    assert np.isclose(sigma_crit_com * (1 + z_l)**2,
                      sigma_crit_phy, rtol=0, atol=1e-12)

    # critical surface density is infinite if lens behind source
    z_l = 0.8
    z_s = 0.5
    sigma_crit = physics.critical_surface_density(z_l, z_s)
    assert sigma_crit == np.inf


def test_to_camb():
    # Test that we can convert to CAMB correctly.

    sigma_8 = 0.7
    cosmology_astropy = default_cosmology
    cosmology_camb = physics._to_camb(cosmology_astropy, sigma_8, 0.96,
                                      [1, 0.5, 0])

    assert np.isclose(cosmology_camb.get_sigma8_0(), sigma_8)

    z = np.linspace(0, 5, 20)

    assert np.allclose(
        cosmology_astropy.angular_diameter_distance(z).to(u.Mpc).value,
        cosmology_camb.angular_diameter_distance(z), rtol=3e-5, atol=1e-6)

    assert np.allclose(
        cosmology_camb.get_Omega('photon', z),
        cosmology_astropy.Ogamma(z), rtol=1e-4, atol=0)

    assert np.allclose(
        cosmology_camb.get_Omega('baryon', z) +
        cosmology_camb.get_Omega('cdm', z),
        cosmology_astropy.Om(z), rtol=0, atol=1e-4)

    # TODO: The disagreement is likely too large. Needs to be investigated.
    assert np.allclose(
        cosmology_camb.get_Omega('neutrino', z) +
        cosmology_camb.get_Omega('nu', z),
        cosmology_astropy.Onu(z), rtol=5e-2, atol=0)


def test_aussian_quadrature_2d():
    # Test 2d Gaussian quadrature on an analytic example.

    def f(x, y):
        return x**2 + np.sin(y)

    x_min = -2.5
    x_max = +2.4
    y_min = -1.2
    y_max = 1.3

    assert np.isclose(
        physics._gaussian_quadrature_2d(f, 10, x_min, x_max, 10, y_min, y_max),
        (x_max**3 - x_min**3) / 3.0 * (y_max - y_min) +
        (x_max - x_min) * (np.cos(y_min) - np.cos(y_max)))
