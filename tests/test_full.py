import numpy as np
import pytest

import treecorr
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, WMAP7
from astropy.cosmology import units as cu
from astropy.table import Table

from dsigma import physics, precompute, stacking
from fixtures import test_catalogs


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_treecorr(test_catalogs, n_jobs):
    # Compare against treecorr brute-force calculations.

    table_l, table_s = test_catalogs
    table_s['z'] = 1e3
    theta_bins = np.logspace(0, 1, 11)

    cat_l = treecorr.Catalog(
        ra=table_l['ra'], dec=table_l['dec'], ra_units='deg', dec_units='deg',
        w=table_l['w_sys'])
    cat_s = treecorr.Catalog(
        ra=table_s['ra'], dec=table_s['dec'], k=table_s['m'],
        g1=table_s['e_1'], g2=table_s['e_2'], ra_units='deg', dec_units='deg',
        w=table_s['w'])

    table_l = precompute.precompute(
        table_l, table_s, theta_bins * u.deg, weighting=0, n_jobs=n_jobs)

    kwargs = dict(
        max_sep=np.amax(theta_bins), min_sep=np.amin(theta_bins),
        nbins=len(theta_bins) - 1, sep_units='deg', metric='Arc', brute=True)
    ng = treecorr.NGCorrelation(**kwargs)
    ng.process(cat_l, cat_s)
    nk = treecorr.NKCorrelation(**kwargs)
    nk.process(cat_l, cat_s)

    assert np.all(
        np.array(ng.npairs, dtype=int) == stacking.number_of_pairs(table_l))
    assert np.allclose(ng.xi, stacking.raw_tangential_shear(
        table_l), atol=1e-9, rtol=0)
    assert np.allclose(nk.xi, stacking.scalar_shear_response_factor(
        table_l), atol=1e-8, rtol=0)


def test_brute_force(test_catalogs):
    # Test against a simple brute-force calculation.

    table_l, table_s = test_catalogs

    def projection_angle(ra_l, dec_l, ra_s, dec_s):

        ra_l, dec_l = np.deg2rad(ra_l), np.deg2rad(dec_l)
        ra_s, dec_s = np.deg2rad(ra_s), np.deg2rad(dec_s)

        tan_phi = (
            (np.cos(dec_s) * np.sin(dec_l) - np.sin(dec_s) * np.cos(dec_l) *
             np.cos(ra_l - ra_s)) / (np.cos(dec_l) * np.sin(ra_l - ra_s)))

        cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
        sin_2phi = 2.0 * tan_phi / (1.0 + tan_phi * tan_phi)

        return cos_2phi, sin_2phi

    cosmology = WMAP7
    rp_bins = np.logspace(-1, 1, 11) * u.Mpc / cu.littleh
    sum_1 = np.zeros(len(rp_bins) - 1, dtype=int)
    sum_w_ls = np.zeros(len(rp_bins) - 1)
    sum_w_ls_e_t = np.zeros(len(rp_bins) - 1)
    sum_w_ls_e_t_sigma_crit = np.zeros(len(rp_bins) - 1)

    for i in range(len(table_l)):
        theta = np.arccos(
            np.sin(np.deg2rad(table_l['dec'].value[i])) *
            np.sin(np.deg2rad(table_s['dec'].value)) +
            np.cos(np.deg2rad(table_l['dec'].value[i])) *
            np.cos(np.deg2rad(table_s['dec'].value)) *
            np.cos(np.deg2rad(table_l['ra'].value[i] - table_s['ra'].value)))
        rp = WMAP7.comoving_distance(table_l['z'][i]).to(
            u.Mpc / cu.littleh, cu.with_H0(cosmology.H0)) * theta
        use = (table_s['z'] > table_l['z'][i]) & (rp <= np.amax(rp_bins))
        if not np.any(use):
            continue
        rp = rp[use]
        sum_1 += np.histogram(rp, bins=rp_bins)[0]
        sigma_crit = physics.critical_surface_density(
            z_l=table_l['z'][i], z_s=table_s['z'][use],
            cosmology=cosmology).value
        w_ls = table_s['w'][use] / sigma_crit**2
        sum_w_ls += table_l['w_sys'][i] * np.histogram(
            rp, bins=rp_bins, weights=w_ls)[0]
        cos_2phi, sin_2phi = projection_angle(
            table_l['ra'].value[i], table_l['dec'].value[i],
            table_s['ra'][use].value, table_s['dec'][use].value)
        e_t = - cos_2phi * table_s['e_1'][use] + sin_2phi * table_s['e_2'][use]
        sum_w_ls_e_t += table_l['w_sys'][i] * np.histogram(
            rp, bins=rp_bins, weights=w_ls * e_t)[0]
        sum_w_ls_e_t_sigma_crit += table_l['w_sys'][i] * np.histogram(
            rp, bins=rp_bins, weights=w_ls * e_t * sigma_crit)[0]

    table_l = precompute.precompute(
        table_l, table_s, rp_bins, cosmology=cosmology)

    assert np.all(sum_1 == stacking.number_of_pairs(table_l))
    assert np.allclose(stacking.tangential_shear(table_l),
                       sum_w_ls_e_t / sum_w_ls, rtol=0, atol=1e-9)
    assert np.allclose(stacking.raw_excess_surface_density(table_l).value,
                       sum_w_ls_e_t_sigma_crit / sum_w_ls, rtol=0, atol=1e-9)


def test_comoving(test_catalogs):
    # Check that comoving coordinates are included correctly.

    table_l, table_s = test_catalogs
    z_l = 0.2
    table_l['z'] = z_l

    rp_bins = np.logspace(0, 1, 11)
    table_l_phy = precompute.precompute(
        table_l.copy(), table_s, rp_bins / (1 + z_l), comoving=False)
    table_l_com = precompute.precompute(
        table_l.copy(), table_s, rp_bins, comoving=True)

    assert np.all(stacking.number_of_pairs(table_l_phy) ==
                  stacking.number_of_pairs(table_l_com))
    assert np.allclose(
        stacking.raw_excess_surface_density(table_l_phy) / (1 + z_l)**2,
        stacking.raw_excess_surface_density(table_l_com), atol=1e-9, rtol=0)


def test_little_h(test_catalogs):
    # Check that results do not depend on h.

    table_l, table_s = test_catalogs

    cosmology_1 = FlatLambdaCDM(100, 0.3)
    cosmology_2 = FlatLambdaCDM(70, 0.3)

    rp_bins = np.logspace(0, 1, 11)
    table_l_1 = precompute.precompute(
        table_l.copy(), table_s, rp_bins, cosmology=cosmology_1)
    table_l_2 = precompute.precompute(
        table_l.copy(), table_s, rp_bins, cosmology=cosmology_2)

    assert np.all(stacking.number_of_pairs(table_l_1) ==
                  stacking.number_of_pairs(table_l_2))
    assert np.all(np.isclose(
        stacking.raw_excess_surface_density(table_l_1),
        stacking.raw_excess_surface_density(table_l_2), atol=1e-9, rtol=0))


def test_f_bias_1(test_catalogs):
    # Check that f_bias is unit when photo-z is always correct.

    table_l, table_s = test_catalogs
    table_c = Table()
    table_c['z'] = table_s['z'][:100]
    table_c['z_true'] = table_s['z'][:100]
    table_c['w'] = 1.0
    table_c['w_sys'] = 1.0

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute.precompute(table_l, table_s, rp_bins, table_c=table_c)
    f_bias = stacking.photo_z_dilution_factor(table_l)

    assert np.allclose(f_bias, 1.0, rtol=0, atol=1e-12)


def test_f_bias_2(test_catalogs):
    # Check that f_bias corrects redshift offsets.

    z_true = 0.7

    table_l, table_s = test_catalogs
    table_s['z'] = 0.5
    table_l['z'] = 0.2
    table_c = Table()
    table_c['z'] = np.repeat(0.5, 1)
    table_c['z_true'] = z_true
    table_c['w'] = 1.0
    table_c['w_sys'] = 1.0

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute.precompute(table_l, table_s, rp_bins, table_c=table_c)
    ds_1 = stacking.excess_surface_density(
        table_l, photo_z_dilution_correction=True)

    assert np.allclose(z_true, stacking.mean_source_redshift(table_l))

    table_s['z'] = 0.7
    table_l = precompute.precompute(table_l, table_s, rp_bins)
    ds_2 = stacking.excess_surface_density(table_l)

    assert np.allclose(ds_1, ds_2, rtol=0, atol=1e-6)


def test_nz(test_catalogs):
    # Test that n(z) distributions are included correctly.

    table_l, table_s = test_catalogs
    table_s = table_s[:1000]

    # Create an n(z) distribution where each source has its own n(z) that peaks
    # at the intrinsic z. Thus, including the n(z) shouldn't change the result.
    table_s['z_bin'] = np.arange(len(table_s))
    table_n = Table()
    table_n['z'] = table_s['z']
    table_n['n'] = np.eye(len(table_s))

    # Place all lenses in front of the sources. Otherwise, some rounding errors
    # may result in different numebers of pairs due to 0 weights. This also
    # save computation time.
    table_l['z'] = 0.15

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute.precompute(table_l, table_s, rp_bins)
    table_s.remove_column('z')
    table_l_nz = precompute.precompute(
        table_l.copy(), table_s, rp_bins, table_n=table_n)

    assert np.all(stacking.number_of_pairs(table_l) ==
                  stacking.number_of_pairs(table_l_nz))
    # Don't make the comparison too strict because of interpolation errors for
    # the effective critical surface density. This is a very unusual example
    # that we won't encounter in real-world applications.
    assert np.all(np.isclose(
        stacking.raw_excess_surface_density(table_l),
        stacking.raw_excess_surface_density(table_l_nz), atol=1e-6, rtol=0))
