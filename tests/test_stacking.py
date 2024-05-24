import numpy as np
import warnings

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from dsigma import precompute, stacking

from fixtures import test_catalogs


def test_treecorr(test_catalogs):

    try:
        import treecorr
    except ImportError:
        warnings.warn('TreeCorr is not installed. Skipping test.',
                      RuntimeWarning)
        return None

    table_l, table_s = test_catalogs
    theta_bins = np.logspace(0, 1, 11)

    cat_l = treecorr.Catalog(ra=table_l['ra'], dec=table_l['dec'],
                             ra_units='deg', dec_units='deg')
    cat_s = treecorr.Catalog(ra=table_s['ra'], dec=table_s['dec'],
                             g1=table_s['e_1'], g2=table_s['e_2'],
                             ra_units='deg', dec_units='deg', w=table_s['w'])

    ng = treecorr.NGCorrelation(
        max_sep=np.amax(theta_bins), min_sep=np.amin(theta_bins),
        nbins=len(theta_bins) - 1, sep_units='deg', metric='Arc', brute=True)
    ng.process(cat_l, cat_s)

    table_l = precompute.precompute(table_l, table_s, theta_bins * u.deg,
                                    weighting=0, lens_source_cut=None)

    assert np.all(
        np.array(ng.npairs, dtype=int) == stacking.number_of_pairs(table_l))
    assert np.all(np.isclose(
        ng.xi, stacking.raw_tangential_shear(table_l), atol=1e-9, rtol=0))


def test_comoving(test_catalogs):

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
    assert np.all(np.isclose(
        stacking.raw_excess_surface_density(table_l_phy) / (1 + z_l)**2,
        stacking.raw_excess_surface_density(table_l_com), atol=1e-9, rtol=0))


def test_little_h(test_catalogs):

    table_l, table_s = test_catalogs

    cosmology = FlatLambdaCDM(100, 0.3)
    h = 0.7
    cosmology_h = FlatLambdaCDM(100 * h, 0.3)

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute.precompute(
        table_l, table_s, rp_bins, cosmology=cosmology)
    table_l_h = precompute.precompute(
        table_l.copy(), table_s, rp_bins / h, cosmology=cosmology_h)

    assert np.all(stacking.number_of_pairs(table_l) ==
                  stacking.number_of_pairs(table_l_h))
    assert np.all(np.isclose(
        stacking.raw_excess_surface_density(table_l),
        stacking.raw_excess_surface_density(table_l_h) / h, atol=1e-9, rtol=0))


def test_f_bias_1(test_catalogs):

    table_l, table_s = test_catalogs
    table_c = Table()
    table_c['z'] = table_s['z']
    table_c['z_true'] = table_s['z']
    table_c['w'] = 1.0
    table_c['w_sys'] = 1.0
    table_c['z_l_max'] = table_s['z']

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute.precompute(table_l, table_s, rp_bins, table_c=table_c)
    f_bias = stacking.photo_z_dilution_factor(table_l)

    assert np.all(np.isclose(f_bias, 1.0, rtol=0, atol=1e-12))


def test_f_bias_2(test_catalogs):

    table_l, table_s = test_catalogs
    table_s['z'] = 0.5
    table_l['z'] = 0.2 + np.random.random(len(table_l)) * 1e-9
    table_c = Table()
    table_c['z'] = np.ones(1) * 0.5
    table_c['z_true'] = 0.7
    table_c['w'] = 1.0
    table_c['w_sys'] = 1.0
    table_c['z_l_max'] = table_c['z']

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute.precompute(table_l, table_s, rp_bins, table_c=table_c)
    ds_1 = stacking.excess_surface_density(
        table_l, photo_z_dilution_correction=True)

    table_s['z'] = 0.7
    table_l = precompute.precompute(table_l, table_s, rp_bins)
    ds_2 = stacking.excess_surface_density(table_l)

    assert np.all(np.isclose(ds_1, ds_2, rtol=0, atol=1e-6))


def test_nz(test_catalogs):

    table_l, table_s = test_catalogs
    table_s = table_s
    table_l = table_l

    # Create an n(z) distribution where each source has its own n(z) that peaks
    # at the intrinsic z. Thus, including the n(z) shouldn't change the result.
    table_s['z_bin'] = np.arange(len(table_s))
    table_n = Table()
    table_n['z'] = table_s['z']
    table_n['n'] = np.eye(len(table_s))

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute.precompute(table_l, table_s, rp_bins)
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
