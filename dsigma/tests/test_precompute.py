import warnings
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from dsigma.precompute import precompute
from dsigma.stacking import number_of_pairs, raw_tangential_shear
from dsigma.stacking import raw_excess_surface_density, photo_z_dilution_factor
from dsigma.stacking import excess_surface_density


def get_test_catalogs(n_l, n_s):

    np.random.seed(0)

    table_l = Table()
    table_l['ra'] = np.random.random(n_l) * 2 * np.pi
    table_l['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_l) - 1))
    table_l['z'] = np.random.random(n_l) * 0.2 + 0.1
    table_l['w_sys'] = 1.0

    table_s = Table()
    table_s['ra'] = np.random.random(n_s) * 2 * np.pi
    table_s['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_s) - 1))
    table_s['z'] = np.random.random(n_s) * 0.5 + 0.1
    table_s['w'] = np.random.random(n_s) * 0.2 + 0.9
    table_s['e_1'] = np.random.normal(loc=0, scale=0.2, size=n_s)
    table_s['e_2'] = np.random.normal(loc=0, scale=0.2, size=n_s)

    return table_l, table_s


def test_precompute_treecorr():

    try:
        import treecorr
    except ImportError:
        warnings.warn('TreeCorr is not installed. Skipping test.',
                      RuntimeWarning)
        return 0

    table_l, table_s = get_test_catalogs(1000, 10000)
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

    table_l = precompute(table_l, table_s, theta_bins * u.deg, weighting=0,
                         lens_source_cut=None)

    assert np.all(np.array(ng.npairs, dtype=int) == number_of_pairs(table_l))
    assert np.all(np.isclose(ng.xi, raw_tangential_shear(table_l), atol=1e-9,
                             rtol=0))


def test_precompute_comoving():

    table_l, table_s = get_test_catalogs(1000, 10000)
    z_l = 0.2
    table_l['z'] = z_l

    rp_bins = np.logspace(0, 1, 11)
    table_l_phy = precompute(table_l.copy(), table_s, rp_bins / (1 + z_l),
                             comoving=False)
    table_l_com = precompute(table_l.copy(), table_s, rp_bins, comoving=True)

    assert np.all(number_of_pairs(table_l_phy) == number_of_pairs(table_l_com))
    assert np.all(np.isclose(
        raw_excess_surface_density(table_l_phy) / (1 + z_l)**2,
        raw_excess_surface_density(table_l_com), atol=1e-9, rtol=0))


def test_precompute_little_h():

    table_l, table_s = get_test_catalogs(1000, 10000)

    cosmology = FlatLambdaCDM(100, 0.3)
    h = 0.7
    cosmology_h = FlatLambdaCDM(100 * h, 0.3)

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute(table_l, table_s, rp_bins, cosmology=cosmology)
    table_l_h = precompute(table_l.copy(), table_s, rp_bins / h,
                           cosmology=cosmology_h)

    assert np.all(number_of_pairs(table_l) == number_of_pairs(table_l_h))
    assert np.all(np.isclose(
        raw_excess_surface_density(table_l),
        raw_excess_surface_density(table_l_h) / h, atol=1e-9, rtol=0))


def test_precompute_f_bias_1():

    table_l, table_s = get_test_catalogs(1000, 10000)
    table_c = Table()
    table_c['z'] = table_s['z']
    table_c['z_true'] = table_s['z']
    table_c['w'] = 1.0
    table_c['w_sys'] = 1.0
    table_c['z_l_max'] = table_s['z']

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute(table_l, table_s, rp_bins, table_c=table_c)
    f_bias = photo_z_dilution_factor(table_l)

    assert np.all(np.isclose(f_bias, 1.0, rtol=0, atol=1e-12))


def test_precompute_f_bias_2():

    table_l, table_s = get_test_catalogs(1000, 10000)
    table_s['z'] = 0.5
    table_l['z'] = 0.2 + np.random.random(len(table_l)) * 1e-9
    table_c = Table()
    table_c['z'] = np.ones(1) * 0.5
    table_c['z_true'] = 0.7
    table_c['w'] = 1.0
    table_c['w_sys'] = 1.0
    table_c['z_l_max'] = table_c['z']

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute(table_l, table_s, rp_bins, table_c=table_c)
    ds_1 = excess_surface_density(table_l, photo_z_dilution_correction=True)

    table_s['z'] = 0.7
    table_l = precompute(table_l, table_s, rp_bins)
    ds_2 = excess_surface_density(table_l)

    assert np.all(np.isclose(ds_1, ds_2, rtol=0, atol=1e-6))


def test_precompute_nz():

    table_l, table_s = get_test_catalogs(10000, 100)
    table_s = table_s
    table_l = table_l

    # Create an n(z) distribution where each source has its own n(z) that peaks
    # at the intrinsic z. Thus, including the n(z) shouldn't change the result.
    table_s['z_bin'] = np.arange(len(table_s))
    table_n = Table()
    table_n['z'] = table_s['z']
    table_n['n'] = np.eye(len(table_s))

    rp_bins = np.logspace(0, 1, 11)
    table_l = precompute(table_l, table_s, rp_bins)
    table_l_nz = precompute(table_l.copy(), table_s, rp_bins, table_n=table_n)

    assert np.all(number_of_pairs(table_l) == number_of_pairs(table_l_nz))
    # Don't make the comparison too strict because of interpolation errors for
    # the effective critical surface density. This is a very unusual example
    # that we won't encounter in real-world applications.
    assert np.all(np.isclose(
        raw_excess_surface_density(table_l),
        raw_excess_surface_density(table_l_nz), atol=1e-6, rtol=0))
