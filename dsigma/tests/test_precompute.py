import warnings
import numpy as np
from astropy.table import Table
from dsigma import precompute
from dsigma.stacking import number_of_pairs, raw_tangential_shear


def get_test_catalogs(n_s, n_l):

    np.random.seed(0)

    table_l = Table()
    table_l['ra'] = np.random.random(n_l) * 2 * np.pi
    table_l['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_l) - 1))
    table_l['z'] = np.random.random(n_l) * 0.2 + 0.1
    table_l['w_sys'] = 1.0

    table_s = Table()
    table_s['ra'] = np.random.random(n_s) * 2 * np.pi
    table_s['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_s) - 1))
    table_s['z'] = np.random.random(n_s) * 0.2 + 0.1
    table_s['w'] = np.random.random(n_s) * 0.2 + 0.9
    table_s['e_1'] = np.random.normal(loc=0, scale=0.2, size=n_s)
    table_s['e_2'] = np.random.normal(loc=0, scale=0.2, size=n_s)

    return table_l, table_s


def test_add_precompute_results_treecorr():

    try:
        import treecorr
    except ImportError:
        warnings.warn('TreeCorr is not installed. Skipping test.',
                      RuntimeWarning)
        return 0

    table_l, table_s = get_test_catalogs(1000, 10000)
    theta_bins = np.logspace(-1, 0, 11)

    cat_l = treecorr.Catalog(ra=table_l['ra'], dec=table_l['dec'],
                             ra_units='deg', dec_units='deg')
    cat_s = treecorr.Catalog(ra=table_s['ra'], dec=table_s['dec'],
                             g1=table_s['e_1'], g2=table_s['e_2'],
                             ra_units='deg', dec_units='deg')

    ng = treecorr.NGCorrelation(
        max_sep=np.amax(theta_bins), min_sep=np.amin(theta_bins),
        nbins=len(theta_bins) - 1, sep_units='deg', metric='Arc', brute=True)
    ng.process(cat_l, cat_s)

    table_l = precompute.add_precompute_results(table_l, table_s, theta_bins,
                                                shear_mode=True, nside=32)

    assert np.all(np.array(ng.npairs, dtype=int) == number_of_pairs(table_l))
    assert np.all(np.isclose(ng.xi, raw_tangential_shear(table_l), atol=1e-9,
                             rtol=0))
