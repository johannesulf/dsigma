import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM, FlatLambdaCDM
from astropy.table import Table
from dsigma.precompute import precompute
from fixtures import test_catalogs


def test_pairs_below_pixel_size():
    # Each lens gets one source 30" away, far below the nside=256 pixel
    # (~13.7'). The sub-pixel search radius must not cause the pixel
    # pre-filter to drop same-pixel pairs, so #pairs == #lenses.
    ra, dec = np.meshgrid(np.linspace(10.0, 18.0, 25),
                          np.linspace(-8.0, 8.0, 20))
    ra, dec = ra.ravel(), dec.ravel()
    n = ra.size

    table_l = Table()
    table_l['ra'] = ra * u.deg
    table_l['dec'] = dec * u.deg
    table_l['z'] = np.full(n, 0.2)
    table_l['w_sys'] = np.ones(n)

    c_s = SkyCoord(table_l['ra'], table_l['dec']).directional_offset_by(
        0 * u.deg, 30 * u.arcsec)
    table_s = Table()
    table_s['ra'] = c_s.ra
    table_s['dec'] = c_s.dec
    table_s['z'] = np.full(n, 0.8)
    table_s['w'] = np.ones(n)
    table_s['e_1'] = np.zeros(n)
    table_s['e_2'] = np.zeros(n)

    rp_bins = [15, 60] * u.arcsec
    table_l = precompute(table_l, table_s, rp_bins,
                         cosmology=FlatLambdaCDM(70, 0.2, 0.7))
    assert table_l['sum 1'].sum() == n


def test_warnings_errors(test_catalogs):
    # Check that dsigma throws expected value errors.

    np.random.seed(0)

    rp_bins = np.logspace(-1, 0, 3)
    table_l, table_s = test_catalogs

    # non-flat cosmology
    cosmology = LambdaCDM(70, 0.2, 0.7)
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins, cosmology=cosmology)

    # lens redshifts negative
    table_l = table_l.copy()
    table_l['z'][0] *= -1
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins)
    table_l['z'][0] *= -1

    # nside invalid
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins, nside=13)

    # n_jobs invalid
    for n_jobs in [0, -1, 1.5]:
        with pytest.raises(ValueError):
            precompute(table_l, table_s, rp_bins, n_jobs=n_jobs)

    # table_c and table_n both given
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins, table_c=Table(), table_n=Table())

    # table_n given, warning that column `z` is ignored
    table_n = Table()
    table_n['z'] = np.linspace(0.5, 1.0, 51)
    table_n['n'] = np.random.random(size=(len(table_n), 4))
    table_s['z_bin'] = np.random.randint(0, 4, len(table_s))
    with pytest.warns(UserWarning):
        precompute(table_l, table_s, rp_bins, table_n=table_n)

    # table_n given but table_s has no `z_bin` column
    table_s.remove_columns(['z', 'z_bin'])
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins, table_n=table_n)

    # `z_bin` contains negative numbers
    table_s['z_bin'] = np.random.randint(0, 4, len(table_s))
    table_s['z_bin'][0] = -1
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins, table_n=table_n)

    # not all tomographic bins in table_n
    table_s['z_bin'] = np.random.randint(0, 5, len(table_s))
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins, table_n=table_n)

    # `z_l_max` larger than `z`
    table_s['z'] = np.random.random(len(table_s)) * 0.5 + 0.2
    table_s['z_l_max'] = table_s['z'] + 0.1
    with pytest.raises(ValueError):
        precompute(table_l, table_s, rp_bins)
