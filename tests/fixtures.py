import numpy as np
import pytest

from astropy.table import Table


@pytest.fixture
def test_catalogs():

    n_l, n_s = 10000, 1000

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
