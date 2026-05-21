import numpy as np
import pytest

from astropy.table import Table


@pytest.fixture
def test_catalogs():
    # Create test catalogs in a 30 deg x 30 deg square field.

    n_l, n_s = 10000, 10000

    np.random.seed(0)

    table_l = Table()
    table_l['ra'] = np.random.uniform(low=0, high=30, size=n_l)
    table_l['dec'] = np.rad2deg(np.arcsin(np.random.uniform(
        low=0, high=np.sin(np.deg2rad(30)), size=n_l)))
    table_l['z'] = np.random.uniform(low=0.1, high=0.3, size=n_l)
    table_l['w_sys'] = np.random.uniform(low=0.8, high=1.2, size=n_l)

    table_s = Table()
    table_s['ra'] = np.random.uniform(low=0, high=30, size=n_s)
    table_s['dec'] = np.rad2deg(np.arcsin(np.random.uniform(
        low=0, high=np.sin(np.deg2rad(30)), size=n_s)))
    table_s['z'] = np.random.uniform(low=0.2, high=0.7, size=n_s)
    table_s['w'] = np.random.uniform(low=0.8, high=1.2, size=n_s)
    table_s['e_1'] = np.random.normal(loc=0, scale=0.2, size=n_s)
    table_s['e_2'] = np.random.normal(loc=0, scale=0.2, size=n_s)
    table_s['m'] = np.random.uniform(low=0.8, high=1.2, size=n_s)
    table_s['m_sel'] = np.random.uniform(low=0.8, high=1.2, size=n_s)
    table_s['e_rms'] = np.random.uniform(low=0.8, high=1.2, size=n_s)
    table_s['R_11'] = np.random.uniform(low=0.8, high=1.2, size=n_s)
    table_s['R_12'] = np.random.uniform(low=0.8, high=1.2, size=n_s)
    table_s['R_21'] = np.random.uniform(low=0.8, high=1.2, size=n_s)
    table_s['R_22'] = np.random.uniform(low=0.8, high=1.2, size=n_s)

    return table_l, table_s
