import numpy as np
import pytest

from dsigma import precompute, stacking
from fixtures import test_catalogs


@pytest.mark.parametrize('statistic', ['ds', 'gt'])
def test_correction_factors(test_catalogs, statistic):
    # Check that all correction factors are correctly included.

    table_l, table_s = test_catalogs
    rp_bins = np.logspace(0, 1, 11)

    table_l = precompute.precompute(table_l, table_s, rp_bins)
    kwargs = dict(scalar_shear_response_correction=True,
                  matrix_shear_response_correction=True,
                  shear_responsivity_correction=True,
                  selection_bias_correction=True, return_table=True)

    if statistic == 'ds':
        result = stacking.excess_surface_density(table_l, **kwargs)
    else:
        result = stacking.tangential_shear(table_l, **kwargs)

    assert np.allclose(
        result[statistic], result[f'{statistic}_raw'] /
        result['1+m'] / result['R_t'] / result['2R'] / result['1+m_sel'],
        rtol=0, atol=1e-9)


def test_random_subtraction(test_catalogs):
    # Check that random subtraction works.

    table_l, table_s = test_catalogs
    rp_bins = np.logspace(0, 1, 11)

    table_l = precompute.precompute(table_l, table_s, rp_bins)
    kwargs = dict(scalar_shear_response_correction=True,
                  matrix_shear_response_correction=True,
                  shear_responsivity_correction=True,
                  selection_bias_correction=True,
                  table_r=table_l, random_subtraction=True)

    assert np.allclose(
        0, stacking.excess_surface_density(
            table_l, **kwargs), rtol=0, atol=1e-9)
    assert np.allclose(
        0, stacking.tangential_shear(
            table_l, **kwargs), rtol=0, atol=1e-9)
