import numpy as np
import pytest

from dsigma import precompute, stacking
from fixtures import test_catalogs


@pytest.mark.parametrize('stacking_function', [
    stacking.excess_surface_density, stacking.tangential_shear])
def test_correction_factors(test_catalogs, stacking_function):
    # Check that all correction factors are correctly included.

    table_l, table_s = test_catalogs
    rp_bins = np.logspace(0, 1, 11)

    table_l = precompute.precompute(table_l, table_s, rp_bins)
    kwargs = dict(scalar_shear_response_correction=True,
                  matrix_shear_response_correction=True,
                  shear_responsivity_correction=True,
                  selection_bias_correction=True, return_table=True)

    result = stacking_function(table_l, **kwargs)
    if stacking_function == stacking.excess_surface_density:
        key = 'ds'
    else:
        key = 'gt'

    assert np.allclose(
        result[key], result[f'{key}_raw'] / result['1+m'] / result['R_t'] /
        result['2R'] / result['1+m_sel'], rtol=0, atol=1e-9)


@pytest.mark.parametrize('stacking_function', [
    stacking.excess_surface_density, stacking.tangential_shear])
def test_random_subtraction(test_catalogs, stacking_function):
    # Check that random subtraction works.

    table_l, table_s = test_catalogs
    rp_bins = np.logspace(0, 1, 11)

    table_l = precompute.precompute(table_l, table_s, rp_bins)
    kwargs = dict(scalar_shear_response_correction=True,
                  matrix_shear_response_correction=True,
                  shear_responsivity_correction=True,
                  selection_bias_correction=True,
                  table_r=table_l, random_subtraction=True)

    assert np.all(0 == stacking_function(table_l, **kwargs))

    kwargs['random_subtraction'] = False
    with pytest.warns():
        stacking_function(table_l, **kwargs)


@pytest.mark.parametrize('stacking_function', [
    stacking.excess_surface_density, stacking.tangential_shear])
def test_boost_factor(test_catalogs, stacking_function):
    # Check that the boost factor works.

    table_l, table_s = test_catalogs
    rp_bins = np.logspace(0, 1, 11)

    table_l = precompute.precompute(table_l, table_s, rp_bins)
    kwargs = dict(boost_correction=True, table_r=table_l)

    assert np.allclose(
        stacking_function(table_l), stacking_function(table_l, **kwargs),
        rtol=0, atol=1e-9)

    kwargs['boost_correction'] = False
    with pytest.warns():
        stacking_function(table_l, **kwargs)


def test_lens_magnification(test_catalogs):
    # Check that lens magnification doesn't crash and passes sanity check.

    table_l, table_s = test_catalogs
    rp_bins = np.logspace(0, 1, 11)

    table_l = precompute.precompute(table_l, table_s, rp_bins)
    ds_lm = stacking.lens_magnification_bias(table_l, 2.0)

    assert np.all(ds_lm > 0)
