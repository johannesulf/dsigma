from dsigma import jackknife, precompute, stacking
import numpy as np
import pytest

from astropy import units as u

from fixtures import test_catalogs


@pytest.mark.parametrize("n_jk", [20, 50, 100, 200])
def test_jackknife(test_catalogs, n_jk):

    table_l, table_s = test_catalogs
    np.random.seed(0)
    table_l['w_sys'] = np.random.random(len(table_l))

    jackknife.compute_jackknife_fields(table_l, n_jk)
    assert len(np.unique(table_l['field_jk']) == n_jk)

    n_bins = 30
    theta_bins = np.logspace(0, 1, n_bins + 1) * u.deg
    table_l = precompute.precompute(table_l, table_s, theta_bins)

    # There should be no tangential shear. So no shear should be detected
    # with high signficance and we should expect roughly a chi^2 distribution.
    y = stacking.tangential_shear(table_l)
    y_err = np.sqrt(np.diag(jackknife.jackknife_resampling(
        stacking.tangential_shear, table_l)))
    assert np.amax(y / y_err) < 5
    assert np.isclose(np.sum((y**2 / y_err**2)), n_bins, rtol=0,
                      atol=5 * np.sqrt(n_bins))

    # Compressing the jackknife fields should not impact the result.
    table_c = jackknife.compress_jackknife_fields(table_l)
    assert np.all(np.isclose(
        jackknife.jackknife_resampling(stacking.tangential_shear, table_l),
        jackknife.jackknife_resampling(stacking.tangential_shear, table_c),
        rtol=0, atol=1e-12))
