from functools import partial

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import units as cu

from dsigma import jackknife, precompute, stacking
from fixtures import test_catalogs


@pytest.mark.parametrize('n_jk', [20, 50, 100, 200])
def test_jackknife(test_catalogs, n_jk):

    table_l, table_s = test_catalogs

    jackknife.compute_jackknife_fields(table_l, n_jk)
    assert len(np.unique(table_l['field_jk']) == n_jk)

    n_bins = 10
    theta_bins = np.logspace(0, 1, n_bins + 1) * u.deg
    table_l = precompute.precompute(table_l, table_s, theta_bins)

    # There should be no tangential shear. So no shear should be detected
    # with high signficance and we should expect roughly a chi^2 distribution.
    gt = stacking.tangential_shear(table_l)
    gt_err = np.sqrt(np.diag(jackknife.jackknife_resampling(
        stacking.tangential_shear, table_l)))
    assert np.amax(np.abs(gt / gt_err)) < 5

    # Compressing the jackknife fields should not impact the result.
    assert np.all(np.isclose(
        jackknife.jackknife_resampling(
            stacking.tangential_shear, table_l, compress=False),
        jackknife.jackknife_resampling(
            stacking.tangential_shear, table_l, compress=True),
        rtol=0, atol=1e-12))

    # Test the smoothing.
    cov = jackknife.jackknife_resampling(stacking.tangential_shear, table_l)
    for sigma in [1.0, 2.0, 5.0]:
        cov_smooth = jackknife.smooth_covariance_matrix(cov, 1.0)

        # Diagonal should be unchanged.
        assert np.allclose(np.diag(cov), np.diag(cov_smooth))

        diag = np.diag(cov)
        cor = cov / np.outer(np.sqrt(diag), np.sqrt(diag))
        cor_smooth = cov_smooth / np.outer(np.sqrt(diag), np.sqrt(diag))

        # The noise should be reduced.
        assert np.std(cor) > np.std(cor_smooth)

        # The mean value of the correlation matrix should be the same.
        assert np.isclose(np.mean(cor), np.mean(cor_smooth), rtol=0, atol=1e-12)

    # With random subtraction, result should always be 0.
    cov = jackknife.jackknife_resampling(
        stacking.tangential_shear, table_l, table_r=table_l,
        random_subtraction=True)
    assert np.allclose(cov, 0)

    # Same when calculating the difference for the same catalog.
    def diff(table_l, table_r=None, table_r_2=None, table_l_2=None,
             random_subtraction=True):
        gt = partial(
            stacking.tangential_shear, random_subtraction=random_subtraction)
        return gt(table_l, table_r) - gt(table_l_2, table_r_2)

    for random_subtraction in [True, False]:
        cov = jackknife.jackknife_resampling(
            diff, table_l, table_r=table_l, table_l_2=table_l,
            table_r_2=table_l, random_subtraction=random_subtraction)
        assert np.allclose(cov, 0)

    # Covariance should have quantities in some cases.
    cov = jackknife.jackknife_resampling(
        stacking.excess_surface_density, table_l)
    assert cov.unit == (cu.littleh**2 * u.Msun**2 / u.pc**4)
