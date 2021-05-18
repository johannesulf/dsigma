import numpy as np
from astropy.cosmology import FlatLambdaCDM

from ..physics import critical_surface_density


def test_critical_surface_density():

    cosmology = FlatLambdaCDM(100, 0.3)
    z_l = 0.3
    z_s = 0.5

    sigma_crit_com = critical_surface_density(z_l, z_s, cosmology)
    sigma_crit_phy = critical_surface_density(z_l, z_s, cosmology,
                                              comoving=False)

    assert np.isclose(sigma_crit_com * (1 + z_l)**2,
                      sigma_crit_phy, rtol=0, atol=1e-12)
