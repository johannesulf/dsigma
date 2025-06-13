"""Module with functions specific to the Hyper-Suprime Cam survey."""
import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'multiplicative_selection_bias']

default_version = 'Y3'
known_versions = ['Y1', 'Y3', 'PDR1', 'PDR2', 'PDR3', 'S16A', 'S19A']
e_2_convention = 'flipped'


def default_column_keys(version=default_version):
    """Return a dictionary of default column keys.

    Parameters
    ----------
    version : string or None, optional
        Version of the catalog.

    Returns
    -------
    keys : dict
        Dictionary of default column keys.

    Raises
    ------
    ValueError
        If `version` does not correspond to a known catalog version.

    """
    if version in ['Y1', 'PDR1', 'PDR2', 'S16A']:
        keys = {
            'ra': 'ira',
            'dec': 'idec',
            'z': 'photoz_best',
            'z_low': 'photoz_err68_min',
            'e_1': 'ishape_hsm_regauss_e1',
            'e_2': 'ishape_hsm_regauss_e2',
            'w': 'ishape_hsm_regauss_derived_shape_weight',
            'm': 'ishape_hsm_regauss_derived_shear_bias_m',
            'e_rms': 'ishape_hsm_regauss_derived_rms_e',
            'R_2': 'ishape_hsm_regauss_resolution'}
    elif version in ['Y3', 'PDR3', 'S19A']:
        keys = {
            'ra': 'i_ra',
            'dec': 'i_dec',
            'z': np.nan,
            'e_1': 'i_hsmshaperegauss_e1',
            'e_2': 'i_hsmshaperegauss_e2',
            'w': 'i_hsmshaperegauss_derived_weight',
            'm': 'i_hsmshaperegauss_derived_shear_bias_m',
            'e_rms': 'i_hsmshaperegauss_derived_rms_e',
            'R_2': 'i_hsmshaperegauss_resolution',
            'z_bin': 'hsc_y3_zbin',
            'mag_A': 'i_apertureflux_10_mag'}
    else:
        raise ValueError(
            "Unkown version of HSC. Supported versions are {}.".format(
                known_versions))

    return keys


def multiplicative_selection_bias(table_s, version=default_version):
    r"""Compute the multiplicative selection bias.

    Parameters
    ----------
    table_s : astropy.table.Table
        HSC weak lensing source catalog.

    Returns
    -------
    m_sel : numpy.ndarray
        Per-object estimate of the HSC selection bias :math:`m_\mathrm{sel}`.

    """
    d_R_2 = 0.01
    d_mag_A = 0.025
    if version in ['Y1', 'PDR1', 'PDR2', 'S16A']:
        # section 5.6.2 in 1710.00885
        return (table_s['R_2'] < 0.3 + d_R_2) / d_R_2 * 0.00865
    elif version in ['Y3', 'PDR3', 'S19A']:
        # eq. (18) in 2304.00703
        return (0.01919 * (table_s['R_2'] < 0.3 + d_R_2) / d_R_2 +
                0.05854 * (table_s['mag_A'] > 25.5 - d_mag_A) / d_mag_A)
    else:
        raise ValueError(
            "Unkown version of HSC. Supported versions are {}.".format(
                known_versions))
