"""Module with functions specific to the DECADE survey."""

import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'multiplicative_shear_bias',
           'selection_response']

default_version = 'NGC'
known_versions = ['NGC', 'SGC']
e_2_convention = 'standard'


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
    if version in ['NGC', 'SGC']:
        keys = {
            'ra': 'RA',
            'dec': 'DEC',
            'z': np.nan,
            'z_bin': 'MCAL_SEL_NOSHEAR',
            'z_bin_1p': 'MCAL_SEL_1P',
            'z_bin_1m': 'MCAL_SEL_1M',
            'z_bin_2p': 'MCAL_SEL_2P',
            'z_bin_2m': 'MCAL_SEL_2M',
            'e_1': 'MCAL_G_1_NOSHEAR',
            'e_2': 'MCAL_G_2_NOSHEAR',
            'R_11': 'R_11',
            'R_12': 'R_12',
            'R_21': 'R_21',
            'R_22': 'R_22',
            'w': 'MCAL_W_NOSHEAR',
            'w_1p': 'MCAL_W_1P',
            'w_1m': 'MCAL_W_1M',
            'w_2p': 'MCAL_W_2P',
            'w_2m': 'MCAL_W_2M'}
    else:
        raise ValueError(
            "Unkown version of DECADE. Supported versions are {}.".format(
                known_versions))

    return keys


def multiplicative_shear_bias(z_bin, version=default_version):
    """Return the multiplicative shear bias.

    Parameters
    ----------
    z_bin : numpy.ndarray
        Tomographic redshift bin.
    version : string, optional
        Which catalog version to use.

    Returns
    -------
    m : numpy.ndarray
        The multiplicative shear for each galaxy.

    Raises
    ------
    ValueError
        If the `version` does not correspond to a known catalog version or
        multiplicative shear biases cannot be defined for this version of the
        catalog.

    """
    if version == 'NGC':
        m = np.array([-0.92, -1.90, -4.00, -3.73]) * 1e-2
        return np.where(z_bin != -1, m[z_bin], np.nan)
    elif version == 'SGC':
        m = np.array([-1.33, -2.26, -3.67, -5.72]) * 1e-2
        return np.where(z_bin != -1, m[z_bin], np.nan)
    else:
        raise ValueError(
            "Unkown version of DECADE. Supported versions are {}.".format(
                known_versions))


def selection_response(table_s):
    """Calculate the DECADE selection response.

    See Sheldon & Huff (2017) and McClintock et al. (2018) for details.

    Parameters
    ----------
    table_s : astropy.table.Table
        Catalog of sources.

    Returns
    -------
    R_sel : numpy.ndarray
        2x2 matrix for each galaxy containing the selection response.

    """
    R_sel = np.zeros((len(table_s), 2, 2))

    for z_bin in range(4):
        for i in range(2):
            for j in range(2):
                e_p_ave = np.average(
                    table_s[f'e_{i+1}'], weights=table_s[f'w_{j+1}p'] *
                    (table_s[f'z_bin_{j+1}p'] == z_bin))
                e_m_ave = np.average(
                    table_s[f'e_{i+1}'], weights=table_s[f'w_{j+1}m'] *
                    (table_s[f'z_bin_{j+1}m'] == z_bin))
                use = table_s['z_bin'] == z_bin
                R_sel[use, i, j] = (e_p_ave - e_m_ave) / 0.02

    return R_sel
