"""Module with functions specific to the DECADE survey."""

import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'multiplicative_shear_bias',
           'selection_response']

default_version = 'default'
known_versions = ['default']
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
    if version == 'default':
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

    For DES Y3, we can define a blending-related multiplicative shear bias.
    This function returns the multiplicative bias :math:`m` as a function
    of the bin. The values can be computed from the blending-corrected Y3
    redshift distributions and are, as expected, very similar to the values in
    Table 4 in MacCrann et al. (2022) where they were calculated for mock
    catalogs.

    Parameters
    ----------
    z_bin : numpy.ndarray
        Tomographic redshift bin.
    version : string, optional
        Which catalog version to use.

    Returns
    -------
    m : numpy.ndarray
        The multiplicative shear bias corresponding to each tomographic bin.

    Raises
    ------
    ValueError
        If the `version` does not correspond to a known catalog version or
        multiplicative shear biases cannot be defined for this version of the
        catalog.

    """
    if version == 'Y1':
        raise ValueError(
            'For Y1, we cannot define a multiplicative shear bias.')

    elif version == 'Y3':
        m = np.array([-0.63, -1.98, -2.41, -3.69]) * 0.01
        return np.where(z_bin != -1, m[z_bin], np.nan)

    else:
        raise ValueError(
            "Unkown version of DES. Supported versions are {}.".format(
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

    for z_bin in range(1, 5):
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
