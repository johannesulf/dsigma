"""Module with functions specific to the Dark Energy Survey."""

import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'tomographic_redshift_bin',
           'multiplicative_shear_bias', 'selection_response']

default_version = 'Y3'
known_versions = ['Y1', 'Y3']
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
    if version == 'Y1':
        keys = {
            'ra': 'ra',
            'dec': 'dec',
            'z': 'MEAN_Z',
            'z_true': 'Z_MC',
            'e_1': 'e1',
            'e_2': 'e2',
            'w': 1,
            'R_11': 'R11',
            'R_12': 'R12',
            'R_21': 'R21',
            'R_22': 'R22',
            'flags_select': 'flags_select',
            'flags_select_1p': 'flags_select_1p',
            'flags_select_1m': 'flags_select_1m',
            'flags_select_2p': 'flags_select_2p',
            'flags_select_2m': 'flags_select_2m'}
    elif version == 'Y3':
        keys = {
            'ra': 'ra',
            'dec': 'dec',
            'z': np.nan,
            'z_bin': 'bhat',
            'e_1': 'e_1',
            'e_2': 'e_2',
            'w': 'weight',
            'w_1p': 'weight_1p',
            'w_1m': 'weight_1m',
            'w_2p': 'weight_2p',
            'w_2m': 'weight_2m',
            'R_11': 'R11',
            'R_12': 'R12',
            'R_21': 'R21',
            'R_22': 'R22',
            'flags_select': 'flags_select',
            'flags_select_1p': 'flags_select_1p',
            'flags_select_1m': 'flags_select_1m',
            'flags_select_2p': 'flags_select_2p',
            'flags_select_2m': 'flags_select_2m'}
    else:
        raise ValueError(
            "Unkown version of DES. Supported versions are {}.".format(
                known_versions))

    return keys


def tomographic_redshift_bin(z_s, version=default_version):
    """Return the photometric redshift bin.

    Parameters
    ----------
    z_s : numpy.ndarray
        Photometric redshifts.
    version : string, optional
        Which catalog version to use.

    Returns
    -------
    z_bin : numpy.ndarray
        The tomographic redshift bin corresponding to each photometric
        redshift. Returns -1 in case a redshift does not fall into any bin.

    Raises
    ------
    ValueError
        If the `version` does not correspond to a known catalog version or
        if tomographic bins were not assigned based on photometric redshifts
        for the given catalog version.

    """
    if version == 'Y1':
        z_bins = [0.2, 0.43, 0.63, 0.9, 1.3]
    elif version == 'Y3':
        raise ValueError('DES Y3 assigns redshift bins based on colors, ' +
                         'not photometric redshifts.')
    else:
        raise ValueError(
            "Unkown version of DES. Supported versions are {}.".format(
                known_versions))

    z_bin = np.digitize(z_s, z_bins) - 1
    z_bin = np.where((z_s < np.amin(z_bins)) | (z_s >= np.amax(z_bins)) |
                     np.isnan(z_s), -1, z_bin)

    return z_bin


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


def selection_response(table_s, version=default_version):
    """Calculate the DES selection response.

    See Sheldon & Huff (2017) and McClintock et al. (2018) for details.

    Parameters
    ----------
    table_s : astropy.table.Table
        Catalog of sources.
    version : string, optional
        Which catalog version to use.

    Returns
    -------
    R_sel : numpy.ndarray
        2x2 matrix containing the selection response.

    """
    R_sel = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):

            if np.issubdtype(table_s['flags_select_1p'].dtype, np.integer):
                select_p = table_s['flags_select_{}p'.format(i + 1)] == 0
                select_m = table_s['flags_select_{}m'.format(i + 1)] == 0
            else:
                select_p = table_s['flags_select_{}p'.format(i + 1)]
                select_m = table_s['flags_select_{}m'.format(i + 1)]
            e = table_s['e_{}'.format(j + 1)]

            if version == 'Y3':
                w_p = table_s['w_{}p'.format(i + 1)]
                w_m = table_s['w_{}m'.format(i + 1)]
            else:
                w_p = np.ones(len(table_s))
                w_m = np.ones(len(table_s))

            R_sel[i, j] = (
                np.average(e[select_p], weights=w_p[select_p]) -
                np.average(e[select_m], weights=w_m[select_m])) / 0.02

    return R_sel
