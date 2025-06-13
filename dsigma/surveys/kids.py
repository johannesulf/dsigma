"""Module with functions specific to the Kilo Degree Survey."""

import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'tomographic_redshift_bin',
           'multiplicative_shear_bias']

default_version = 'DR4'
known_versions = ['DR3', 'KV450', 'DR4', 'KiDS-1000', '1000']
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
    if version == 'DR3':
        keys = {
            'ra': 'RAJ2000',
            'dec': 'DECJ2000',
            'z': 'Z_B',
            'z_low': 'Z_B_MIN',
            'e_1': 'e1',
            'e_2': 'e2',
            'w': 'weight',
            'm': 'm'}
    elif version == 'KV450':
        keys = {
            'ra': 'ALPHA_J2000',
            'dec': 'DELTA_J2000',
            'z': 'Z_B',
            'z_low': 'Z_B_MIN',
            'e_1': 'bias_corrected_e1',
            'e_2': 'bias_corrected_e2',
            'w': 'weight'}
    elif version in ['DR4', 'KiDS-1000', '1000']:
        keys = {
            'ra': 'ALPHA_J2000',
            'dec': 'DELTA_J2000',
            'z': 'Z_B',
            'z_low': 'Z_B_MIN',
            'e_1': 'e1',
            'e_2': 'e2',
            'w': 'weight'}
    else:
        raise ValueError(
            "Unkown version of KiDS. Supported versions are {}.".format(
                known_versions))

    return keys


def tomographic_redshift_bin(z_s, version=default_version):
    """Return the photometric redshift bin.

    Parameters
    ----------
    z_s : numpy.ndarray
        Photometric redshifts.
    version : string, optional
        Which catalog version to use. Currently ignored.

    Returns
    -------
    z_bin : numpy.ndarray
        The tomographic redshift bin corresponding to each photometric
        redshift. Returns -1 in case a redshift does not fall into any bin.

    """
    z_bin = np.digitize(z_s, np.array(
        [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]) + 1e-6, right=True) - 1
    z_bin = np.where((z_s < 0.1) | (z_s >= 1.2), -1, z_bin)

    return z_bin


def multiplicative_shear_bias(z_bin, version=default_version):
    """Return the multiplicative shear bias.

    For many version of KiDS, the multiplicative shear bias is not estimated
    on the basis of individual sources but for broad photometric redshift
    bins. This function returns the multiplicative bias :math:`m` as a function
    of the bin.

    Parameters
    ----------
    z_bin : numpy.ndarray
        Tomographic redshift bin.
    version : string
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
    if version == 'DR3':

        raise ValueError('For DR3, the multiplicative shear bias is ' +
                         'defined for each object individually.')

    elif version in ['KV450', 'DR4', 'KiDS-1000', '1000']:

        if version == 'KV450':
            m = np.array([-0.017, -0.008, -0.015, 0.010, 0.006])
        else:
            m = np.array([-0.009, -0.011, -0.015, 0.002, 0.007])

        return np.where(z_bin != -1, m[z_bin], np.nan)

    else:
        raise ValueError(
            "Unkown version of KiDS. Supported versions are {}.".format(
                known_versions))
