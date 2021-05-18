import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'tomographic_redshift_bin',
           'multiplicative_shear_bias']

default_version = 'DR4'
known_versions = ['DR3', 'KV450', 'DR4']
e_2_convention = 'standard'


def default_column_keys(version=default_version):

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
    elif version == 'DR4':
        keys = {
            'ra': 'ALPHA_J2000',
            'dec': 'DELTA_J2000',
            'z': 'Z_B',
            'z_low': 'Z_B_MIN',
            'e_1': 'e1',
            'e_2': 'e2',
            'w': 'weight'}
    else:
        raise RuntimeError(
            "Unkown version of KiDS. Supported versions are {}.".format(
                known_versions))

    return keys


def tomographic_redshift_bin(z_s, version=default_version):
    """KiDS KV450 and DR4 analyses work in pre-defined tomographic redshift
    bins. This function returns the photometric redshift bin as a function of
    photometric redshift.

    Parameters
    ----------
    z_s : numpy array
        Photometric redshifts.
    version : string
        Which catalog version to use.

    Returns
    -------
    z_bin : numpy array
        The tomographic redshift bin corresponding to each photometric
        redshift. Returns -1 in case a redshift does not fall into any bin.
    """

    z_bin = np.digitize(z_s, [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]) - 1
    z_bin = np.where((z_s < 0.1) | (z_s >= 1.2), -1, z_bin)

    return z_bin


def multiplicative_shear_bias(z_s, version=default_version):
    """For many version of KiDS, the multiplicative shear bias is not estimated
    on the basis of individual sources but for broad photometric redshift
    bins. This function returns the multiplicative bias :math:`m` as a function
    of source photometric redshift.

    Parameters
    ----------
    z_s : numpy array
        Photometric redshifts.
    version : string
        Which catalog version to use.

    Returns
    -------
    m : numpy array
        The multiplicative shear bias corresponding to each photometric
        redshift.
    """

    if version == 'DR3':

        raise RuntimeError('For DR3, the multiplicative shear bias is ' +
                           'defined for each object individually.')

    elif version in ['KV450', 'DR4']:

        if version == 'KV450':
            m = np.array([-0.017, -0.008, -0.015, 0.010, 0.006])
        else:
            m = np.array([-0.009, -0.011, -0.015, 0.002, 0.007])

        z_bin = tomographic_redshift_bin(z_s, version=version)

        return np.where(z_bin != -1, m[z_bin], np.nan)

    else:
        raise RuntimeError(
            "Unkown version of KiDS. Supported versions are {}.".format(
                known_versions))
