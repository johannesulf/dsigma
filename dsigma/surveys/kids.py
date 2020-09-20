import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'multiplicative_shear_bias']

default_version = 'DR3'
known_versions = ['DR3', 'KV450']
e_2_convention = 'flipped'


def default_column_keys(version=None):

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
    else:
        raise RuntimeError(
            "Unkown version of KiDS. Supported versions are {}.".format(
                known_versions))

    return keys


def multiplicative_shear_bias(z_s, version=None):

    if version == 'DR3':
        raise RuntimeError('For DR3, the multiplicative shear bias is ' +
                           'defined for each object individually.')
    elif version == 'KV450':
        m = np.array([-0.017, -0.008, -0.015, 0.010, 0.006])

        z_bins = np.digitize(z_s, np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2 + 1e-6])) - 1

        if np.any(z_bins < 0) or np.any(z_bins > 4):
            raise RuntimeError(
                'The multiplicative shear bias is only defined for source ' +
                'redshifts in the range 0.1 <= z <= 1.2.')
    else:
        raise RuntimeError(
            "Unkown version of KiDS. Supported versions are {}.".format(
                known_versions))

    return m[z_bins]
    
    
