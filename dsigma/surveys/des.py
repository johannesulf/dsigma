import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'tomographic_redshift_bin']

default_version = 'Y1'
known_versions = ['Y1']
e_2_convention = 'standard'


def default_column_keys(version=default_version):

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
    else:
        raise RuntimeError(
            "Unkown version of DES. Supported versions are {}.".format(
                known_versions))

    return keys


def tomographic_redshift_bin(z_s, version=default_version):
    """DES analyses work in pre-defined tomographic redshift bins. This
    function returns the photometric redshift bin as a function of photometric
    redshift.

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

    if version == 'Y1':
        z_bins = [0.2, 0.43, 0.63, 0.9, 1.3]
    else:
        raise RuntimeError(
            "Unkown version of DES. Supported versions are {}.".format(
                known_versions))

    z_bin = np.digitize(z_s, z_bins) - 1
    z_bin = np.where((z_s < np.amin(z_bins)) | (z_s >= np.amax(z_bins)) |
                     np.isnan(z_s), -1, z_bin)

    return z_bin


def selection_response(table_s):
    """Calculate the DES selection response. See Sheldon & Huff (2017) and
    McClintock et al. (2018) for details.

    Parameters
    ----------
    table_s : astropy.table.Table
        Catalog of sources.

    Returns
    -------
    R_sel : numpy array
        2x2 matrix containing the selection response.
    """

    R_sel = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            use_p = table_s['flags_select_{}p'.format(i + 1)] == 0
            use_m = table_s['flags_select_{}m'.format(i + 1)] == 0
            e = table_s['e_{}'.format(j + 1)]
            R_sel[i, j] = (np.mean(e[use_p]) - np.mean(e[use_m])) / 0.02

    return R_sel
