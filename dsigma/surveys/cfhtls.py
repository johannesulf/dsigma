"""Module with functions specific to the CFHTLenS survey."""

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys']

e_2_convention = 'standard'
default_version = None
known_versions = [None, ]


def default_column_keys(version=None):
    """Return a dictionary of default column keys.

    Parameters
    ----------
    version : optional
        Ignored.

    Returns
    -------
    keys : dict
        Dictionary of default column keys.

    """
    keys = {
        'ra': 'ALPHA_J2000',
        'dec': 'DELTA_J2000',
        'z': 'Z_B',
        'z_low': 'Z_B_MIN',
        'e_1': 'e1',
        'e_2': 'e2',
        'w': 'weight',
        'm': 'm'}
    return keys
