"""Convenience functions for the dsigma pipeline."""

import numpy as np
from astropy.table import Table

from . import surveys

__all__ = ['dsigma_table', 'spherical_to_cartesian']


def dsigma_table(table, table_type, survey=None, copy=False, verbose=True,
                 **kwargs):
    """Convenience function to convert a table into an astropy table that is
    easily parsed by ``dsigma``. Specifically, this table will have all
    necessary columns for ``dsigma`` to work.

    Parameters
    ----------
    table : object
        Input table. This can be any object easily converted into an astropy
        table. For example, a structured numpy array, a list of dictionaries
        or an astropy table all work.
    table_type : string
        String describing the table type. Valid choices are 'lens', 'source'
        or 'calibration'.
    survey : string, optional
        String describing the specific survey. The general format is
        '<Survey>:<Version>', i.e. 'HSC:S16A'. Providing the survey makes the
        function look for specific keys in the input table.
    copy : boolean, optional
        Whether the output table shares memory with the input table. Setting to
        False can save memory. However, data will be corupted if the original
        input table is manipulated.
    verbose : boolean, optional
        Whether to output information about the assignments.
    **kwargs : dict, optional
        This function has a set of default keys for columns in the input table
        that it associates with certain data. These default choices might be
        overwritten by survey-specific keys. If that does not work
        sufficiently, keys can be overwritten by the user with this argument.
        For example, ``ra='R.A.'`` would associate the right ascensions with
        the ``R.A.`` key in the input table.

    Returns
    -------
    table : astropy.table.Table
        Table with all necessary data to be parsed by dsigma.
    """

    try:
        table = Table(table, copy=copy)
    except:
        raise Exception(
            "Input table cannot be converted into an astropy table.")

    if table_type not in ['lens', 'source', 'calibration']:
        raise Exception(
            "The catalog+type argument must be 'lens', 'source' or " +
            "'calibration' but received '{}'.".format(table_type))

    if (survey is not None and survey.lower().split(':')[0] not in
            surveys.__all__):
        raise Exception('Unknown survey. The known surveys are: {}.'.format(
                        ', '.join(surveys.__all__)))

    # Set the generic keys.
    keys = {'z': 'z'}

    if table_type != 'calibration':
        keys['ra'] = 'ra'
        keys['dec'] = 'dec'
    if table_type != 'source':
        keys['w_sys'] = 'w_sys'
    if table_type != 'lens':
        keys['w'] = 'w'
    if table_type == 'source':
        keys['e_1'] = 'e_1'
        keys['e_2'] = 'e_2'
    if table_type == 'calibration':
        keys['z_true'] = 'z_true'

    # Overwrite with survey-specific keys.
    if survey is not None:
        table.meta['survey'] = survey
        if survey.split(':')[0].lower() == 'hsc':
            keys.update(surveys.hsc.column_keys())
        else:
            raise Exception("Unknown survey!")

    # Finally, update with user-specified keys.
    keys.update(**kwargs)

    try:
        assert len(np.unique(list(keys.values()))) == len(keys)
    except AssertionError:
        raise Exception("Every column in the input table can correspond to " +
                        "at most one column in the output table.")

    if verbose:
        print("Assignment for {} table...".format(table_type))
        for output_key, input_key in keys.items():
            print("    {:<10} -> {}".format(output_key, input_key))

    # Assert columns exist, re-name them and drop unnecessary columns.
    for input_key in keys.values():
        try:
            assert input_key in table.colnames
        except AssertionError:
            raise Exception("Key '{}' must be present in ".format(input_key) +
                            "the input table.")

    table.keep_columns(list(keys.values()))

    for output_key, input_key in keys.items():
        table.rename_column(input_key, output_key)

    return table


def spherical_to_cartesian(ra, dec):
    """Convert spherical coordinates into cartesian coordinates on a unit
    sphere.

    Parameters
    ----------
    ra, dec : float or numpy array
        Spherical coordinates.

    Returns
    -------
    x, y, z : float or numpy array
        Cartesian coordinates.

    """
    x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    z = np.sin(np.deg2rad(dec))
    return x, y, z
