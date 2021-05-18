"""Convenience functions for the dsigma pipeline."""

import numbers
import numpy as np
from astropy.table import Table

from . import surveys

__all__ = ['dsigma_table', 'spherical_to_cartesian', 'cartesian_to_spherical']


def dsigma_table(table, table_type, survey=None, version=None, copy=False,
                 verbose=True, e_2_convention=None, **kwargs):
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
        String describing the specific survey. Providing the survey makes the
        function look for specific keys in the input table.
    version : string, optional
        If a survey is provided, one can specify the version or data release
        of the survey.
    copy : boolean, optional
        Whether the output table shares memory with the input table. Setting to
        False can save memory. However, data will be corupted if the original
        input table is manipulated.
    verbose : boolean, optional
        Whether to output information about the assignments.
    e_2_convention : string, optional
        Whether to switch the sign of e_2 in the input catalog. If 'standard',
        e_2 is not changed. On the other hand, if 'flipped', the sign of e_2
        will be changed. If None, it defaults to 'standard' unless overwritten
        by the specific survey.
    **kwargs : dict, optional
        This function has a set of default keys for columns in the input table
        that it associates with certain data. These default choices might be
        overwritten by survey-specific keys. If that does not work
        sufficiently, keys can be overwritten by the user with this argument.
        For example, ``ra='R.A.'`` would associate the right ascensions with
        the ``R.A.`` key in the input table. Alternatively, one can also
        provide numbers that will be applied to all entries. This only really
        makes sense for weights that one wants to ignore, i.e. ``w_sys=1``.

    Returns
    -------
    table_out : astropy.table.Table
        Table with all necessary data to be parsed by dsigma.
    """

    try:
        table = Table(table, copy=copy)
    except:
        raise Exception(
            "Input table cannot be converted into an astropy table.")

    if table_type not in ['lens', 'source', 'calibration']:
        raise Exception(
            "The table_type argument must be 'lens', 'source' or " +
            "'calibration' but received '{}'.".format(table_type))

    if (survey is not None and survey.lower().split(':')[0] not in
            surveys.__all__):
        raise Exception('Unknown survey. The known surveys are: {}.'.format(
                        ', '.join(surveys.__all__)))

    if (e_2_convention is not None and e_2_convention not in
            ['standard', 'flipped']):
        raise Exception("e_2_convention must be None, 'standard' or " +
                        "flipped but received '{}'".format(e_2_convention))

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

        if version is None:
            version = getattr(surveys, survey.lower()).default_version
        table.meta['version'] = version
        keys.update(getattr(surveys, survey.lower()).default_column_keys(
            version=version))
        if table_type == 'calibration':
            del keys['ra']
            del keys['dec']
            del keys['e_1']
            del keys['e_2']
            del keys['w']
            del keys['m']
            keys.pop('e_rms', None)
            keys.pop('R_2', None)
        if e_2_convention is None:
            e_2_convention = getattr(surveys, survey.lower()).e_2_convention

    # Finally, update with user-specified keys.
    keys.update(**kwargs)

    if verbose:
        print("Assignment for {} table...".format(table_type))
        for output_key, input_key in keys.items():
            print("    {:<10} -> {}".format(output_key, input_key))

    # Assert columns exist, re-name them and drop unnecessary columns.
    for input_key in keys.values():
        try:
            assert input_key in table.colnames or isinstance(input_key,
                                                             numbers.Number)
        except AssertionError:
            raise Exception("Key '{}' must be present in ".format(input_key) +
                            "the input table.")

    # Keep only those columns with relevant data.
    table_out = Table()

    for output_key, input_key in keys.items():
        if isinstance(input_key, numbers.Number):
            table_out[output_key] = np.repeat(input_key, len(table))
        else:
            if isinstance(table[input_key].data, np.ma.MaskedArray):
                if np.ma.is_masked(table[input_key].data):
                    raise RuntimeError('Input table contained masked data.')
                table_out[output_key] = table[input_key].data.filled()
            else:
                table_out[output_key] = table[input_key].data

    if e_2_convention == 'flipped' and table_type == 'source':
        table_out['e_2'] = - table_out['e_2']
        if verbose:
            print("Info: Flipping sign of e_2 component.")

    return table_out


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


def cartesian_to_spherical(x, y, z):
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
    r = np.sqrt(x**2 + y**2 + z**2)
    ra = np.arctan2(y, x)
    ra = np.where(ra < 0, ra + 2 * np.pi, ra)
    dec = np.arcsin(z / r)
    return np.rad2deg(ra), np.rad2deg(dec)
