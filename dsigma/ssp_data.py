"""Functions about HSC SSP Data."""

import os
import warnings

from astropy.table import Table, Column

import numpy as np

__all__ = ["S16A_WIDE", "Field", "prep_catalog_s16a"]


S16A_WIDE = [
    {
        'name': 'GAMA09H',
        'ra_min': 125.0,
        'ra_max': 165.0,
        'dec_min': -3.0,
        'dec_max': 6.0,
        'id': 1
    },
    {
        'name': 'GAMA15H',
        'ra_min': 205.0,
        'ra_max': 228.0,
        'dec_min': -3.0,
        'dec_max': 3.0,
        'id': 2
    },
    {
        'name': 'WIDE12H',
        'ra_min': 165.0,
        'ra_max': 195.0,
        'dec_min': -3.0,
        'dec_max': 3.0,
        'id': 3
    },
    {
        'name': 'HECTOMAP',
        'ra_min': 225.0,
        'ra_max': 255.0,
        'dec_min': 40.0,
        'dec_max': 46.5,
        'id': 4
    },
    {
        'name': 'XMM',
        'ra_min': 27.5,
        'ra_max': 41.0,
        'dec_min': -9.0,
        'dec_max': 2.5,
        'id': 5
    },
    {
        'name': 'VVDS',
        'ra_min': 328.0,
        'ra_max': 351.0,
        'dec_min': -3.0,
        'dec_max': 4.5,
        'id': 6
    },
    {
        'name': 'AEGIS',
        'ra_min': 212.0,
        'ra_max': 216.0,
        'dec_min': 51.5,
        'dec_max': 53.5,
        'id': 7
    },
]


class Field():
    """Class to define HSC SSP field.

    Parameters
    ----------
    ra_min, ra_max : float, float
        Min / Max range of RA

    dec_min, dec_max : float, float
        Min / Max range of Dec

    name : string
        Name of the field

    """

    def __init__(self, ra_min, ra_max, dec_min, dec_max, name):
        """Initialize the object."""
        self.ra_min = ra_min
        self.ra_max = ra_max
        self.dec_min = dec_min
        self.dec_max = dec_max
        self.name = name

        assert ra_min < ra_max, "# Wrong RA range !"
        assert dec_min < dec_max, '# Wrong Dec range !'

    def print_field(self):
        """Just print out the ra, dec range of the field."""
        print("# Field %s: RA %7.4f-%7.4f, Dec %7.4f-%7.4f" % (
            self.name, self.ra_min, self.ra_max, self.dec_min, self.dec_max))

    def filter_radec(self, table, ra_col='ra', dec_col='dec'):
        """Select objects in this field.

        Parameters
        ----------
        table : astropy.table.table.Table
            Input table.

        ra_col : string, optional
            Name of the column for RA (default: 'ra')

        dec_col : string, optional
            Name of the column for Dec (default: 'dec')

        """
        if not isinstance(table, Table):
            raise TypeError("# Input should be astropy table !")

        return ((table[ra_col] >= self.ra_min) &
                (table[ra_col] <= self.ra_max) &
                (table[dec_col] >= self.dec_min) &
                (table[dec_col] <= self.dec_max))


def prep_catalog_s16a(catalog, ra_col='ra', dec_col='dec', field_col='field'):
    """Prepare catalog for HSC SSP S16A datasets.

    Parameters
    ----------
    catalog : string
        Name of the input catalog

    """
    if not os.path.isfile(catalog):
        raise Exception("# Can not find input catalog! ")

    data = Table.read(catalog)
    if (ra_col not in data.colnames) or (dec_col not in data.colnames):
        raise Exception("# Wrong RA, DEC columns !")

    # Make a new field column
    if field_col in data.colnames:
        warnings.warn('# %s exists! Replace it with a new one!' % field_col)
        data.remove_column(field_col)

    data.add_column(Column(data=np.full(len(data), 0), name=field_col))

    # For S16A, we have seven wild fields
    # field_names = ['GAMA09H', 'GAMA15H', 'WIDE12H',
    #                'HECTOMAP', 'XMM', 'VVDS', 'AEGIS']
    for f in S16A_WIDE:
        data[field_col][Field(
            f['ra_min'], f['ra_max'], f['dec_min'],
            f['dec_max'], f['name']).filter_radec(data)] = f['id']

    if np.any(data[field_col] == 0):
        print("# There are %d objects not in any field!" % sum(
            data[field_col] == 0))

    return data


def assign_field(data, ssp_fields,
                 ra_col='ra', dec_col='dec', field_col='field'):
    """Assign field ID to the catalog."""
    # Make a new field column
    if field_col in data.colnames:
        warnings.warn('# %s exists! Replace it with a new one!' % field_col)
        data.remove_column(field_col)

    data.add_column(Column(data=np.full(len(data), 0), name=field_col))

    for f in ssp_fields:
        data[field_col][Field(
            f['ra_min'], f['ra_max'], f['dec_min'],
            f['dec_max'], f['name']).filter_radec(
                data, ra_col=ra_col, dec_col=dec_col)] = f['id']

    if np.any(data[field_col] == 0):
        print("# There are %d objects not in any field!" % sum(
            data[field_col] == 0))

    return data
