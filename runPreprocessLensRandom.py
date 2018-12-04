#!/usr/bin/env python3
'''
Pre-process the lens and random catalog.
'''

import os
import argparse
import warnings

import numpy as np

from astropy.table import Table, Column

from dsigma.ssp_data import assign_field, S16A_WIDE


def _load_catalogs(args):
    """Read in the lens and random catalogs.
    """
    # Read in the lens and random catalogs
    _, lens_type = os.path.splitext(args.cat_lens)
    _, rand_type = os.path.splitext(args.cat_rand)

    if lens_type == '.fits':
        lens_data = Table.read(args.cat_lens, format='fits')
    elif lens_type == '.npy':
        lens_data = Table(np.load(args.cat_lens))
    else:
        raise Exception("# Wrong type of lens catalog, "
                        "Should be .fits, .npy")

    if rand_type == '.fits':
        rand_data = Table.read(args.cat_rand, format='fits')
    elif rand_type == '.npz':
        rand_data = Table(np.load(args.cat_rand)['random'])
    elif rand_type == '.npy':
        rand_data = Table(np.load(args.cat_rand))
    else:
        raise Exception("# Wrong type of lens catalog, "
                        "Should be .fits, .npy, or .npz")

    return lens_data, rand_data


def _downsample_random(rand_data, args):
    """If necessary, downsample the random catalog.
    """
    if args.nrand < len(rand_data):
        rand_data = Table(np.random.choice(rand_data, args.nrand))
    else:
        warnings.warn("# N_rand should be smaller than len(rand_data)")

    return rand_data


def _check_columns(lens_data, rand_data, args):
    """Make sure ra, dec, and redshift are available.
    """
    # Make sure `ra` and `dec` are available in the catalog.
    assert 'ra' in rand_data.colnames, '`ra` should be in rand catalog'
    assert 'dec' in rand_data.colnames, '`dec` should be in rand catalog'

    # Rename the RA, Dec columns in lens catalog.
    if args.ra_col is not 'ra':
        lens_data.rename_column(args.ra_col, 'ra')
    if args.dec_col is not 'dec':
        lens_data.rename_column(args.dec_col, 'dec')

    # Rename the redshift column in the lens catalog.
    if args.z_col is not 'z':
        lens_data.rename_column(args.z_col, 'z')

    return lens_data, rand_data


def _add_weight(data, args):
    """Add a uniform weight to the data.
    """
    if args.weight not in data.colnames:
        data.add_column(Column(data=np.full(len(data), 1, dtype=int),
                               name=args.weight))

    return data


def _assign_redshift(lens_data, rand_data, args):
    """Assign redshift to the random objects.

    Right now, only make uniform distribution of redshift.
    """
    # Get the minimum and maximum redshift for lens.
    z_min, z_max = np.nanmin(lens_data['z']), np.nanmax(lens_data['z'])
    print("# Min/Max redshift: %6.4f-%6.4f" % (z_min, z_max))

    if 'z' in rand_data.colnames:
        warnings.warn('# `z` exists in random! replace it with a new one!')
        rand_data.remove_column('z')

    if args.match:
        # Match the redshift distribution
        rand_data.add_column(Column(data=np.random.choice(lens_data['z'],
                                                          len(rand_data)),
                                    name='z'))
    else:
        # Add uniform redshift
        rand_data.add_column(Column(data=np.random.uniform(z_min - 0.001,
                                                           z_max + 0.001,
                                                           len(rand_data)),
                                    name='z'))

    return rand_data


def prepare_lens_random(args):
    """Pre-process the lens and random catalog.

    For lens catalog:
        1. Make sure the `field` ID is assigned.
        2. Make sure a `weight` array is added.
        3. Make sure the `ra`, `dec`, and `z` arrays are available.

    For random catalog:
        1. Make sure the `field` ID is assigned.
        2. Make sure a `weight` array is added.
        3. Make sure the `ra`, `dec` are available.
        4. Make sure the redshift range is matched with the lens catalog.
        5. Has the option to match the redshift distribution.
    """
    lens_prefix, _ = os.path.splitext(args.cat_lens)
    rand_prefix, _ = os.path.splitext(args.cat_rand)

    # Read in the data
    lens_data, rand_data = _load_catalogs(args)

    # Check the column names, make sure RA, DEC, and redshift are available
    lens_data, rand_data = _check_columns(lens_data, rand_data, args)

    # If necessary, downsample the random galaxies.
    if args.nrand is not None:
        rand_data = _downsample_random(rand_data, args)

    # Assign field ID.
    # Right now there is just S16A_WIDE
    lens_data = assign_field(lens_data, S16A_WIDE)
    rand_data = assign_field(rand_data, S16A_WIDE)

    # Add weight to the data
    # Right now, we only add uniform weight
    if not args.no_weight:
        lens_data = _add_weight(lens_data, args)
        rand_data = _add_weight(rand_data, args)
    else:
        # Do not change the weight in lens and random catalog
        print("# Do not add weight to lens and random")

    # Assign redshift to random
    if not args.no_redshift:
        rand_data = _assign_redshift(lens_data, rand_data, args)
    else:
        # Do not add redshift to random
        print("# Do not add redshift to randoms")

    print(lens_data.colnames)
    print(rand_data.colnames)

    # Save the results
    if args.save_fits:
        rand_out = rand_prefix + '_prep.fits'
        lens_out = lens_prefix + '_prep.fits'
        rand_data.write(rand_out, format='fits', overwrite=True)
        lens_data.write(lens_out, format='fits', overwrite=True)
    else:
        rand_out = rand_prefix + '_prep.npy'
        lens_out = lens_prefix + '_prep.npy'
        np.save(rand_out, rand_data)
        np.save(lens_out, lens_data)

    return lens_data, rand_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("cat_lens", help="Lens catalog")
    parser.add_argument("cat_rand", help="Random catalog")
    parser.add_argument('-n', '--nrand', type=int,
                        help='Number of random objects',
                        dest='nrand', default=None)
    parser.add_argument('-m', '--match', dest='match',
                        action="store_true",
                        default=False)
    parser.add_argument("-z", '--z_col',
                        help="Column or key name for redshift",
                        default='z_best')
    parser.add_argument("-ra", '--ra_col',
                        help="Column or key name for RA",
                        default='ra')
    parser.add_argument("-dec", '--dec_col',
                        help="Column or key name for Dec",
                        default='dec')
    parser.add_argument("-w", '--weight',
                        help="Column or key name for Dec",
                        default='weight')
    parser.add_argument('-f', '--fits', dest='save_fits',
                        action="store_true",
                        default=False)
    parser.add_argument('--no-weight', dest='no_weight',
                        action="store_true",
                        default=False)
    parser.add_argument('--no-redshift', dest='no_redshift',
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    prepare_lens_random(args)
