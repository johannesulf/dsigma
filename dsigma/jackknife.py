"""Jackknife resampling functions."""
import math

import numpy as np
from numpy.lib.recfunctions import append_fields

import kmeans_radec


__all__ = ["angular_distance", "closest_point", "get_field_id",
           "get_jk_regions_per_field", "add_jackknife_both",
           "add_jackknife_field"]

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

def angular_distance(ra_1, dec_1, ra_arr_2, dec_arr_2, radian=False):
    """Angular distance between coordinates.

    Based on calcDistanceAngle from gglens_dsigma_pz_hsc.py by Hironao Miyatake

    Parameters
    ----------
    ra_1, dec_1 : float, float
        RA, Dec of the first sets of coordinates; can be array.
    ra_arr_2, dec_arr_2 : numpy array, numpy array
        RA, Dec of the second sets of coordinates; can be array
    radian: boolen, option
        Whether the input and output are in radian unit. Default=False

    Return
    ------
        Angular distance in unit of arcsec
    """
    # Convert everything into radian if necessary, and make everything
    # float64 array
    if not radian:
        ra_1 = np.array(ra_1 * D2R, dtype=np.float64)
        dec_1 = np.array(dec_1 * D2R, dtype=np.float64)
        ra_2 = np.array(ra_arr_2 * D2R, dtype=np.float64)
        dec_2 = np.array(dec_arr_2 * D2R, dtype=np.float64)
    else:
        ra_1 = np.array(ra_1, dtype=np.float64)
        dec_1 = np.array(dec_1, dtype=np.float64)
        ra_2 = np.array(ra_arr_2, dtype=np.float64)
        dec_2 = np.array(dec_arr_2, dtype=np.float64)

    if radian:
        return np.arccos(
            np.cos(dec_1) * np.cos(dec_2) * np.cos(ra_1 - ra_2) +
            np.sin(dec_1) * np.sin(dec_2))

    return np.arccos(
        np.cos(dec_1) * np.cos(dec_2) * np.cos(ra_1 - ra_2) +
        np.sin(dec_1) * np.sin(dec_2)) * R2D * 3600.0


def closest_point(ra, dec, ra_arr, dec_arr):
    """Find the closest point.

    Returns index of closest point to (ra, dec) from array of coordinates

    Parameters
    ----------
    ra, dec : float, float
        Coordinate of an object.
    ra_arr, dec_arr : numpy array, numpy array
        Arrays of coordinates.

    Return
    ------
        Index of object from the coordinate array with the closest distance.
    """
    return np.argmin(angular_distance(ra, dec, ra_arr, dec_arr))


def get_field_id(catalog):
    """Get field IDs.

    Get the number of fields available in the catalog.

    Parameters
    ----------
    catalog : numpy array
        Catalog for lenses or random objects.

    Return
    ------
        Unique IDs for different fields.
    """
    return np.unique(catalog['field'])


def get_jk_regions_per_field(catalog, njackfields):
    """Assign jackknife region ID in each field.

    Parameters
    ----------
    catalog : numpy array
        Catalog for lenses or random objects.
    njackfields : int
        Number of Jackknife fields.

    Return
    ------
    """
    # Field IDs
    fields = get_field_id(catalog)

    # Fractions of objects in each fields
    fractions = np.zeros(len(fields))
    for i, field in enumerate(fields):
        objects = catalog[catalog['field'] == field]
        fractions[i] = len(objects) / len(catalog)

    # Perturb the ideal number of jk fields so that no two are identical,
    # unless it is 0
    # Use a constant seed (else np wil pull from /dev/urandom) so this is
    # deterministic
    np.random.seed(0)
    perturbation = ((np.random.random_sample(len(fields)) /
                     len(catalog)) *
                    (fractions != 0)).astype(int)
    float_jk_fields = (fractions * njackfields) + perturbation

    # we should never have to look more than 0.2 around (I think),
    # and this should converge relatively quickly...
    factor, step, max_iterations = 1, 0.2, 20
    for i in range(max_iterations):
        cur_fields = np.ceil(float_jk_fields * factor).astype(int)
        if sum(cur_fields) == njackfields:
            return cur_fields
        elif sum(cur_fields) < njackfields:
            factor += step
        else:
            factor -= step
        step /= 2

    return cur_fields


def add_jackknife_both(lens_ds, rand_ds, njack,
                       lens_ds_2=None, rand_ds_2=None):
    """Assign jackknife regions to random and lens catalogs.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses
    rand_ds : numpy array
        Pre-compute results for randoms
    njack : int
        Number of required jackknife fields
    lens_ds_2 : numpy array, optional
        Second pre-compute results for lenses. Default: None
    rand_ds_2 : numpy array, optional
        Second pre-compute results for randoms. Default: None

    Return
    ------
        Updated lens and random catalogs with `jk_field` information.
    """
    # Get field ID
    fields = get_field_id(lens_ds)

    # Make sure that `jk_field` key is available. If not, add one.
    try:
        lens_ds['jk_field']
    except ValueError:
        lens_ds = append_fields(lens_ds, 'jk_field',
                                np.zeros(len(lens_ds), int),
                                usemask=False)
    # The same for the random catalog.
    try:
        rand_ds['jk_field']
    except ValueError:
        rand_ds = append_fields(rand_ds, 'jk_field',
                                np.zeros(len(rand_ds), int),
                                usemask=False)

    # If njackfields = 1, just add 1 to everything, although that is
    # bad idea for resampling...print out a warning.
    if njack == 1:
        lens_ds['jk_field'] = 1
        rand_ds['jk_field'] = 1
        print("# Only one Jackknife field? Seriously?")
        return lens_ds, rand_ds

    if (lens_ds_2 is not None) and (rand_ds_2 is not None):
        try:
            lens_ds_2['jk_field']
        except ValueError:
            lens_ds_2 = append_fields(lens_ds_2, 'jk_field',
                                      np.zeros(len(lens_ds_2), int),
                                      usemask=False)
        try:
            rand_ds_2['jk_field']
        except ValueError:
            rand_ds_2 = append_fields(rand_ds_2, 'jk_field',
                                      np.zeros(len(rand_ds_2), int),
                                      usemask=False)

        if njack == 1:
            lens_ds['jk_field'] = 1
            rand_ds['jk_field'] = 1
            lens_ds_2['jk_field'] = 1
            rand_ds_2['jk_field'] = 1
            print("# Only one Jackknife field? Seriously?")
            return lens_ds, rand_ds, lens_ds_2, rand_ds_2

    # Use the results with more objects as reference
    # In principle, the random catalog should have many more objects than the lenses
    # TODO: Still, when the number of lens is smaller than a threshold,
    # We should do something else
    if len(rand_ds) > len(lens_ds):
        jk_fields_per_field = get_jk_regions_per_field(rand_ds, njack)
    else:
        jk_fields_per_field = get_jk_regions_per_field(lens_ds, njack)

    jk_next = 0

    if (lens_ds_2 is not None) and (rand_ds_2 is not None):
        # Make sure both pre-compute results share the same jackknife fields
        for i, field in enumerate(fields):
            rand_mask = rand_ds['field'] == field
            lens_mask = lens_ds['field'] == field

            rand_mask_2 = rand_ds_2['field'] == field
            lens_mask_2 = lens_ds_2['field'] == field

            if sum(rand_mask) == 0:
                continue

            rand_field = rand_ds[rand_mask]
            lens_field = lens_ds[lens_mask]

            rand_field_2 = rand_ds_2[rand_mask_2]
            lens_field_2 = lens_ds_2[lens_mask_2]

            # perform kmeans
            radec = np.column_stack((rand_field['ra'], rand_field['dec']))
            km = kmeans_radec.kmeans_sample(
                radec, jk_fields_per_field[i], maxiter=100,
                tol=1.0e-5, verbose=False)

            # assign jk_field, shifting up by jk_next as labels are 0-n
            rand_field['jk_field'] = km.labels + jk_next

            # kmeans centers
            ra_centers = np.array([k[0] for k in km.centers])
            dec_centers = np.array([k[1] for k in km.centers])

            # assign jackknife field in lens catalog based on nearest kmeans center
            for lens in lens_field:
                closest_jk = closest_point(lens['ra'], lens['dec'],
                                           ra_centers, dec_centers)
                lens['jk_field'] = closest_jk + jk_next

            for lens in lens_field_2:
                closest_jk = closest_point(lens['ra'], lens['dec'],
                                           ra_centers, dec_centers)
                lens['jk_field'] = closest_jk + jk_next

            for rand in rand_field_2:
                closest_jk = closest_point(rand['ra'], rand['dec'],
                                           ra_centers, dec_centers)
                rand['jk_field'] = closest_jk + jk_next

            # increment jk_next so that next field has higher jk numbers
            jk_next += jk_fields_per_field[i]

            # write back to the catalog.
            # Do this rather than recreating so that order is preserved
            rand_ds['jk_field'][rand_mask] = rand_field['jk_field']
            lens_ds['jk_field'][lens_mask] = lens_field['jk_field']

            rand_ds_2['jk_field'][rand_mask_2] = rand_field_2['jk_field']
            lens_ds_2['jk_field'][lens_mask_2] = lens_field_2['jk_field']

        return lens_ds, rand_ds, lens_ds_2, rand_ds_2
    else:
        for i, field in enumerate(fields):
            rand_mask = rand_ds['field'] == field
            lens_mask = lens_ds['field'] == field

            if sum(rand_mask) == 0 and sum(lens_mask) == 0:
                continue

            rand_field = rand_ds[rand_mask]
            lens_field = lens_ds[lens_mask]

            # perform kmeans
            radec = np.column_stack((rand_field['ra'], rand_field['dec']))
            km = kmeans_radec.kmeans_sample(
                radec, jk_fields_per_field[i], maxiter=100,
                tol=1.0e-5, verbose=False)

            # assign jk_field, shifting up by jk_next as labels are 0-n
            rand_field['jk_field'] = km.labels + jk_next

            # kmeans centers
            ra_centers = np.array([k[0] for k in km.centers])
            dec_centers = np.array([k[1] for k in km.centers])

            # assign jackknife field in lens catalog based on nearest kmeans center
            for lens in lens_field:
                closest_jk = closest_point(lens['ra'], lens['dec'],
                                           ra_centers, dec_centers)
                lens['jk_field'] = closest_jk + jk_next

            # increment jk_next so that next field has higher jk numbers
            jk_next += jk_fields_per_field[i]

            # write back to the catalog.
            # Do this rather than recreating so that order is preserved
            rand_ds['jk_field'][rand_mask] = rand_field['jk_field']
            lens_ds['jk_field'][lens_mask] = lens_field['jk_field']

        return lens_ds, rand_ds


def add_jackknife_field(catalog, njackfields):
    """Assign jackknife regions to random or lens catalogs.

    Parameters
    ----------
    catalog : numpy array
        Lens or random catalog.
    njackfields : int
        Number of Jackknife resampling fields.

    Return
    ------
        Catalog with `jk_field` column.
    """
    # Get field ID
    fields = get_field_id(catalog)

    # Make sure that `jk_field` key is available. If not, add one.
    try:
        catalog['jk_field']
    except ValueError:
        catalog = append_fields(catalog, 'jk_field',
                                np.zeros(len(catalog), int),
                                usemask=False)

    # If njackfields = 1, just add 1 to everything, although that is
    # bad idea for resampling...print out a warning.
    if njackfields == 1:
        catalog['jk_field'] = 1
        print("# Only one Jackknife field? Seriously?")
        return catalog

    # Calculate the number of Jackknife regions per field
    jk_fields_per_field = get_jk_regions_per_field(catalog, njackfields)
    jk_next = 0

    for i, field in enumerate(fields):
        indexes = (catalog['field'] == field).nonzero()
        new_catalog = catalog[indexes]

        # perform kmeans
        radec = np.column_stack((new_catalog['ra'], new_catalog['dec']))
        km = kmeans_radec.kmeans_sample(radec,
                                        jk_fields_per_field[i], maxiter=100,
                                        tol=1.0e-5, verbose=False)

        # assign jk_field, shifting up by jk_next as labels are 0-n
        new_catalog['jk_field'] = km.labels + jk_next

        # Write back to the catalog.
        # Do this rather than recreating so that order is preserved
        catalog['jk_field'][indexes] = new_catalog['jk_field']

        # increment jk_next so that next field has higher jk numbers
        jk_next += jk_fields_per_field[i]

    return catalog
