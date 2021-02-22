"""Jackknife resampling functions."""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from scipy.spatial import cKDTree
from astropy.table import Table
from scipy.ndimage.filters import gaussian_filter

from .helpers import spherical_to_cartesian


__all__ = ['add_continous_fields', 'transfer_continous_fields',
           'jackknife_field_centers', 'add_jackknife_fields',
           'compress_jackknife_fields', 'smooth_correlation_matrix',
           'jackknife_resampling']


def add_continous_fields(table, n_samples=10000, distance_threshold=1):
    """Many surveys target specific patches of the sky that are not connected.
    For assigning jackknife regions, it is often useful to know which objects
    belong to which continous field. If this is not given in the input catalog,
    then this function uses agglomerative clustering algorithm to link nearby
    objects and estimate the fields from the data itself. The resulting labels
    are assigned to the 'field' column.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog containing objects. The catalog needs to have coordinates.
    n_samples : int, optional
        How many points of the original sample to use in the clustering. Note
        that the clustering algorithm can use large amounts of memory for
        n_samples being larger than a few thousand. Use with caution!
    distance_threshold : astropy.units.quantity.Quantity
        The angular separation used to link points. If no unit is given, it is
        interpreted in degrees.

    Returns
    -------
    table : astropy.table.Table
        Catalog with the continous fields written to the column ``field``.

    """

    if not isinstance(distance_threshold, u.quantity.Quantity):
        distance_threshold *= u.deg

    mask = np.random.random(size=len(table)) < n_samples / len(table)
    x, y, z = spherical_to_cartesian(table['ra'][mask].data,
                                     table['dec'][mask].data)
    distance_threshold = np.sqrt(
        2 - 2 * np.cos(distance_threshold.to(u.rad).value))
    table['field'] = np.repeat(-1, len(table))
    table['field'][mask] = AgglomerativeClustering(
        distance_threshold=distance_threshold, n_clusters=None,
        linkage='single').fit_predict(np.vstack((x, y, z)).T)
    table = transfer_continous_fields(table[mask], table)

    return table


def transfer_continous_fields(table_1, table_2):
    """Transfer the field names from one table to another by looking for
    closest neighbors. The field names are stored in the 'field' column.

    Parameters
    ----------
    table_1 : astropy.table.Table
        Catalog containing the fields to be transferred. The catalog needs to
        have coordinates and field IDs.
    table_2 : astropy.table.Table
        Catalog to which fields will be transferred. The catalog needs to have
        coordinates.

    Returns
    -------
    table_2 : astropy.table.Table
        Catalog with the continous fields written to the column ``field``.
    """

    coord_1 = SkyCoord(table_1['ra'], table_1['dec'], unit='deg')
    coord_2 = SkyCoord(table_2['ra'], table_2['dec'], unit='deg')

    idx = coord_2.match_to_catalog_sky(coord_1)[0]
    table_2['field'] = table_1['field'][idx]

    return table_2


def _jackknife_fields_per_field(table, n_jk):
    """Compute the number of jackknife fields per field in a table.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog containing objects. The catalog needs to have field IDs.
    n_jk : int
        Total number of jackknife fields.

    Returns
    -------
    unique_fields : numpy array
        The unique field IDs in the input table.
    n_jk_per_field : numpy array
        The number of jackknife regions in each of the fields. Has the same
        shape as unique_fields.
    """

    unique_fields, counts = np.unique(table['field'], return_counts=True)
    if n_jk < len(unique_fields):
        raise RuntimeError('The number of jackknife regions cannot be ' +
                           'smaller than the number of fields.')

    # Assign the number of jackknife fields according to the total number of
    # objects in each field.
    n_jk_per_field = np.diff(np.rint(
        np.cumsum(counts) / np.sum(counts) * n_jk).astype(np.int), prepend=0)

    # It can happen that one field is assigned 0 jackknife fields. In this
    # case, we will assign 1.
    while np.any(n_jk_per_field == 0):
        n_jk_per_field[np.argmin(n_jk_per_field)] += 1
        n_jk_per_field[np.argmax(n_jk_per_field)] -= 1

    return unique_fields, n_jk_per_field


def jackknife_field_centers(table, n_jk, optimize=False, weight=None):
    """Compute the centers (in cartesian coordinates on a unit sphere) for
    jackknife regions.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog containing objects. The catalog needs to have coordinates and
        field IDs.
    n_jk : int
        Total number of jackknife fields.
    weight : string, optional
        Name of the column to be used as weight when calculating jackknife
        field centers.

    Returns
    -------
    centers : numpy array
        The coordinates of the centers of the jackknife regions. The array has
        shape (n_jk, 3).
    """

    unique_fields, n_jk_per_field = _jackknife_fields_per_field(table, n_jk)

    centers = None

    for field, n in zip(unique_fields, n_jk_per_field):
        mask = table['field'] == field
        kmeans = MiniBatchKMeans(n_clusters=n)
        x, y, z = spherical_to_cartesian(table['ra'][mask].data,
                                         table['dec'][mask].data)
        kmeans.fit(np.vstack((x, y, z)).T,
                   sample_weight=None if weight is None else
                   table[weight][mask])

        if centers is None:
            centers = kmeans.cluster_centers_
        else:
            centers = np.concatenate([centers, kmeans.cluster_centers_])

    return centers


def add_jackknife_fields(table, centers):
    """Assign jackknife regions to all objects in the table. The jackknife
    number is assigned to the column 'field_jk'.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog containing objects. The catalog needs to have coordinates.
    centers : numpy array
        The coordinates of the centers of the jackknife regions. The array has
        shape (n_jk, 3).

    Returns
    -------
    table : astropy.table.Table
        Catalog with the jackknife fields written to the column ``field_jk``.
    """

    x, y, z = spherical_to_cartesian(table['ra'].data, table['dec'].data)
    kdtree = cKDTree(centers)
    table['field_jk'] = kdtree.query(np.vstack([x, y, z]).T)[1]

    return table


def compress_jackknife_fields(table):
    """After assigning jackknife fields, for most applications, we do not need
    information on individual objects anymore. Compress the information in each
    jackknife field by taking weighted averages. The only exception is the
    weight column where the sum is taken.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog containing objects. The catalog needs to have been assigned
        jackknife fields.

    Returns
    -------
    table_jk : astropy.table.Table
        Catalog containing the information for each jackknife field. It has
        exactly as many rows as there are jackknife fields.
    """

    all_field_jk = np.unique(table['field_jk'])
    table_jk = Table(table[:len(all_field_jk)], copy=True)

    for i, field_jk in enumerate(all_field_jk):
        mask = table['field_jk'] == field_jk
        for key in table.colnames:
            if key in ['field', 'field_jk']:
                table_jk[i][key] = table[key][mask][0]
            elif key in ['w_sys', 'sum 1']:
                table_jk[i][key] = np.sum(table[key][mask], axis=0)
            else:
                table_jk[i][key] = np.average(
                    table[key][mask], weights=table['w_sys'][mask], axis=0)

    return table_jk


def smooth_correlation_matrix(cor, sigma, exclude_diagonal=True):
    """Apply a simple gaussian filter on a correlation matrix.

    Parameters
    ----------
    cor : numpy array
        Correlation matrix.
    sigma : int, optional
        Scale of the gaussian filter.
    exclude_diagonal : boolean, optional
        Whether to exclude the diagonal from the smoothing. That is what should
        be done generally because the diagonal is 1 by definition.

    Returns
    -------
    cor_new : numpy array
        Smoothed correlation matrix.
    """

    n_dim = len(np.diag(cor))
    cor_new = np.copy(cor)

    if exclude_diagonal:
        cor_new[0, 0] = 0.5 * (cor[0, 1] + cor[1, 0])
        cor_new[n_dim - 1, n_dim - 1] = 0.5 * (cor[n_dim - 1, n_dim - 2] +
                                               cor[n_dim - 2, n_dim - 1])

        for i in range(1, n_dim - 1):
            cor_new[i, i] = 0.25 * (cor[i, i - 1] + cor[i, i + 1] +
                                    cor[i - 1, i] + cor[i + 1, i])

    cor_new = gaussian_filter(cor_new, sigma, mode='nearest')

    if exclude_diagonal:
        for i in range(n_dim):
            cor_new[i, i] = cor[i, i]

    return cor_new


def jackknife_resampling(function, table_l, table_r=None, table_l_2=None,
                         table_r_2=None, **kwargs):
    """Compute the covariance of the output of a function from jackknife
    re-sampling.

    Parameters
    ----------
    function :
        Function that returns a result for which we want to have uncertainties.
        The function must take exactly one positional argument, the lens table.
        Additionally, it can have several additional keyword arguments.
    table_l : astropy.table.Table
        Precompute results for the lenses. The catalog must have jackknife
        regions assigned to it.
    table_r : optional, astropy.table.Table, optional
        Precompute results for random lenses. The input function must accept
        the random lens table via the 'table_r' keyword argument.
    table_l_2 : optional, astropy.table.Table
        Precompute results for a second set of lenses.The input function must
        accept the second lens table via the 'table_l_2' keyword argument.
    table_r_2 : optional, astropy.table.Table, optional
        Precompute results for a second set of random lenses. The input
        function must accept the second random lens table via the 'table_r_2'
        keyword argument.
    kwargs : dict
        Additional keyword arguments to be passed to the function.

    Returns
    -------
    cov : numpy array
        Covariance matrix of the result derived from jackknife re-sampling.
    """

    samples = []

    for field_jk in np.unique(table_l['field_jk']):

        mask_l = table_l['field_jk'] != field_jk

        for name, table in zip(['table_r', 'table_l_2', 'table_r_2'],
                               [table_r, table_l_2, table_r_2]):
            if table is not None:
                kwargs[name] = table[table['field_jk'] != field_jk]

        samples.append(function(table_l[mask_l], **kwargs))

    return ((len(np.unique(table_l['field_jk'])) - 1) *
            np.cov(np.array(samples), rowvar=False, ddof=0))
