"""Module containing jackknife resampling functions."""

from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.table import Table
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, MiniBatchKMeans

from .helpers import in_degrees, spherical_to_cartesian

__all__ = ['compress_jackknife_fields', 'compute_jackknife_fields',
           'jackknife_resampling', 'smooth_correlation_matrix']


def compute_jackknife_fields(table, centers, distance_threshold=1,
                             weights=None, seed=None):
    """Compute the centers for jackknife regions using DBSCAN and KMeans.

    The function first runs DBSCAN to identify continuous fields of points.
    Afterwards, KMeans clustering is run. The initial cluster centers are
    random points from each continuous field. The number of initial cluster
    centers per field is determined according to the total weight of each
    continuous field. The centers are defined in cartesian coordinates on a unit
    sphere.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog containing objects. The catalog needs to have coordinates and
        field IDs. The jackknife field for each galaxy will be added in the
        `field_jk` column.
    centers : int or numpy.ndarray
        If a number, total number of jackknife fields. Otherwise, the centers
        returned from a previous call to that function. This allows for
        different samples to have the same jackknife fields.
    distance_threshold : float, optional
        The angular separation in degrees used to link points and calculate
        continuous fields before running KMeans. Default is 1.
    weights : numpy.ndarray or None, optional
        Per-lens weights for clustering. If ``None``, assume the same weight
        for all points. Default is ``None``.
    seed : int or None, optional
        Random seed to initialize the random number generator. Default is
        ``None``.

    Returns
    -------
    centers : numpy.ndarray
        The coordinates of the centers of the jackknife regions.

    """
    xyz = np.column_stack(spherical_to_cartesian(
        in_degrees(table['ra'].quantity), in_degrees(table['dec'].quantity)))

    if isinstance(centers, np.ndarray):
        kdtree = cKDTree(centers)
        table['field_jk'] = kdtree.query(xyz)[1]
        return centers

    if weights is None:
        weights = np.ones(len(table))

    n_jk = centers

    if not isinstance(distance_threshold, u.quantity.Quantity):
        distance_threshold *= u.deg

    eps = np.sqrt(
        2 - 2 * np.cos(distance_threshold.to(u.rad).value))
    c = DBSCAN(eps=eps, algorithm='kd_tree').fit(xyz).labels_

    w_c = np.bincount(c[c != -1], weights=weights[c != -1])
    if n_jk < len(w_c):
        msg = ("The number of jackknife regions cannot be smaller than the "
               "number of continuous fields. Try increasing "
               "`distance_threshold` or decreasing `centers`.")
        raise RuntimeError(msg)

    # Assign the number of jackknife fields according to the total number of
    # objects in each field.
    n_jk_per_c = np.diff(np.rint(
        np.cumsum(w_c) / np.sum(w_c) * n_jk).astype(int), prepend=0)

    # It can happen that one field is assigned 0 jackknife fields. In this
    # case, we will assign 1.
    while np.any(w_c[n_jk_per_c == 0] > 0):
        n_jk_per_c[np.argmin(n_jk_per_c)] += 1
        n_jk_per_c[np.argmax(n_jk_per_c)] -= 1

    rng = np.random.default_rng(seed)

    init = np.zeros((0, 3))
    for i in range(len(w_c)):
        mask = i != c
        if w_c[i] > 0:
            init = np.vstack([init, xyz[~mask][rng.choice(
                np.sum(~mask), n_jk_per_c[i], replace=False,
                p=weights[~mask] / w_c[i])]])

    kmeans = MiniBatchKMeans(n_clusters=n_jk, init=init, n_init=1,
                             random_state=int(rng.integers(2**31)))
    centers = kmeans.fit(
        xyz[weights > 0], sample_weight=weights[weights > 0]).cluster_centers_
    compute_jackknife_fields(table, centers)

    return centers


def compress_jackknife_fields(table):
    """Sum together all lenses in each jackknife field.

    After assigning jackknife fields, for most applications, we do not need
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
    table = table.copy()
    table.sort('field_jk')
    table_jk = Table()
    table_jk.meta = deepcopy(table.meta)

    table_jk['field_jk'], counts = np.unique(
        table['field_jk'], return_counts=True)

    for key in table.colnames:

        if not (key in ['w_sys', 'ra', 'dec', 'z'] or key[:3] == 'sum'):
            continue

        table_jk[key] = np.zeros((len(table_jk), ) + table[key].shape[1:],
                                 dtype=table[key].dtype)

        for i in range(len(table_jk)):
            k_min = 0 if i == 0 else np.cumsum(counts)[i - 1]
            k_max = np.cumsum(counts)[i]
            if key == 'w_sys':
                table_jk[key][i] = np.sum(table[key][k_min:k_max])
            elif key == 'sum 1':
                table_jk[key][i] = np.sum(table[key][k_min:k_max], axis=0)
            elif np.sum(table['w_sys'][k_min:k_max]) > 0:
                table_jk[key][i] = np.average(
                    table[key][k_min:k_max],
                    weights=table['w_sys'][k_min:k_max], axis=0)

    return table_jk


def smooth_correlation_matrix(cor, sigma, exclude_diagonal=True):
    """Apply a simple gaussian filter on a correlation matrix.

    Parameters
    ----------
    cor : numpy.ndarray
        Correlation matrix.
    sigma : int
        Scale of the gaussian filter.
    exclude_diagonal : bool, optional
        Whether to exclude the diagonal from the smoothing. That is what should
        be done generally because the diagonal is 1 by definition. Default is
        ``True``.

    Returns
    -------
    cor_new : numpy.ndarray
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


def jackknife_resampling(f, table_l, table_r=None, table_l_2=None,
                         table_r_2=None, compress=True, **kwargs):
    """Compute the covariance of a function from jackknife re-sampling.

    Parameters
    ----------
    f : callable
        Function that returns a result for which we want to have uncertainties.
        The function must take exactly one positional argument, the lens table.
        Additionally, it can have several additional keyword arguments.
    table_l : astropy.table.Table
        Precompute results for the lenses. The catalog must have jackknife
        regions assigned to it.
    table_r : astropy.table.Table or None, optional
        Precompute results for random lenses. The input function must accept
        the random lens table via the ``table_r`` keyword argument. Default
        is ``None``.
    table_l_2 : astropy.table.Table or None, optional
        Precompute results for a second set of lenses.The input function must
        accept the second lens table via the ``table_l_2`` keyword argument.
        Default is ``None``.
    table_r_2 : astropy.table.Table or None, optional
        Precompute results for a second set of random lenses. The input
        function must accept the second random lens table via the ``table_r_2``
        keyword argument. Default is ``None``.
    compress : bool, optional
        If ``True``, compress jackknife fields via
        ``dsigma.jackknife.compress_jackknife_fields`` before performing the
        jackknife calculation. This can substantially improve performance.
        Default is ``True``.
    **kwargs
        Additional keyword arguments to be passed to the function.

    Returns
    -------
    cov : numpy.ndarray
        Covariance matrix of the result derived from jackknife re-sampling.

    """
    samples = []

    if compress:
        table_l = compress_jackknife_fields(table_l)
        if table_r is not None:
            table_r = compress_jackknife_fields(table_r)
        if table_l_2 is not None:
            table_l_2 = compress_jackknife_fields(table_l_2)
        if table_r_2 is not None:
            table_r_2 = compress_jackknife_fields(table_r_2)

    for field_jk in np.unique(table_l['field_jk']):
        for name, table in zip(['table_r', 'table_l_2', 'table_r_2'],
                               [table_r, table_l_2, table_r_2]):
            if table is not None:
                kwargs[name] = table[table['field_jk'] != field_jk]

        samples.append(f(table_l[table_l['field_jk'] != field_jk], **kwargs))

    return ((len(np.unique(table_l['field_jk'])) - 1) *
            np.cov(np.array(samples), rowvar=False, ddof=0))
