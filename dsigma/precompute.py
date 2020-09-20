from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import units as u

from .physics import mpc_per_degree, critical_surface_density
from .physics import projection_angle_sin_cos
from .helpers import spherical_to_cartesian
from . import surveys

__all__ = ["add_maximum_lens_redshift", "precompute_catalog",
           "merge_precompute_catalogs"]


def _search_around_sky(ra, dec, kdtree, rmin, rmax):
    """Cross-match coordinates.

    Parameters
    ----------
    ra, dec : float
        Coordinates around which objects are searched.
    kdtree : scipy.spatial.cKDTree
        KDTree containing the objects to be searched.
    rmin : float, optional
        Minimum radius to search for objects in degrees.
    rmax : float, optional
        Maximum radius to search for objects in degrees.

    Returns
    -------
    idx : numpy array
        Indices of objects in the KDTree that lie in the search area.
    theta : numpy array
        The distance to the search center in degrees.
    """

    # Convert the angular distance into a 3D distance on the unit sphere.
    rmax_3d = np.sqrt(2 - 2 * np.cos(np.deg2rad(rmax)))

    x, y, z = spherical_to_cartesian(ra, dec)
    idx = np.fromiter(kdtree.query_ball_point([x, y, z], rmax_3d),
                      dtype=np.int64)

    # Convert 3D distance back into angular distances.
    if len(idx) != 0:
        dx = x - kdtree.data[idx, 0]
        dy = y - kdtree.data[idx, 1]
        dz = z - kdtree.data[idx, 2]
        dist3d = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        dist3d = np.zeros(0)
    theta = np.rad2deg(np.arcsin(dist3d * np.sqrt(1 - dist3d**2 / 4)))
    mask = theta > rmin

    return idx[mask], theta[mask]


def _photo_z_dilution_factor(z_l, d_l, table_c):
    """Calculate the inverse of the photo-z bias factor.

    Parameters
    ----------
    z_l : float
        Redshift of the lens.
    d_l : float
        Comoving distance to the lens.
    table_c : astropy.table.Table, optional
        Photometric redshift calibration catalog.

    Returns
    -------
        The denominator and numberator of the photo-z bias factor, `f_bias`.
    """

    sigma_crit_phot = critical_surface_density(
        z_l, table_c['z'], d_l=d_l, d_s=table_c['d_com'])
    sigma_crit_true = critical_surface_density(
        z_l, table_c['z_true'], d_l=d_l, d_s=table_c['d_com_true'])
    mask = z_l < table_c['z_l_max']

    return (np.sum((table_c['w_sys'] * table_c['w'] / sigma_crit_phot /
                    sigma_crit_true)[mask]),
            np.sum((table_c['w_sys'] * table_c['w'] /
                    sigma_crit_phot**2)[mask]))


def add_maximum_lens_redshift(table_s, dz_min=0.0, z_err_factor=0,
                              apply_z_low=False):
    """For each source in the table, determine the maximum lens redshift
    :math:`z_{\mathrm{max}}`. During the precomputation phase, only lens-source
    pairs with :math:`z_{\mathrm{l}} \leq z_{\mathrm{max}}` are being used. The
    maximum redshift is the minimum of the following three quantities where
    :math:`z_{\mathrm{s}}` is the source redshift.

    1. :math:`z_{\mathrm{s}} - \Delta z_{\mathrm{min}}`
    2. :math:`z_{\mathrm{s}} - \sigma_z r`
    3. :math:`z_{\mathrm{s, low}}`

    Parameters
    ----------
    table_s : astropy.table.Table
        Catalog of weak lensing sources.
    dz_min : float, optional
        Minimum redshift separation :math:`\Delta z_{\mathrm{min}}` between
        lens and source.
    z_err_factor : float, optional
        Minimum redshift separation :math:`r` in units of the source redshift
        error.
    apply_z_low : boolean, optional
        Whether to apply the cut on :math:`z_{\mathrm{s, low}}`, the lower
        limit on the redshift of the source stored in column :code:`z_low`.

    Returns
    -------
    table_s : astropy.table.Table
        Table with the maximum lens redshfit assigned to the :code:`z_l_max`
        column.
    """

    z_l_max = table_s['z'] - dz_min

    if z_err_factor > 0:
        z_l_max = np.minimum(
            z_l_max, table_s['z'] - z_err_factor * table_s['z_err'])

    if apply_z_low:
        z_l_max = np.minimum(z_l_max, table_s['z_low'])

    table_s['z_l_max'] = z_l_max

    return table_s


def _precompute_core(lens, table_s, rp, rp_bins, comoving=True):
    """Perform the precomputation for a single lens and all the sources
    specified in the source catalog. For this function to work correctly, all
    sources in the source catalog must be within the bins in projected
    distance defined by 'rp_bins'.

    Parameters
    ----------
    lens : astropy.table.row.Row
        Information about a single lens.
    table_s : dict
        Catalog of weak lensing sources.
    rp : numpy array
        Projected distances of all sources in Mpc.
    rp_bins : numpy array
        Bins in projected radius (in Mpc) to use for the stacking.
    comoving : boolean
        Whether to use comoving coordinates.

    Returns
    -------
    result : dict
        Pre-compute result for a single lens.
    """

    result = {}

    # Calculate the critical surface density.
    sigma_crit = critical_surface_density(
        lens['z'], table_s['z'], comoving=comoving, d_l=lens['d_com'],
        d_s=table_s['d_com'])

    # Calculate the projection angle for each lens-source pair.
    cos2phi, sin2phi = projection_angle_sin_cos(
        lens['sin ra'], lens['cos ra'], lens['sin dec'], lens['cos dec'],
        table_s['sin ra'], table_s['cos ra'], table_s['sin dec'],
        table_s['cos dec'])

    # Calculate the tangential and cross shear terms.
    e_t = - table_s['e_1'] * cos2phi - table_s['e_2'] * sin2phi
    e_x = + table_s['e_1'] * sin2phi - table_s['e_2'] * cos2phi

    # The weight of each lens-source pair is weighted by the critical surface
    # density.
    w_ls = table_s['w'] / sigma_crit**2
    sigma_crit_w_ls = table_s['w'] / sigma_crit

    rp_binned = np.digitize(rp, rp_bins) - 1
    sum_s = partial(np.bincount, rp_binned, minlength=len(rp_bins) - 1)

    result['sum w_s e_t'] = sum_s(weights=e_t * table_s['w'])
    result['sum w_s e_x'] = sum_s(weights=e_x * table_s['w'])
    result['sum w_s'] = sum_s(weights=table_s['w'])

    result['sum w_ls e_t sigma_crit'] = sum_s(weights=e_t * sigma_crit_w_ls)
    result['sum w_ls e_x sigma_crit'] = sum_s(weights=e_x * sigma_crit_w_ls)
    result['sum w_ls'] = sum_s(weights=w_ls)

    # This is used to calculate the naive error for the lensing signal.
    result['sum (w_ls e_t sigma_crit)^2'] = sum_s(
        weights=(e_t * sigma_crit_w_ls)**2)
    result['sum (w_ls e_x sigma_crit)^2'] = sum_s(
        weights=(e_x * sigma_crit_w_ls)**2)

    # Multiplicative bias m.
    if 'm' in table_s.keys():
        result['sum w_ls m'] = sum_s(weights=w_ls * table_s['m'])

    # Responsivity R.
    if 'sigma_rms' in table_s.keys():
        result['sum w_ls (1 - sigma_rms^2)'] = sum_s(
            weights=w_ls * (1 - table_s['sigma_rms']**2))

    # Square term about the shape noise.
    result['sum w_s e_t^2'] = sum_s(weights=e_t**2 * table_s['w'])
    result['sum w_s e_x^2'] = sum_s(weights=e_x**2 * table_s['w'])

    # Number of pairs in each radial bin.
    result['sum 1'] = sum_s()

    # Calculate the sum of lens-source distance in each bin.
    result['sum r_p'] = sum_s(weights=rp)

    # Resolution selection bias.
    if 'R_2' in table_s.keys():
        result.update(surveys.hsc.precompute_selection_bias_factor(
            table_s['R_2'], w_ls, rp_binned, len(rp_bins) - 1))

    # METACALIBRATION response.
    if 'R_MCAL' in table_s.keys():
        result['sum w_ls R_MCAL'] = sum_s(weights=w_ls * table_s['R_MCAL'])

    return result


def _precompute_single(lens, table_s, kdtree_s, rp_bins,
                       cosmology=FlatLambdaCDM(H0=100, Om0=0.3),
                       comoving=True):
    """Perform the precomputation for a single lens and all the sources in the
    source catalog.

    Parameters
    ----------
    lens : astropy.table.row.Row
        Information about a single lens.
    table_s : astropy.table.Table
        Catalog of weak lensing sources.
    kdtree_s : scipy.spatial.cKDTree
        KDTree of the cartesian coordinates of all sources.
    rp_bins : numpy array
        Bins in projected radius (in Mpc) to use for the stacking.
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations.
    comoving : boolean, optional
        Whether to use comoving or physical quantities.

    Returns
    -------
        Precompute result for single lens.
    """

    # Compute the conversion from angles to physical scales.
    mpc_deg = mpc_per_degree(cosmology, lens['z'], comoving=comoving)

    rmin = np.amin(rp_bins) / mpc_deg
    rmax = np.amax(rp_bins) / mpc_deg
    idx, theta = _search_around_sky(lens['ra'], lens['dec'], kdtree_s, rmin,
                                    rmax)

    # Apply additional photo-z cuts.
    mask = lens['z'] < table_s['z_l_max'][idx]
    idx = idx[mask]
    theta = theta[mask]

    table_s_part = {}
    for col in table_s.colnames:
        if col in ['sin ra', 'cos ra', 'sin dec', 'cos dec', 'z', 'd_com',
                   'w', 'e_1', 'e_2', 'sigma_rms', 'm', 'R_2', 'R_MCAL']:
            table_s_part[col] = table_s[col].data[idx]

    rp = theta * mpc_deg

    return _precompute_core(lens, table_s_part, rp, rp_bins, comoving=comoving)


def precompute_catalog(table_l, table_s, rp_bins, table_c=None,
                       cosmology=FlatLambdaCDM(H0=100, Om0=0.3),
                       comoving=True, n_jobs=1, table_s_chunk_size=10000000,
                       trim=True):
    """For all lenses in the catalog, perform the precomputation of lensing
    statistics.

    Parameters
    ----------
    table_l : astropy.table.Table
        Catalog of lenses.
    table_s : astropy.table.Table
        Catalog of sources.
    rp_bins : numpy array
        Bins in projected radius (in Mpc) to use for the stacking.
    table_c : astropy.table.Table, optional
        Additional photometric redshift calibration catalog.
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations.
    comoving : boolean, optional
        Whether to use comoving or physical quantities.
    n_jobs : int, optional
        Number of jobs to run at the same time.
    table_s_chunk_size : int, optional
        Maximum number of sources to be processed simultaneously. This is only
        used when more than one job is run at the same time. Larger numbers
        might result in shorter runtime but more memory use when running in
        parralel. Also, the program might crash if the number is too large
        due to limitations of the multiprocessing library.
    trim : boolean, optional
        If set to true, the output table will omit lenses that do not have any
        nearby sources.

    Returns
    -------
    table_p : astropy.table.Table
        Results of the precomputation. The table has the same ordering as the
        lens catalog if trim is set to false.
    """

    try:
        assert cosmology.Ok0 == 0
    except AssertionError:
        raise Exception('Currently, dsigma does not support non-flat ' +
                        'cosmologies.')

    try:
        assert np.all(table_l['z'] > 0)
    except AssertionError:
        raise Exception('The redshifts of all lens galaxies must be positive.')
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise Exception('Illegal number of jobs. Expected positive integer ' +
                        'but received {}.'.format(n_jobs))

    if not isinstance(table_s_chunk_size, int) or table_s_chunk_size < 0:
        raise Exception('Illegal maximum number of sources to be processed ' +
                        'at the same time. Expected positive integer ' +
                        'but received {}.'.format(table_s_chunk_size))

    for table in [table_l, table_s, table_c]:
        if table is not None and 'd_com' not in table.colnames:
            table['d_com'] = cosmology.comoving_transverse_distance(
                table['z']).to(u.Mpc).value

    if np.any(table_l['z'] < 0):
        raise Exception('Input lens redshifts must all be non-negative.')

    if table_c is not None:
        if 'd_com_true' not in table_c.colnames:
            table_c['d_com_true'] = cosmology.comoving_transverse_distance(
                table_c['z_true']).to(u.Mpc).value

    if 'w_sys' not in table_l.colnames:
        print("Warning: Could not find systematic weights for lenses. " +
              "Weights can be specified in the `w` column of the lens" +
              "table. Weights are set to unity.")
        table_l['w_sys'] = 1

    for table in [table_s, table_c]:
        if table is not None:
            if 'z_l_max' not in table.colnames:
                print("Warning: Could not find a lens-source separation cut." +
                      " Thus, only z_l < z_s is required. Consider running " +
                      "`add_maximum_lens_redshift` to define a lens-source " +
                      "separation cut.")
                table['z_l_max'] = table['z']

    if trim:
        coord_s = SkyCoord(ra=table_s['ra'], dec=table_s['dec'], unit='deg')
        coord_l = SkyCoord(ra=table_l['ra'], dec=table_l['dec'], unit='deg')

        idx, d2d, d3d = coord_l.match_to_catalog_sky(coord_s)
        alpha_max = np.amax(rp_bins) / table_l['d_com'] * u.rad
        if not comoving:
            alpha_max *= (1 + table_l['z'])
        table_l = table_l[d2d < alpha_max]

    for f in [np.sin, np.cos]:
        for table in [table_s, table_l]:
            for angle in ['ra', 'dec']:
                col = f.__name__ + ' ' + angle
                if col not in table.colnames:
                    table[col] = f(np.deg2rad(table[angle]))

    # If we only use one thread, we do not need to split the lens table.
    if n_jobs == 1:

        x, y, z = spherical_to_cartesian(table_s['ra'], table_s['dec'])
        kdtree_s = cKDTree(np.column_stack([x, y, z]), leafsize=1000)

        results = []

        for lens in table_l:
            results.append(_precompute_single(
                lens, table_s, kdtree_s, rp_bins, cosmology=cosmology,
                comoving=comoving))

        # Convert the list of results into a table.
        table_r = Table(rows=results)
        for key in table_r.colnames:
            table_l[key] = table_r[key]
        table_l.meta['rp_bins'] = rp_bins
        table_l.meta['comoving'] = comoving
        table_l.meta['H0'] = cosmology.H0.value
        table_l.meta['Ok0'] = cosmology.Ok0
        table_l.meta['Om0'] = cosmology.Om0

        # If necessary, estimate the photo-z bias factor.
        if table_c is not None:
            f_bias = np.array([_photo_z_dilution_factor(z_l, d_l, table_c) for
                               z_l, d_l in zip(table_l['z'], table_l['d_com'])])
            table_l['calib: sum w_ls w_c sigma_crit_p / sigma_crit_t'] = f_bias[:, 0]
            table_l['calib: sum w_ls w_c'] = f_bias[:, 1]

        return table_l

    # If running more than one thread, simply split up the lens table into
    # equal parts.
    else:

        # If the source table is too large in size, the multiprocessing
        # module might crash because the arguments are not pickable. In this
        # case, let's split up the source table and merge the precompute
        # tables.
        if len(table_s) > table_s_chunk_size:
            table_p_1 = precompute_catalog(
                table_l, table_s[0::2], rp_bins, table_c=table_c,
                cosmology=cosmology, comoving=comoving, n_jobs=n_jobs,
                table_s_chunk_size=table_s_chunk_size, trim=False)
            table_p_2 = precompute_catalog(
                table_l, table_s[1::2], rp_bins, table_c=table_c,
                cosmology=cosmology, comoving=comoving, n_jobs=n_jobs,
                table_s_chunk_size=table_s_chunk_size, trim=False)
            return merge_precompute_catalogs(table_p_1, table_p_2)

        else:
            # Prepare jobs to be submitted to individual single threads.
            precompute_catalog_partial = partial(
                precompute_catalog, table_s=table_s, table_c=table_c,
                rp_bins=rp_bins, cosmology=cosmology, comoving=comoving,
                n_jobs=1, trim=False)

            # Create a random split of the lens catalog. A non-random split
            # might lead to uneven load distribution in case the lens table is
            # sorted by properties that correlate with the compuational time
            # for each lens, e.g. redshift.
            idx = np.random.choice(np.arange(len(table_l)), size=len(table_l),
                                   replace=False)
            table_l_chunks = []
            for i in range(n_jobs):
                table_l_chunks.append(table_l[np.array_split(idx, n_jobs)[i]])

            with Pool(processes=n_jobs) as pool:
                table_p = vstack(pool.map(precompute_catalog_partial,
                                          table_l_chunks))

            # vstack joins arrays in the meta-data from different tables
            # together. Revert this change.
            table_p.meta['rp_bins'] = rp_bins

            # Undo the random split and shuffling when returning the result
            # table.
            return table_p[np.argsort(idx)]


def merge_precompute_catalogs(table_p_1, table_p_2):
    """Merge precompute results for the same lenses and different sources.

    Parameters
    ----------
    table_p_1 : astropy.table.Table
        First catalog of precompute results.
    table_p_2 : astropy.table.Table
        Second catalog of precompute results. Is assumed to have the same
        ordering as first table.

    Returns
    -------
    table_p : astropy.table.Table
        Merged catalog of precompute results.
    """

    table_p = Table()
    table_p.meta = table_p_1.meta

    for key in ['rp_bins', 'comoving', 'H0', 'Ok0', 'Om0']:
        if not np.all(table_p_1.meta[key] == table_p_2.meta[key]):
            raise RuntimeError(
                'Inconsistent meta-data for key {}.'.format(key))

    for key in table_p_1.colnames:

        if key not in table_p_2.colnames:
            raise RuntimeError('Key {} present in table 1 but '.format(key) +
                               'not in table_2.')

        if 'sum' not in key:
            if np.any(table_p_1[key] != table_p_2[key]):
                raise RuntimeError('Mismatch between table 1 and 2 for key' +
                                   '{}'.format(key))
            table_p[key] = table_p_1[key]
        else:
            table_p[key] = table_p_1[key] + table_p_2[key]

    return table_p
