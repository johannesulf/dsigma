from functools import partial
from multiprocessing import Pool

import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

from astropy.table import Table, vstack, hstack
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord

from .physics import mpc_per_degree, critical_surface_density
from .physics import projection_angle_sin_cos
from .physics import effective_critical_surface_density
from .helpers import spherical_to_cartesian
from . import surveys
from .jackknife import compress_jackknife_fields as compress_jackknife_fields_function

import time


__all__ = ["add_maximum_lens_redshift", "precompute_photo_z_dilution_factor",
           "precompute_catalog", "merge_precompute_catalogs"]


precompute_keys = [
    'sum w_s e_t', 'sum w_s e_x', 'sum w_s', 'sum w_ls e_t sigma_crit',
    'sum w_ls e_x sigma_crit', 'sum w_ls', 'sum (w_ls e_t sigma_crit)^2',
    'sum (w_ls e_x sigma_crit)^2', 'sum w_ls m',
    'sum w_ls (1 - sigma_rms^2)', 'sum w_s e_t^2', 'sum w_s e_x^2',
    'sum 1', 'sum w_ls A p(R_2=0.3)', 'sum w_ls R_MCAL']


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
    #return idx
    theta = np.rad2deg(np.arcsin(dist3d * np.sqrt(1 - dist3d**2 / 4)))
    mask = theta > rmin

    return idx[mask], theta[mask]


def precompute_photo_z_dilution_factor(
        table_l, table_c, cosmology=FlatLambdaCDM(H0=100, Om0=0.3),
        nz=None):
    """Calculate the photo-z bias for a single lens.

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
        The denominator and numerator of the photo-z bias factor, `f_bias`.
    """

    for table in [table_l, table_c]:
        if 'd_com' not in table.colnames:
            table['d_com'] = cosmology.comoving_transverse_distance(
                table['z']).to(u.Mpc).value

    if 'd_com_true' not in table_c.colnames:
        table_c['d_com_true'] = cosmology.comoving_transverse_distance(
            table_c['z_true']).to(u.Mpc).value

    if 'z_l_max' not in table_c.colnames:
        print("Warning: Could not find a lens-source separation cut." +
              " Thus, only z_l < z_s is required. Consider running " +
              "`add_maximum_lens_redshift` to define a lens-source " +
              "separation cut.")
        table_c['z_l_max'] = table_c['z']

    table_l['calib: sum w_ls w_c sigma_crit_p / sigma_crit_t'] = np.zeros(
        len(table_l))
    table_l['calib: sum w_ls w_c'] = np.zeros(len(table_l))

    for i, lens in enumerate(table_l):
        sigma_crit_phot = critical_surface_density(
            lens['z'], table_c['z'], d_l=lens['d_com'], d_s=table_c['d_com'])
        sigma_crit_true = critical_surface_density(
            lens['z'], table_c['z_true'], d_l=lens['d_com'],
            d_s=table_c['d_com_true'])
        mask = lens['z'] < table_c['z_l_max']
        table_l['calib: sum w_ls w_c sigma_crit_p / sigma_crit_t'][i] = np.sum(
            (table_c['w_sys'] * table_c['w'] / sigma_crit_phot /
             sigma_crit_true)[mask])
        table_l['calib: sum w_ls w_c'][i] = np.sum((
            table_c['w_sys'] * table_c['w'] / sigma_crit_phot**2)[mask])

    return table_l


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
        error :math:`\sigma_z`.
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


def precompute_chunk(table_l, table_s, rp_bins, table_c=None,
                     sigma_crit_eff_inv=None,
                     cosmology=FlatLambdaCDM(H0=100, Om0=0.3), comoving=True,
                     compress_jackknife_fields=False):
    """Do all the precomputation for all lens-source pairs. Compared to
    :func:`~dsigma.precompute.precompute_catalog`, this function calculates
    the separations between all sources and all lenses and assumes many
    pre-computed results. Most users will use
    :func:`~dsigma.precompute.precompute_catalog`, instead.

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
    sigma_crit_eff_inv : list, optional
        List of functions that return the inverse of the effective critical
        surface density as a function of the scale factor of the lens. Each
        function in the list corresponds to one source redshift bin.
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations.
    comoving : boolean, optional
        Whether to use comoving or physical quantities.
    compress_jackknife_fields : boolean, optional
        If set to true, use :func:`dsigma.jackknife.compress_jackknife_fields`
        to compress jackknife fields into a single row and save memory.
        However, doing so means that lenses inside each jackknife field can
        no longer be studied individually or in subsets.

    Returns
    -------
    table_l : astropy.table.Table
        Lens catalog with the pre-computation results attached in the table.
    """

    precompute_keys_chunk = precompute_keys.copy()

    if 'm' not in table_s.keys():
        precompute_keys_chunk.remove('sum w_ls m')
    if 'sigma_rms' not in table_s.keys():
        precompute_keys_chunk.remove('sum w_ls (1 - sigma_rms^2)')

    if 'R_2' not in table_s.keys():
        precompute_keys_chunk.remove('sum w_ls A p(R_2=0.3)')
    if np.all(np.isin(['R_11', 'R_22', 'R_12', 'R_21'],
                      list(table_s.keys()))):
        precompute_keys_chunk.remove('sum w_ls R_T')

    for key in precompute_keys_chunk:
        table_l[key] = np.zeros((len(table_l), len(rp_bins) - 1))

    for i, lens in enumerate(table_l):

        cos_theta = (
            (lens['sin dec'] * table_s['sin dec']) +
            (lens['cos dec'] * lens['sin ra'] *
             table_s['cos dec'] * table_s['sin ra']) +
            (lens['cos dec'] * lens['cos ra'] *
             table_s['cos dec'] * table_s['cos ra']))

        mpc_deg = mpc_per_degree(lens['z'], cosmology=cosmology,
                                 comoving=comoving)

        rp_binned = np.digitize(
            cos_theta, np.cos(np.deg2rad(rp_bins / mpc_deg))) - 1

        use = ((rp_binned >= 0) & (rp_binned < len(rp_bins) - 1) &
               (lens['z'] < table_s['z_l_max']))
        sel = np.arange(len(use))[use]
        rp_binned = rp_binned[sel]

        table_s_sub = {}
        for col in table_s.colnames:
            if col in ['sin ra', 'cos ra', 'sin dec', 'cos dec', 'z', 'd_com',
                       'w', 'e_1', 'e_2', 'sigma_rms', 'm', 'R_2', 'R_11',
                       'R_22', 'R_12', 'R_21', 'z_bin']:
                table_s_sub[col] = table_s[col].data[sel]

        # Calculate the critical surface density.
        if sigma_crit_eff_inv is None:
            sigma_crit = critical_surface_density(
                lens['z'], table_s_sub['z'], comoving=comoving,
                d_l=lens['d_com'], d_s=table_s_sub['d_com'])
        else:
            sigma_crit = np.zeros(len(table_s_sub['z_bin']))
            for z_bin in np.unique(table_s_sub['z_bin']):
                mask = table_s_sub['z_bin'] == z_bin
                sigma_crit[mask] = (
                    1.0 / sigma_crit_eff_inv[z_bin](1.0 / (1.0 + lens['z'])))

        # Calculate the projection angle for each lens-source pair.
        cos2phi, sin2phi = projection_angle_sin_cos(
            lens['sin ra'], lens['cos ra'], lens['sin dec'], lens['cos dec'],
            table_s_sub['sin ra'], table_s_sub['cos ra'],
            table_s_sub['sin dec'], table_s_sub['cos dec'])

        # Calculate the tangential and cross shear terms.
        e_t = - table_s_sub['e_1'] * cos2phi - table_s_sub['e_2'] * sin2phi
        e_x = + table_s_sub['e_1'] * sin2phi - table_s_sub['e_2'] * cos2phi

        # The weight of each lens-source pair is weighted by the critical
        # surface density.
        w_ls = np.where(
            sigma_crit != np.inf, table_s_sub['w'] / sigma_crit**2, 0)
        sigma_crit_w_ls = np.where(
            sigma_crit != np.inf, table_s_sub['w'] / sigma_crit, 0)

        sum_s = partial(np.bincount, rp_binned, minlength=len(rp_bins) - 1)

        table_l['sum w_s e_t'][i, :] = sum_s(weights=e_t * table_s_sub['w'])
        table_l['sum w_s e_x'][i, :] = sum_s(weights=e_x * table_s_sub['w'])
        table_l['sum w_s'][i, :] = sum_s(weights=table_s_sub['w'])

        table_l['sum w_ls e_t sigma_crit'][i, :] = sum_s(
            weights=e_t * sigma_crit_w_ls)
        table_l['sum w_ls e_x sigma_crit'][i, :] = sum_s(
            weights=e_x * sigma_crit_w_ls)
        table_l['sum w_ls'][i, :] = sum_s(weights=w_ls)

        # This is used to calculate the naive error for the lensing signal.
        table_l['sum (w_ls e_t sigma_crit)^2'][i, :] = sum_s(
            weights=(e_t * sigma_crit_w_ls)**2)
        table_l['sum (w_ls e_x sigma_crit)^2'][i, :] = sum_s(
            weights=(e_x * sigma_crit_w_ls)**2)

        # Multiplicative bias m.
        if 'm' in table_s_sub.keys():
            table_l['sum w_ls m'][i, :] = sum_s(
                weights=w_ls * table_s_sub['m'])

        # Responsivity R.
        if 'sigma_rms' in table_s_sub.keys():
            table_l['sum w_ls (1 - sigma_rms^2)'][i, :] = sum_s(
                weights=w_ls * (1 - table_s_sub['sigma_rms']**2))

        # Square term about the shape noise.
        table_l['sum w_s e_t^2'][i, :] = sum_s(
            weights=e_t**2 * table_s_sub['w'])
        table_l['sum w_s e_x^2'][i, :] = sum_s(
            weights=e_x**2 * table_s_sub['w'])

        # Number of pairs in each radial bin.
        table_l['sum 1'][i, :] = sum_s()

        # Resolution selection bias.
        if 'R_2' in table_s_sub.keys():
            table_l['sum w_ls A p(R_2=0.3)'][i, :] = (
                surveys.hsc.precompute_selection_bias_factor(
                    table_s_sub['R_2'], w_ls, rp_binned, len(rp_bins) - 1))

        # METACALIBRATION response.
        if np.all(np.isin(['R_11', 'R_22', 'R_12', 'R_21'],
                          list(table_s_sub.keys()))):
            r_t = (table_s_sub['R_11'] * cos2phi**2 +
                   table_s_sub['R_22'] * sin2phi**2 +
                   (table_s_sub['R_12'] + table_s_sub['R_21']) *
                   sin2phi * cos2phi)
            table_l['sum w_ls R_T'][i, :] = sum_s(weights=w_ls * r_t)

    # If necessary, estimate the photo-z bias factor.
    if table_c is not None:
        table_l = precompute_photo_z_dilution_factor(
            table_l, table_c, cosmology=cosmology)

    if compress_jackknife_fields:
        table_l = compress_jackknife_fields_function(table_l)

    return table_l


def precompute_catalog(table_l, table_s, rp_bins, table_c=None, nz=None,
                       cosmology=FlatLambdaCDM(H0=100, Om0=0.3),
                       comoving=True, trim=True,
                       compress_jackknife_fields=False, nside=64, n_jobs=1):
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
    nz : numpy array, optional
        Source redshift distributions. Must have shape (n, 2, m), where n is
        the number of source redshift bins and m the number of redshifts for
        which n(z) is tabulated. The first entry in the second dimension is
        assumed to be the redshift and the second entry in the second dimension
        is the n(z).
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations.
    comoving : boolean, optional
        Whether to use comoving or physical quantities.
    trim : boolean, optional
        If set to true, the output table will omit lenses that do not have any
        nearby sources.
    compress_jackknife_fields : boolean, optional
        If set to true, use :func:`dsigma.jackknife.compress_jackknife_fields`
        to compress jackknife fields into a single row and save memory.
        However, doing so means that lenses inside each jackknife field can
        no longer be studied individually or in subsets.
    nside : int, optional
        dsigma uses pixelization to group nearby lenses together and process
        them simultaneously. This parameter determine the number of pixels.
        See the documentation of healpix or healpy for reference. Has to be
        a power of 2.
    n_jobs : int, optional
        Number of jobs to run at the same time.

    Returns
    -------
    table_l : astropy.table.Table
        Lens catalog with the pre-computation results attached in the table.
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
    if not isinstance(nside, int) or not np.isin(nside, 2**np.arange(15)):
        raise Exception('nside must be a positive power of 2 but received ' +
                        '{}.'.format(nside))
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise Exception('Illegal number of jobs. Expected positive integer ' +
                        'but received {}.'.format(n_jobs))

    if 'd_com' not in table_l.colnames:
        table_l['d_com'] = cosmology.comoving_transverse_distance(
            table_l['z']).to(u.Mpc).value

    if 'd_com' not in table_s.colnames and nz is None:
        table_s['d_com'] = cosmology.comoving_transverse_distance(
            table_s['z']).to(u.Mpc).value

    if np.any(table_l['z'] < 0):
        raise Exception('Input lens redshifts must all be non-negative.')

    if nz is not None:
        if not (isinstance(nz, np.ndarray) and np.issubdtype(nz.dtype,
                                                             np.float)):
            raise Exception('nz must be a numpy array of floats.')
        if len(nz.shape) != 3 or nz.shape[1] != 2:
            raise Exception('nz must have shape (n, 2, m).')
        if 'z_bin' not in table_s.colnames:
            raise Exception('To use source redshift distributions, the ' +
                            'source table needs to have a `z_bin` column.')
        if not np.issubdtype(table_s['z_bin'].data.dtype, np.int) or np.amin(
                table_s['z_bin']) < 0:
            raise Exception('The `z_bin` column in the source table must ' +
                            'contain only non-negative integers.')
        if np.amax(table_s['z_bin']) > nz.shape[0]:
            raise Exception('The source table contains more redshift bins ' +
                            'than where passed via the nz argument.')

    if table_c is not None:
        if 'd_com_true' not in table_c.colnames:
            table_c['d_com_true'] = cosmology.comoving_transverse_distance(
                table_c['z_true']).to(u.Mpc).value

    if 'w_sys' not in table_l.colnames:
        print("Warning: Could not find systematic weights for lenses. " +
              "Weights can be specified in the `w` column of the lens" +
              "table. Weights are set to unity.")
        table_l['w_sys'] = 1

    if 'z_l_max' not in table_s.colnames:
        print("Warning: Could not find a lens-source separation cut." +
              " Thus, only z_l < z_s is required. Consider running " +
              "`add_maximum_lens_redshift` to define a lens-source " +
              "separation cut.")
        table_s['z_l_max'] = table_s['z']

    if trim:
        coord_s = SkyCoord(ra=table_s['ra'], dec=table_s['dec'], unit='deg')
        coord_l = SkyCoord(ra=table_l['ra'], dec=table_l['dec'], unit='deg')

        idx, d2d, d3d = coord_l.match_to_catalog_sky(coord_s)
        alpha_max = np.amax(rp_bins) / table_l['d_com'] * u.rad
        if not comoving:
            alpha_max *= (1 + table_l['z'])
        table_l = table_l[d2d < alpha_max]

    if nz is not None:
        sigma_crit_eff_inv = []
        a_l = np.linspace(1.0 / (1 + np.amax(table_l['z'])),
                          1.0 / (1 + np.amin(table_l['z'])), 1000)
        z_l = 1 / a_l - 1
        for i in range(nz.shape[0]):
            sigma_crit_eff = effective_critical_surface_density(
                z_l, nz[i, 0, :], nz[i, 1, :], cosmology=cosmology,
                comoving=comoving)
            sigma_crit_eff_inv.append(interp1d(
                a_l, 1.0 / sigma_crit_eff, kind='cubic', bounds_error=False,
                fill_value=(sigma_crit_eff[0]**-1, sigma_crit_eff[-1]**-1)))
    else:
        sigma_crit_eff_inv = None

    for f in [np.sin, np.cos]:
        for table in [table_s, table_l]:
            for angle in ['ra', 'dec']:
                col = f.__name__ + ' ' + angle
                if col not in table.colnames:
                    table[col] = f(np.deg2rad(table[angle]))

    pix_l = hp.ang2pix(nside, table_l['ra'].data, table_l['dec'].data,
                       lonlat=True)
    pix_group_l, n_group_l = np.unique(pix_l, return_counts=True)
    ra_group_l, dec_group_l = hp.pix2ang(nside, pix_group_l, lonlat=True)

    x, y, z = spherical_to_cartesian(table_s['ra'], table_s['dec'])
    kdtree_s = cKDTree(np.column_stack([x, y, z]), leafsize=1000)

    kwargs = {'table_c': table_c, 'sigma_crit_eff_inv': sigma_crit_eff_inv,
              'cosmology': cosmology, 'comoving': comoving,
              'compress_jackknife_fields': compress_jackknife_fields}

    pool = Pool(processes=n_jobs)
    result_list = []

    idx_l_sorted = np.arange(len(table_l))
    idx_l_unsorted = np.zeros(0, dtype=np.int)

    table_l = table_l[np.argsort(pix_l)]
    idx_l_sorted = idx_l_sorted[np.argsort(pix_l)]
    n_group_l = n_group_l[np.argsort(pix_group_l)]
    pix_group_l = np.sort(pix_group_l)

    for i, pix in enumerate(pix_group_l):

        # Calculate the result for all lenses in a healpix pixel and all
        # sources that could possibly be associated with any lens in the pixel.
        i_min, i_max = np.sum(n_group_l[:i]), np.sum(n_group_l[:i+1])
        table_l_pix = table_l[i_min:i_max]
        idx_l_unsorted = np.concatenate((
            idx_l_unsorted, idx_l_sorted[i_min:i_max]))

        alpha_max = np.amax(rp_bins) / np.amin(table_l_pix['d_com']) * u.rad
        if not comoving:
            alpha_max *= (
                1 + table_l_pix['z'][np.argmin(table_l_pix['d_com'])])
        alpha_max += hp.max_pixrad(nside, degrees=True) * u.deg

        r_max = np.sqrt(2 - 2 * np.cos(alpha_max.to(u.rad).value))
        x_l, y_l, z_l = spherical_to_cartesian(ra_group_l[i], dec_group_l[i])
        sel = np.fromiter(kdtree_s.query_ball_point([x_l, y_l, z_l], r_max),
                          dtype=np.int64)
        sel = np.sort(sel)

        table_s_sub = table_s[sel]

        result_list.append(pool.apply_async(
            precompute_chunk, (table_l_pix, table_s_sub, rp_bins), kwargs))

        while pool._taskqueue.qsize() >= n_jobs:
            time.sleep(0.1)

    pool.close()
    pool.join()

    # Convert the list of results into a table that is merged with the original
    # lens table and add useful meta-data.
    if not compress_jackknife_fields:
        table_l = vstack([result.get() for result in result_list])[
            np.argsort(idx_l_unsorted)]
    else:
        table_l = vstack([result.get() for result in result_list])
        table_l = compress_jackknife_fields_function(table_l)
    table_l.meta['rp_bins'] = rp_bins
    table_l.meta['comoving'] = comoving
    table_l.meta['H0'] = cosmology.H0.value
    table_l.meta['Ok0'] = cosmology.Ok0
    table_l.meta['Om0'] = cosmology.Om0

    return table_l


def merge_precompute_catalogs(table_l_list):
    """Merge precompute results for the same lenses and different sources.

    Parameters
    ----------
    table_l_list : list of astropy.table.Table
        List of precompute results.

    Returns
    -------
    table_l : astropy.table.Table
        Merged catalog of precompute results.
    """

    table_p = Table()
    table_p.meta = table_l_list[0].meta

    for i in range(len(table_l_list)):
        for key in ['rp_bins', 'comoving', 'H0', 'Ok0', 'Om0']:
            if not np.all(table_p.meta[key] == table_l_list[i].meta[key]):
                raise RuntimeError(
                    'Inconsistent meta-data for key {}.'.format(key))

    for i in range(len(table_l_list)):
        for key in table_l_list[0].colnames:
            if i == 0:
                table_p[key] = table_l_list[0][key]
            else:
                if key not in precompute_keys:
                    if np.any(table_p[key] != table_l_list[i][key]):
                        raise RuntimeError(
                            'Mismatch between tables for key {}.'.format(key))
                else:
                    table_p[key] += table_l_list[i][key]

    return table_p
