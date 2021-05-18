import warnings
import multiprocessing as mp
import queue as Queue

import numpy as np
import healpy as hp
from scipy.interpolate import interp1d

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from .physics import critical_surface_density
from .physics import effective_critical_surface_density
from .precompute_engine import precompute_engine


__all__ = ["add_maximum_lens_redshift", "photo_z_dilution_factor",
           "add_precompute_results"]


def photo_z_dilution_factor(z_l, table_c, cosmology):
    """Calculate the photo-z bias for a single lens.

    Parameters
    ----------
    z_l : float
        Redshift of the lens.
    table_c : astropy.table.Table
        Photometric redshift calibration catalog.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.

    Returns
    -------
        The photo-z bias factor, `f_bias`.
    """

    for key in ['', '_true']:
        if 'd_com' + key not in table_c.colnames:
            table_c['d_com' + key] = cosmology.comoving_transverse_distance(
                table_c['z' + key]).to(u.Mpc).value

    if 'z_l_max' not in table_c.colnames:
        warnings.warn('No lens-source cut given in calibration catalog. Will' +
                      ' use z_l < z_s.', RuntimeWarning)
        table_c['z_l_max'] = table_c['z']

    sigma_crit_phot = critical_surface_density(
        z_l, table_c['z'], d_s=table_c['d_com'], cosmology=cosmology)
    sigma_crit_true = critical_surface_density(
        z_l, table_c['z_true'], d_s=table_c['d_com_true'], cosmology=cosmology)
    mask = z_l < table_c['z_l_max']
    w = table_c['w_sys'] * table_c['w']

    if np.sum(mask) > 0:
        return (np.sum((w / sigma_crit_phot**2)[mask]) /
                np.sum((w / sigma_crit_phot / sigma_crit_true)[mask]))
    else:
        warnings.warn('Could not find valid calibration sources for some ' +
                      'lens redshifts. The f_bias correction may be ' +
                      'undefined.', RuntimeWarning)
        return np.nan


def add_maximum_lens_redshift(table_s, dz_min=0.0, z_err_factor=0,
                              apply_z_low=False):
    r"""For each source in the table, determine the maximum lens redshift
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


def get_raw_multiprocessing_array(array):
    """ To save memeory, use shared-memory multiprocessing arrays. This
    function converts an integer or float numpy array into such an array.

    Parameters
    ----------
    array : numpy array or None
        Input array.

    Returns
    -------
    array_mp : multiprocessing.RawArray or None
        Output array. None if input is None.

    """

    if array is None:
        return None

    array_mp = mp.RawArray('l' if np.issubdtype(array.dtype, np.integer) else
                           'd', len(array))
    array_np = np.ctypeslib.as_array(array_mp)
    array_np[:] = array

    return array_mp


def add_precompute_results(
        table_l, table_s, bins, table_c=None, table_n=None,
        cosmology=FlatLambdaCDM(H0=100, Om0=0.3), comoving=True,
        shear_mode=False, nside=256, n_jobs=1):
    """For all lenses in the catalog, perform the precomputation of lensing
    statistics.

    Parameters
    ----------
    table_l : astropy.table.Table
        Catalog of lenses.
    table_s : astropy.table.Table
        Catalog of sources.
    bins : numpy array
        Bins in radius to use for the stacking. By default, these are in
        projected distance and Mpc. However, if `shear_mode` is set to True,
        these will be assumed to be angular separations in degrees.
    table_c : astropy.table.Table, optional
        Additional photometric redshift calibration catalog. Only relevant if
        `shear_mode` is not active.
    table_n : astropy.table.Table, optional
        Source redshift distributions. Only relevant if `shear_mode` is not
        active.
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations. Only relevant if `shear_mode` is
        not active.
    comoving : boolean, optional
        Whether to use comoving or physical quantities. Only relevant if
        `shear_mode` is not active.
    shear_mode : boolean, optional
        If true, bins are assumed to be in degrees. Also, the individual
        weights of lens-source pairs are only determined by the source weight,
        not also by the critical surface density. Finally, all lens-source
        pairs will be analzyed unless a lens-source cut was previously
        specified. This mode is useful for calculating tangential shear.
    nside : int, optional
        dsigma uses pixelization to group nearby lenses together and process
        them simultaneously. This parameter determines the number of pixels.
        It has to be a power of 2. This number likely impacts performance.
    n_jobs : int, optional
        Number of jobs to run at the same time.

    Returns
    -------
    table_l : astropy.table.Table
        Lens catalog with the pre-computation results attached to the table.
    """

    try:
        assert cosmology.Ok0 == 0
    except AssertionError:
        raise Exception('Currently, dsigma does not support non-flat ' +
                        'cosmologies.')

    if np.any(table_l['z'] < 0):
        raise Exception('Input lens redshifts must all be non-negative.')
    if not isinstance(nside, int) or not np.isin(nside, 2**np.arange(15)):
        raise Exception('nside must be a positive power of 2 but received ' +
                        '{}.'.format(nside))
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise Exception('Illegal number of jobs. Expected positive integer ' +
                        'but received {}.'.format(n_jobs))

    if table_n is not None:
        if 'z_bin' not in table_s.colnames:
            raise Exception('To use source redshift distributions, the ' +
                            'source table needs to have a `z_bin` column.')
        if not np.issubdtype(table_s['z_bin'].data.dtype, int) or np.amin(
                table_s['z_bin']) < 0:
            raise Exception('The `z_bin` column in the source table must ' +
                            'contain only non-negative integers.')
        if np.amax(table_s['z_bin']) > table_n['n'].data.shape[1]:
            raise Exception('The source table contains more redshift bins ' +
                            'than where passed via the nz argument.')

    npix = hp.nside2npix(nside)
    pix_l = hp.ang2pix(nside, table_l['ra'], table_l['dec'], lonlat=True)
    pix_s = hp.ang2pix(nside, table_s['ra'], table_s['dec'], lonlat=True)
    argsort_pix_l = np.argsort(pix_l)
    argsort_pix_s = np.argsort(pix_s)
    pix_l_counts = np.ascontiguousarray(np.bincount(pix_l, minlength=npix))
    pix_s_counts = np.ascontiguousarray(np.bincount(pix_s, minlength=npix))
    pix_l_cum_counts = np.ascontiguousarray(np.cumsum(pix_l_counts))
    pix_s_cum_counts = np.ascontiguousarray(np.cumsum(pix_s_counts))

    z_l = np.ascontiguousarray(table_l['z'][argsort_pix_l], dtype=np.float64)
    z_s = np.ascontiguousarray(table_s['z'][argsort_pix_s], dtype=np.float64)

    # Parrallelize comoving distance calculation because it's slow.
    with mp.Pool(n_jobs) as pool:
        d_com_l = np.ascontiguousarray((np.concatenate(pool.map(
            cosmology.comoving_transverse_distance,
            np.array_split(table_l['z'], n_jobs))).to(u.Mpc).value)[
                argsort_pix_l])
        d_com_s = np.ascontiguousarray(np.concatenate(pool.map(
            cosmology.comoving_transverse_distance,
            np.array_split(table_s['z'], n_jobs))).to(u.Mpc).value[
                argsort_pix_s])

    sin_ra_l = np.ascontiguousarray(
        np.sin(np.deg2rad(table_l['ra']))[argsort_pix_l])
    cos_ra_l = np.ascontiguousarray(
        np.cos(np.deg2rad(table_l['ra']))[argsort_pix_l])
    sin_dec_l = np.ascontiguousarray(
        np.sin(np.deg2rad(table_l['dec']))[argsort_pix_l])
    cos_dec_l = np.ascontiguousarray(
        np.cos(np.deg2rad(table_l['dec']))[argsort_pix_l])
    sin_ra_s = np.ascontiguousarray(
        np.sin(np.deg2rad(table_s['ra']))[argsort_pix_s])
    cos_ra_s = np.ascontiguousarray(
        np.cos(np.deg2rad(table_s['ra']))[argsort_pix_s])
    sin_dec_s = np.ascontiguousarray(
        np.sin(np.deg2rad(table_s['dec']))[argsort_pix_s])
    cos_dec_s = np.ascontiguousarray(
        np.cos(np.deg2rad(table_s['dec']))[argsort_pix_s])
    w_s = np.ascontiguousarray(table_s['w'][argsort_pix_s], dtype=np.float64)
    e_1 = np.ascontiguousarray(table_s['e_1'][argsort_pix_s], dtype=np.float64)
    e_2 = np.ascontiguousarray(table_s['e_2'][argsort_pix_s], dtype=np.float64)

    if 'z_l_max' not in table_s.colnames:
        if not shear_mode:
            warnings.warn('No lens-source cut given in source catalog. Will' +
                          ' use z_l < z_s.', RuntimeWarning)
            z_l_max = np.ascontiguousarray(table_s['z'][argsort_pix_s],
                                           dtype=np.float64)
        if shear_mode:
            z_l_max = np.ascontiguousarray(
                np.repeat(np.amax(z_l) * 10, len(table_s)), dtype=np.float64)
    else:
        z_l_max = np.ascontiguousarray(table_s['z_l_max'][argsort_pix_s],
                                       dtype=np.float64)

    if table_c is not None and table_n is None:
        z_min = np.amin(table_l['z'])
        z_max = np.amax(table_l['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.001)))
        f_bias_interp = np.array(
            [photo_z_dilution_factor(z, table_c, cosmology) for z in z_interp])
        f_bias_interp = interp1d(
            z_interp, f_bias_interp, kind='cubic', bounds_error=False,
            fill_value=(f_bias_interp[0], f_bias_interp[-1]))
        f_bias = np.ascontiguousarray(f_bias_interp(z_l))
        sigma_crit_eff = None
        z_bin = None
    elif table_c is None and table_n is not None:
        z_min = np.amin(table_l['z'])
        z_max = np.amax(table_l['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.001)))
        n_bins = table_n['n'].data.shape[1]
        sigma_crit_eff = np.ascontiguousarray(np.zeros(len(table_l) * n_bins,
                                                       dtype=np.float64))
        for i in range(n_bins):
            sigma_crit_eff_inv_interp = effective_critical_surface_density(
                z_interp, table_n['z'], table_n['n'][:, i],
                cosmology=cosmology, comoving=comoving)**-1
            sigma_crit_eff_inv_interp = interp1d(
                z_interp, sigma_crit_eff_inv_interp, kind='cubic',
                bounds_error=False,
                fill_value=(sigma_crit_eff_inv_interp[0],
                            sigma_crit_eff_inv_interp[-1]))
            sigma_crit_eff_inv_interp = sigma_crit_eff_inv_interp(z_l)
            sigma_crit_eff_interp = np.repeat(np.inf, len(table_l))
            mask = sigma_crit_eff_inv_interp == 0
            sigma_crit_eff_interp[~mask] = sigma_crit_eff_inv_interp[~mask]**-1
            sigma_crit_eff[i::n_bins] = sigma_crit_eff_interp
        z_bin = np.ascontiguousarray(table_s['z_bin'][argsort_pix_s],
                                     dtype=int)
        f_bias = None
    elif table_c is not None and table_s is not None:
        raise Exception('table_c and table_n cannot both be given.')
    else:
        f_bias = None
        sigma_crit_eff = None
        z_bin = None

    if 'm' in table_s.colnames:
        m = np.ascontiguousarray(table_s['m'][argsort_pix_s], dtype=np.float64)
    else:
        m = None

    if 'e_rms' in table_s.colnames:
        e_rms = np.ascontiguousarray(table_s['e_rms'][argsort_pix_s],
                                     dtype=np.float64)
    else:
        e_rms = None

    if 'R_2' in table_s.colnames:
        R_2 = np.ascontiguousarray(table_s['R_2'][argsort_pix_s],
                                   dtype=np.float64)
    else:
        R_2 = None

    if (('R_11' in table_s.colnames) and ('R_12' in table_s.colnames) and
            ('R_21' in table_s.colnames) and ('R_22' in table_s.colnames)):
        R_11 = np.ascontiguousarray(table_s['R_11'][argsort_pix_s],
                                    dtype=np.float64)
        R_12 = np.ascontiguousarray(table_s['R_12'][argsort_pix_s],
                                    dtype=np.float64)
        R_21 = np.ascontiguousarray(table_s['R_21'][argsort_pix_s],
                                    dtype=np.float64)
        R_22 = np.ascontiguousarray(table_s['R_22'][argsort_pix_s],
                                    dtype=np.float64)
    else:
        R_11, R_12, R_21, R_22 = None, None, None, None

    if not shear_mode:
        theta = (np.tile(bins, len(table_l)) /
                 np.repeat(d_com_l, len(bins))).flatten()
    else:
        theta = np.tile(np.deg2rad(bins), len(table_l))

    if not shear_mode and not comoving:
        theta *= (1 + np.repeat(z_l, len(bins))).flatten()

    dist_3d_sq_bins = np.minimum(4 * np.sin(theta / 2.0)**2, 2.0)

    # Create arrays that will hold the final results.
    n_results = len(table_l) * (len(bins) - 1)
    sum_1 = np.ascontiguousarray(np.zeros(n_results, dtype=np.int64))
    sum_w_ls = np.ascontiguousarray(np.zeros(n_results, dtype=np.float64))
    sum_w_ls_e_t = np.ascontiguousarray(np.zeros(n_results, dtype=np.float64))

    if not shear_mode:
        sum_w_ls_e_t_sigma_crit = np.ascontiguousarray(np.zeros(
            n_results, dtype=np.float64))
        if table_c is not None:
            sum_w_ls_e_t_sigma_crit_f_bias = np.ascontiguousarray(np.zeros(
                n_results, dtype=np.float64))
        else:
            sum_w_ls_e_t_sigma_crit_f_bias = None
        sum_w_ls_e_t_sigma_crit_sq = np.ascontiguousarray(np.zeros(
            n_results, dtype=np.float64))
    else:
        sum_w_ls_e_t_sigma_crit = None
        sum_w_ls_e_t_sigma_crit_f_bias = None
        sum_w_ls_e_t_sigma_crit_sq = None

    sum_w_ls_z_s = np.ascontiguousarray(np.zeros(n_results, dtype=np.float64))

    if 'm' in table_s.colnames:
        sum_w_ls_m = np.ascontiguousarray(
            np.zeros(n_results, dtype=np.float64))
    else:
        sum_w_ls_m = None

    if 'e_rms' in table_s.colnames:
        sum_w_ls_1_minus_e_rms_sq = np.ascontiguousarray(
            np.zeros(n_results, dtype=np.float64))
    else:
        sum_w_ls_1_minus_e_rms_sq = None

    if 'R_2' in table_s.colnames:
        sum_w_ls_A_p_R_2 = np.ascontiguousarray(
            np.zeros(n_results, dtype=np.float64))
    else:
        sum_w_ls_A_p_R_2 = None

    if (('R_11' in table_s.colnames) and ('R_12' in table_s.colnames) and
            ('R_21' in table_s.colnames) and ('R_22' in table_s.colnames)):
        sum_w_ls_R_T = np.ascontiguousarray(
            np.zeros(n_results, dtype=np.float64))
    else:
        sum_w_ls_R_T = None

    # When running in parrallel, replace numpy arrays with shared-memory
    # multiprocessing arrays.
    if n_jobs > 1:
        pix_l_counts = get_raw_multiprocessing_array(pix_l_counts)
        pix_s_counts = get_raw_multiprocessing_array(pix_s_counts)
        pix_l_cum_counts = get_raw_multiprocessing_array(pix_l_cum_counts)
        pix_s_cum_counts = get_raw_multiprocessing_array(pix_s_cum_counts)
        z_l = get_raw_multiprocessing_array(z_l)
        z_s = get_raw_multiprocessing_array(z_s)
        d_com_l = get_raw_multiprocessing_array(d_com_l)
        d_com_s = get_raw_multiprocessing_array(d_com_s)
        sin_ra_l = get_raw_multiprocessing_array(sin_ra_l)
        cos_ra_l = get_raw_multiprocessing_array(cos_ra_l)
        sin_dec_l = get_raw_multiprocessing_array(sin_dec_l)
        cos_dec_l = get_raw_multiprocessing_array(cos_dec_l)
        sin_ra_s = get_raw_multiprocessing_array(sin_ra_s)
        cos_ra_s = get_raw_multiprocessing_array(cos_ra_s)
        sin_dec_s = get_raw_multiprocessing_array(sin_dec_s)
        cos_dec_s = get_raw_multiprocessing_array(cos_dec_s)
        w_s = get_raw_multiprocessing_array(w_s)
        e_1 = get_raw_multiprocessing_array(e_1)
        e_2 = get_raw_multiprocessing_array(e_2)
        z_l_max = get_raw_multiprocessing_array(z_l_max)
        f_bias = get_raw_multiprocessing_array(f_bias)
        sigma_crit_eff = get_raw_multiprocessing_array(sigma_crit_eff)
        z_bin = get_raw_multiprocessing_array(z_bin)
        m = get_raw_multiprocessing_array(m)
        e_rms = get_raw_multiprocessing_array(e_rms)
        R_2 = get_raw_multiprocessing_array(R_2)
        R_11 = get_raw_multiprocessing_array(R_11)
        R_12 = get_raw_multiprocessing_array(R_12)
        R_21 = get_raw_multiprocessing_array(R_21)
        R_22 = get_raw_multiprocessing_array(R_22)
        dist_3d_sq_bins = get_raw_multiprocessing_array(dist_3d_sq_bins)
        sum_1 = get_raw_multiprocessing_array(sum_1)
        sum_w_ls = get_raw_multiprocessing_array(sum_w_ls)
        sum_w_ls_e_t = get_raw_multiprocessing_array(sum_w_ls_e_t)
        sum_w_ls_e_t_sigma_crit = get_raw_multiprocessing_array(
            sum_w_ls_e_t_sigma_crit)
        sum_w_ls_e_t_sigma_crit_f_bias = get_raw_multiprocessing_array(
            sum_w_ls_e_t_sigma_crit_f_bias)
        sum_w_ls_e_t_sigma_crit_sq = get_raw_multiprocessing_array(
            sum_w_ls_e_t_sigma_crit_sq)
        sum_w_ls_z_s = get_raw_multiprocessing_array(
            sum_w_ls_z_s)
        sum_w_ls_m = get_raw_multiprocessing_array(
            sum_w_ls_m)
        sum_w_ls_1_minus_e_rms_sq = get_raw_multiprocessing_array(
            sum_w_ls_1_minus_e_rms_sq)
        sum_w_ls_A_p_R_2 = get_raw_multiprocessing_array(sum_w_ls_A_p_R_2)
        sum_w_ls_R_T = get_raw_multiprocessing_array(sum_w_ls_R_T)

    # Create a queue that holds all the pixels containing lenses.
    if n_jobs == 1:
        queue = Queue.Queue()
    else:
        queue = mp.Queue()

    for pix in np.unique(pix_l):
        queue.put(pix)

    args = (pix_l_counts, pix_s_counts, pix_l_cum_counts, pix_s_cum_counts,
            z_l, z_s, d_com_l, d_com_s, sin_ra_l, cos_ra_l, sin_dec_l,
            cos_dec_l, sin_ra_s, cos_ra_s, sin_dec_s, cos_dec_s, w_s, e_1, e_2,
            z_l_max, f_bias, z_bin, sigma_crit_eff, m, e_rms, R_2, R_11, R_12,
            R_21, R_22, dist_3d_sq_bins, sum_1, sum_w_ls, sum_w_ls_e_t,
            sum_w_ls_e_t_sigma_crit, sum_w_ls_e_t_sigma_crit_f_bias,
            sum_w_ls_e_t_sigma_crit_sq, sum_w_ls_z_s, sum_w_ls_m,
            sum_w_ls_1_minus_e_rms_sq, sum_w_ls_A_p_R_2, sum_w_ls_R_T)

    if n_jobs == 1:
        precompute_engine(*args, bins, comoving, shear_mode, nside, queue)
    else:
        processes = []
        for i in range(n_jobs):
            process = mp.Process(
                target=precompute_engine,
                args=(*args, bins, comoving, shear_mode, nside, queue))
            process.start()
            processes.append(process)
        for i in range(n_jobs):
            processes[i].join()

    inv_argsort_pix_l = np.argsort(argsort_pix_l)
    table_l['sum 1'] = np.array(sum_1).reshape(
        len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    table_l['sum w_ls'] = np.array(sum_w_ls).reshape(
        len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    table_l['sum w_ls e_t'] = np.array(sum_w_ls_e_t).reshape(
        len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    if sum_w_ls_e_t_sigma_crit is not None:
        table_l['sum w_ls e_t sigma_crit'] = np.array(
            sum_w_ls_e_t_sigma_crit).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    if sum_w_ls_e_t_sigma_crit_f_bias is not None:
        table_l['sum w_ls e_t sigma_crit f_bias'] = np.array(
            sum_w_ls_e_t_sigma_crit_f_bias).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    if sum_w_ls_e_t_sigma_crit_sq is not None:
        table_l['sum (w_ls e_t sigma_crit)^2'] = np.array(
            sum_w_ls_e_t_sigma_crit_sq).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    table_l['sum w_ls z_l'] = table_l['z'][:, np.newaxis] * table_l['sum w_ls']
    table_l['sum w_ls z_s'] = np.array(sum_w_ls_z_s).reshape(
        len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    if sum_w_ls_m is not None:
        table_l['sum w_ls m'] = np.array(sum_w_ls_m).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    if sum_w_ls_1_minus_e_rms_sq is not None:
        table_l['sum w_ls (1 - e_rms^2)'] = np.array(
            sum_w_ls_1_minus_e_rms_sq).reshape(
                len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    if sum_w_ls_A_p_R_2 is not None:
        table_l['sum w_ls A p(R_2=0.3)'] = np.array(
            sum_w_ls_A_p_R_2).reshape(
                len(table_l), len(bins) - 1)[inv_argsort_pix_l]
    if sum_w_ls_R_T is not None:
        table_l['sum w_ls R_T'] = np.array(sum_w_ls_R_T).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]

    table_l.meta['bins'] = bins
    table_l.meta['comoving'] = comoving
    table_l.meta['H0'] = cosmology.H0.value
    table_l.meta['Ok0'] = cosmology.Ok0
    table_l.meta['Om0'] = cosmology.Om0

    return table_l
