import warnings
import multiprocessing as mp
import queue as Queue

import numpy as np
from astropy_healpix import HEALPix
from scipy.interpolate import interp1d

from tqdm import tqdm

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from .physics import critical_surface_density
from .physics import effective_critical_surface_density
from .precompute_engine import precompute_engine


__all__ = ["photo_z_dilution_factor", "mean_photo_z_offset",
           "add_precompute_results", "add_maximum_lens_redshift"]


def parallel_compute(f, x, n_jobs):
    with mp.Pool(n_jobs) as pool:
        y = np.concatenate(pool.map(f, np.array_split(x, n_jobs)))
    return y


def photo_z_dilution_factor(z_l, table_c, cosmology):
    """Calculate the photo-z bias as a function of the lens redshift.

    Parameters
    ----------
    z_l : float or numpy array
        Redshift(s) of the lens.
    table_c : astropy.table.Table
        Photometric redshift calibration catalog.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.

    Returns
    -------
        The photo-z bias factor, `f_bias`, for the lens redshift(s).
    """

    if 'z_l_max' not in table_c.colnames:
        warnings.warn('No lens-source cut given in calibration catalog. Will' +
                      ' use z_l < z_s.', RuntimeWarning)
        table_c['z_l_max'] = table_c['z']

    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(table_c['z']).to(u.Mpc).value
    d_s_true = cosmology.comoving_transverse_distance(
        table_c['z_true']).to(u.Mpc).value
    z_l_max = table_c['z_l_max']
    w = table_c['w_sys'] * table_c['w']

    if hasattr(z_l, '__len__'):
        shape = (len(z_l), len(table_c))
        z_s = np.tile(z_s, len(z_l)).reshape(shape)
        z_s_true = np.tile(z_s_true, len(z_l)).reshape(shape)
        d_s = np.tile(d_s, len(z_l)).reshape(shape)
        d_s_true = np.tile(d_s_true, len(z_l)).reshape(shape)
        z_l_max = np.tile(z_l_max, len(z_l)).reshape(shape)
        w = np.tile(w, len(z_l)).reshape(shape)
        z_l = np.repeat(z_l, len(table_c)).reshape(shape)
        d_l = np.repeat(d_l, len(table_c)).reshape(shape)

    sigma_crit_phot = critical_surface_density(z_l, z_s, d_l=d_l, d_s=d_s)
    sigma_crit_true = critical_surface_density(z_l, z_s_true, d_l=d_l,
                                               d_s=d_s_true)
    mask = z_l_max < z_l

    if np.any(np.all(mask, axis=-1)):
        warnings.warn('Could not find valid calibration sources for some ' +
                      'lens redshifts. The f_bias correction may be ' +
                      'undefined.', RuntimeWarning)

    return (np.sum((w / sigma_crit_phot**2) * (~mask), axis=-1) /
            np.sum((w / sigma_crit_phot / sigma_crit_true) * (~mask),
                   axis=-1))


def mean_photo_z_offset(z_l, table_c, cosmology):
    """Calculate the mean offset of source photometric redshifts compared
    to true redshifts.

    Parameters
    ----------
    z_l : float or numpy array
        Redshift(s) of the lens.
    table_c : astropy.table.Table, optional
        Photometric redshift calibration catalog.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.

    Returns
    -------
        The mean source redshift offset for the lens redshift(s).
    """

    if 'z_l_max' not in table_c.colnames:
        warnings.warn('No lens-source cut given in calibration catalog. ' +
                      'Will use z_l < z_s.', RuntimeWarning)
        table_c['z_l_max'] = table_c['z']

    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(table_c['z']).to(
        u.Mpc).value
    z_l_max = table_c['z_l_max']
    w = table_c['w_sys'] * table_c['w']

    if not np.isscalar(z_l):
        shape = (len(z_l), len(table_c))
        z_s = np.tile(z_s, len(z_l)).reshape(shape)
        z_s_true = np.tile(z_s_true, len(z_l)).reshape(shape)
        d_s = np.tile(d_s, len(z_l)).reshape(shape)
        z_l_max = np.tile(z_l_max, len(z_l)).reshape(shape)
        w = np.tile(w, len(z_l)).reshape(shape)
        z_l = np.repeat(z_l, len(table_c)).reshape(shape)
        d_l = np.repeat(d_l, len(table_c)).reshape(shape)

    sigma_crit = critical_surface_density(z_l, z_s, d_l=d_l, d_s=d_s)
    w = w / sigma_crit**2

    mask = z_l_max < z_l

    return np.sum((z_s - z_s_true) * w * (~mask), axis=-1) / np.sum(
        w * (~mask), axis=-1)


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
        shear_mode=False, nside=256, n_jobs=1, progress_bar=False):
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
        Additional photometric redshift calibration catalog. Only used if
        `shear_mode` is not active.
    table_n : astropy.table.Table, optional
        Source redshift distributions. Only used if `shear_mode` is not active.
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
    progress_bar : boolean, option
        Whether to show a progress bar for the main loop over lens pixels.
        Default is false.

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

    hp = HEALPix(nside, order='ring')
    npix = hp.npix
    pix_l = hp.lonlat_to_healpix(table_l['ra'] * u.deg, table_l['dec'] * u.deg)
    pix_s = hp.lonlat_to_healpix(table_s['ra'] * u.deg, table_s['dec'] * u.deg)
    argsort_pix_l = np.argsort(pix_l)
    argsort_pix_s = np.argsort(pix_s)
    pix_l_counts = np.ascontiguousarray(np.bincount(pix_l, minlength=npix))
    pix_s_counts = np.ascontiguousarray(np.bincount(pix_s, minlength=npix))
    pix_l_cum_counts = np.ascontiguousarray(np.cumsum(pix_l_counts))
    pix_s_cum_counts = np.ascontiguousarray(np.cumsum(pix_s_counts))

    table_engine_l = {}
    table_engine_s = {}

    table_engine_l['z'] = np.ascontiguousarray(
        table_l['z'][argsort_pix_l], dtype=np.float64)
    table_engine_s['z'] = np.ascontiguousarray(
        table_s['z'][argsort_pix_s], dtype=np.float64)

    for f, f_name in zip([np.sin, np.cos], ['sin', 'cos']):
        for table, argsort_pix, table_engine in zip(
                [table_l, table_s], [argsort_pix_l, argsort_pix_s],
                [table_engine_l, table_engine_s]):
            for angle in ['ra', 'dec']:
                table_engine['{} {}'.format(f_name, angle)] =\
                    np.ascontiguousarray(f(np.deg2rad(table[angle]))[
                        argsort_pix])

    for key in ['w', 'e_1', 'e_2', 'm', 'e_rms', 'R_2', 'R_11', 'R_22',
                'R_12', 'R_21']:
        if key in table_s.colnames:
            table_engine_s[key] = np.ascontiguousarray(
                table_s[key][argsort_pix_s], dtype=np.float64)

    if 'z_l_max' not in table_s.colnames:
        if not shear_mode:
            warnings.warn('No lens-source cut given in source catalog. Will' +
                          ' use z_l < z_s.', RuntimeWarning)
            z_l_max = table_s['z']
        if shear_mode:
            z_l_max = np.repeat(1e4, len(table_s))
    else:
        z_l_max = table_s['z_l_max']

    table_engine_s['z_l_max'] = np.ascontiguousarray(
        z_l_max[argsort_pix_s], dtype=np.float64)

    for table, argsort_pix, table_engine in zip(
            [table_l, table_s], [argsort_pix_l, argsort_pix_s],
            [table_engine_l, table_engine_s]):
        table_engine['d_com'] = np.ascontiguousarray(parallel_compute(
            cosmology.comoving_transverse_distance, table['z'], n_jobs).to(
                u.Mpc).value[argsort_pix], dtype=np.float64)

    if table_c is not None and table_n is None:
        z_min = np.amin(table_l['z'])
        z_max = np.amax(table_l['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.001)))
        f_bias_interp = photo_z_dilution_factor(z_interp, table_c, cosmology)
        f_bias_interp = interp1d(
            z_interp, f_bias_interp, kind='cubic', bounds_error=False,
            fill_value=(f_bias_interp[0], f_bias_interp[-1]))
        table_engine_l['f_bias'] = np.ascontiguousarray(
            f_bias_interp(np.array(table_engine_l['z'])), dtype=np.float64)
        dz_s_interp = mean_photo_z_offset(
            z_interp, table_c=table_c, cosmology=cosmology)
        dz_s_interp = interp1d(
            z_interp, dz_s_interp, kind='cubic', bounds_error=False,
            fill_value=(dz_s_interp[0], dz_s_interp[-1]))
        table_engine_l['delta z_s'] = np.ascontiguousarray(
            dz_s_interp(np.array(table_engine_l['z'])), dtype=np.float64)

    elif table_c is None and table_n is not None:
        z_min = np.amin(table_l['z'])
        z_max = np.amax(table_l['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.001)))
        n_bins = table_n['n'].data.shape[1]
        sigma_crit_eff = np.zeros(len(table_l) * n_bins, dtype=np.float64)
        for i in range(n_bins):
            sigma_crit_eff_inv_interp = effective_critical_surface_density(
                z_interp, table_n['z'], table_n['n'][:, i],
                cosmology=cosmology, comoving=comoving)**-1
            sigma_crit_eff_inv_interp = interp1d(
                z_interp, sigma_crit_eff_inv_interp, kind='cubic',
                bounds_error=False,
                fill_value=(sigma_crit_eff_inv_interp[0],
                            sigma_crit_eff_inv_interp[-1]))
            sigma_crit_eff_inv_interp = sigma_crit_eff_inv_interp(
                np.array(table_engine_l['z']))
            sigma_crit_eff_interp = np.repeat(np.inf, len(table_l))
            mask = sigma_crit_eff_inv_interp == 0
            sigma_crit_eff_interp[~mask] = sigma_crit_eff_inv_interp[~mask]**-1
            sigma_crit_eff[i::n_bins] = sigma_crit_eff_interp
        table_engine_l['sigma_crit_eff'] = np.ascontiguousarray(
            sigma_crit_eff, dtype=np.float64)
        table_engine_s['z_bin'] = np.ascontiguousarray(
            table_s['z_bin'][argsort_pix_s], dtype=int)
    elif table_c is not None and table_s is not None:
        raise Exception('table_c and table_n cannot both be given.')

    # Create arrays that will hold the final results.
    table_engine_r = {}
    n_results = len(table_l) * (len(bins) - 1)

    key_list = ['sum 1', 'sum w_ls', 'sum w_ls e_t', 'sum w_ls z_s']

    if not shear_mode:
        key_list.append('sum w_ls e_t sigma_crit')
        key_list.append('sum (w_ls e_t sigma_crit)^2')

    if 'm' in table_s.colnames:
        key_list.append('sum w_ls m')

    if 'e_rms' in table_s.colnames:
        key_list.append('sum w_ls (1 - e_rms^2)')

    if 'R_2' in table_s.colnames:
        key_list.append('sum w_ls A p(R_2=0.3)')

    if (('R_11' in table_s.colnames) and ('R_12' in table_s.colnames) and
            ('R_21' in table_s.colnames) and ('R_22' in table_s.colnames)):
        key_list.append('sum w_ls R_T')

    for key in key_list:
        table_engine_r[key] = np.ascontiguousarray(
            np.zeros(n_results, dtype=(
                np.int64 if key == 'sum 1' else np.float64)))

    z_l = np.array(table_engine_l['z'])
    d_com_l = np.array(table_engine_l['d_com'])

    if not shear_mode:
        theta = (np.tile(bins, len(table_l)) /
                 np.repeat(d_com_l, len(bins))).flatten()
    else:
        theta = np.tile(np.deg2rad(bins), len(table_l))

    if not shear_mode and not comoving:
        theta *= (1 + np.repeat(z_l, len(bins))).flatten()

    dist_3d_sq_bins = np.minimum(4 * np.sin(theta / 2.0)**2, 2.0)

    if progress_bar:
        pbar = tqdm(total=np.sum(pix_l_counts > 0))
    else:
        pbar = None

    # When running in parrallel, replace numpy arrays with shared-memory
    # multiprocessing arrays.
    if n_jobs > 1:
        dist_3d_sq_bins = get_raw_multiprocessing_array(dist_3d_sq_bins)
        pix_l_counts = get_raw_multiprocessing_array(pix_l_counts)
        pix_s_counts = get_raw_multiprocessing_array(pix_s_counts)
        pix_l_cum_counts = get_raw_multiprocessing_array(pix_l_cum_counts)
        pix_s_cum_counts = get_raw_multiprocessing_array(pix_s_cum_counts)
        for table_engine in [table_engine_l, table_engine_s, table_engine_r]:
            for key in table_engine.keys():
                table_engine[key] = get_raw_multiprocessing_array(
                    table_engine[key])

    # Create a queue that holds all the pixels containing lenses.
    if n_jobs == 1:
        queue = Queue.Queue()
    else:
        queue = mp.Queue()

    for pix in np.unique(pix_l):
        queue.put(pix)

    args = (pix_l_counts, pix_s_counts, pix_l_cum_counts, pix_s_cum_counts,
            dist_3d_sq_bins, table_engine_l, table_engine_s, table_engine_r,
            bins, comoving, shear_mode, nside, queue, pbar)

    if n_jobs == 1:
        precompute_engine(*args)
    else:
        processes = []
        for i in range(n_jobs):
            process = mp.Process(target=precompute_engine, args=(*args, ))
            process.start()
            processes.append(process)
        for i in range(n_jobs):
            processes[i].join()

    if progress_bar:
        pbar.close()

    inv_argsort_pix_l = np.argsort(argsort_pix_l)
    for key in table_engine_r.keys():
        table_l[key] = np.array(table_engine_r[key]).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]

    table_l['sum w_ls z_l'] = table_l['z'][:, np.newaxis] * table_l['sum w_ls']

    if 'f_bias' in table_engine_l.keys():
        table_l['sum w_ls e_t sigma_crit f_bias'] = (
            np.array(table_engine_l['f_bias'])[inv_argsort_pix_l][
                :, np.newaxis] * table_l['sum w_ls e_t sigma_crit'])

    if 'delta z_s' in table_engine_l.keys():
        table_l['sum w_ls (z_s - delta z_s)'] = (
            table_l['sum w_ls z_s'] - table_l['sum w_ls'] * np.array(
                table_engine_l['delta z_s'])[inv_argsort_pix_l][:, np.newaxis])

    table_l.meta['bins'] = bins
    table_l.meta['comoving'] = comoving
    table_l.meta['H0'] = cosmology.H0.value
    table_l.meta['Ok0'] = cosmology.Ok0
    table_l.meta['Om0'] = cosmology.Om0
    table_l.meta['shear_mode'] = shear_mode

    return table_l
