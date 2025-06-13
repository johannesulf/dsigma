"""Module for pre-computing lensing results."""

import multiprocessing as mp
import numbers
import numpy as np
import queue as Queue
import warnings


from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.units import UnitConversionError
from astropy_healpix import HEALPix
from scipy.interpolate import interp1d

from .physics import critical_surface_density
from .physics import effective_critical_surface_density
from .precompute_engine import precompute_engine


__all__ = ["photo_z_dilution_factor", "mean_photo_z_offset", "precompute"]


def photo_z_dilution_factor(z_l, table_c, cosmology, weighting=-2,
                            lens_source_cut=0):
    """Calculate the photo-z delta sigma bias as a function of lens redshift.

    Parameters
    ----------
    z_l : float or numpy.ndarray
        Redshift(s) of the lens.
    table_c : astropy.table.Table
        Photometric redshift calibration catalog.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.
    weighting : float, optional
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
        Default is -2.
    lens_source_cut : None, float or numpy.ndarray, optional
        Determine the lens-source redshift separation cut. If None, no cut is
        applied. If a float, determines the minimum redshift separation between
        lens and source redshift for lens-source pairs to be used. If an array,
        it has to be the same length as the source table and determines the
        maximum lens redshift for a lens-source pair to be used. Default is 0.

    Returns
    -------
    f_bias : float or numpy.ndarray
        The photo-z bias factor, `f_bias`, for the lens redshift(s).

    """
    if lens_source_cut is None:
        z_l_max = np.repeat(np.amax(z_l) + 1, len(table_c))
    elif isinstance(lens_source_cut, numbers.Number):
        z_l_max = table_c['z'] - lens_source_cut
    else:
        z_l_max = np.array(lens_source_cut)

    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(table_c['z']).to(u.Mpc).value
    d_s_true = cosmology.comoving_transverse_distance(
        table_c['z_true']).to(u.Mpc).value
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

    mask = (z_l_max < z_l) | (z_l > z_s)

    if np.any(np.all(mask, axis=-1)):
        warnings.warn('Could not find valid calibration sources for some ' +
                      'lens redshifts. The f_bias correction may be ' +
                      'undefined.', RuntimeWarning)

    return (np.sum((w * sigma_crit_phot**weighting) * (~mask), axis=-1) /
            np.sum((w * sigma_crit_phot**(weighting + 1) / sigma_crit_true) *
                   (~mask), axis=-1))


def mean_photo_z_offset(z_l, table_c, cosmology, weighting=-2,
                        lens_source_cut=0):
    """Calculate the mean offset of source photometric redshifts.

    Parameters
    ----------
    z_l : float or numpy.ndarray
        Redshift(s) of the lens.
    table_c : astropy.table.Table, optional
        Photometric redshift calibration catalog.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.
    weighting : float, optional
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
        Default is -2.
    lens_source_cut : None, float or numpy.ndarray, optional
        Determine the lens-source redshift separation cut. If None, no cut is
        applied. If a float, determines the minimum redshift separation between
        lens and source redshift for lens-source pairs to be used. If an array,
        it has to be the same length as the source table and determines the
        maximum lens redshift for a lens-source pair to be used. Default is 0.

    Returns
    -------
    dz : float or numpy.ndarray
        The mean source redshift offset for the lens redshift(s).

    """
    if lens_source_cut is None:
        z_l_max = np.repeat(np.amax(z_l) + 1, len(table_c))
    elif isinstance(lens_source_cut, numbers.Number):
        z_l_max = table_c['z'] - lens_source_cut
    else:
        z_l_max = np.array(lens_source_cut)

    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(table_c['z']).to(
        u.Mpc).value
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
    w = w * sigma_crit**weighting

    mask = (z_l_max < z_l) | (z_l > z_s)

    return np.sum((z_s - z_s_true) * w * (~mask), axis=-1) / np.sum(
        w * (~mask), axis=-1)


def get_raw_multiprocessing_array(array):
    """Convert a numpy array into a shared-memory multiprocessing array.

    Parameters
    ----------
    array : numpy.ndarray or None
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


def precompute(
        table_l, table_s, bins, table_c=None, table_n=None,
        cosmology=FlatLambdaCDM(H0=100, Om0=0.3), comoving=True,
        weighting=-2, lens_source_cut=0, nside=256, n_jobs=1,
        progress_bar=False):
    """For all lenses in the catalog, precompute the lensing statistics.

    Parameters
    ----------
    table_l : astropy.table.Table
        Catalog of lenses.
    table_s : astropy.table.Table
        Catalog of sources.
    bins : numpy.ndarray or astropy.units.quantity.Quantity
        Bins in radius to use for the stacking. If a numpy array, bins are
        assumed to be in Mpc. If an astropy quantity, one can pass both length
        units, e.g. kpc and Mpc, as well as angular units, i.e. deg and rad.
    table_c : astropy.table.Table, optional
        Additional photometric redshift calibration catalog. If provided, this
        will be used to statistically correct the photometric source redshifts
        and critical surface densities. Default is None.
    table_n : astropy.table.Table, optional
        Source redshift distributions. If provided, this will be used to
        compute mean source redshifts and critical surface densities. These
        mean quantities would be used instead the individual photometric
        redshift estimates. The table needs to have a `z` column giving the
        redshift and a `n` column with the :math:`n(z)` for all samples.
        Default is None.
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations. Default is a flat LambdaCDM
        cosmology with h=1 and Om0=0.3.
    comoving : bool, optional
        Whether to use comoving or physical quantities for radial bins (if
        given in physical units) and the excess surface density. Default is
        True.
    weighting : float, optional
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
        Default is -2.
    lens_source_cut : None, float or numpy.ndarray, optional
        Determine the lens-source redshift separation cut. If None, no cut is
        applied. If a float, determines the minimum redshift separation between
        lens and source redshift for lens-source pairs to be used. If an array,
        it has to be the same length as the source table and determines the
        maximum lens redshift for a lens-source pair to be used. Default is 0.
    nside : int, optional
        dsigma uses pixelization to group nearby lenses together and process
        them simultaneously. This parameter determines the number of pixels.
        It has to be a power of 2. May impact performance. Default is 256.
    n_jobs : int, optional
        Number of jobs to run at the same time. Default is 1.
    progress_bar : bool, option
        Whether to show a progress bar for the main loop over lens pixels.
        Default is False.

    Returns
    -------
    table_l : astropy.table.Table
        Lens catalog with the pre-computation results attached to the table.

    Raises
    ------
    ValueError
        If there are problems in the input.

    """
    try:
        assert cosmology.Ok0 == 0
    except AssertionError:
        raise ValueError('dsigma does not support non-flat cosmologies.')

    if np.any(table_l['z'] < 0):
        raise ValueError('Input lens redshifts must all be non-negative.')
    if not isinstance(nside, int) or not np.isin(nside, 2**np.arange(15)):
        raise ValueError('nside must be a positive power of 2. Received ' +
                         '{}.'.format(nside))
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError('Number of jobs must be positive integer. Received ' +
                         '{}.'.format(n_jobs))

    if table_n is not None:
        if 'z_bin' not in table_s.colnames:
            raise ValueError('To use source redshift distributions, the ' +
                             'source table needs to have a `z_bin` column.')
        if (not np.issubdtype(table_s['z_bin'].data.dtype, np.integer) or
                np.any(table_s['z_bin'] < 0)):
            raise ValueError('The `z_bin` column in the source table must ' +
                             'contain only non-negative integers.')
        if np.amax(table_s['z_bin']) > table_n['n'].data.shape[1]:
            raise ValueError('The source table contains more redshift bins ' +
                             'than where passed via the nz argument.')

    hp = HEALPix(nside, order='ring')
    pix_l = hp.lonlat_to_healpix(table_l['ra'] * u.deg, table_l['dec'] * u.deg)
    pix_s = hp.lonlat_to_healpix(table_s['ra'] * u.deg, table_s['dec'] * u.deg)
    argsort_pix_l = np.argsort(pix_l)
    argsort_pix_s = np.argsort(pix_s)
    u_pix_l, n_pix_l = np.unique(pix_l, return_counts=True)
    u_pix_l = np.ascontiguousarray(u_pix_l)
    n_pix_l = np.ascontiguousarray(np.cumsum(n_pix_l))
    u_pix_s, n_pix_s = np.unique(pix_s, return_counts=True)
    u_pix_s = np.ascontiguousarray(u_pix_s)
    n_pix_s = np.ascontiguousarray(np.cumsum(n_pix_s))

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

    for key in ['w', 'e_1', 'e_2', 'm', 'e_rms', 'm_sel', 'R_11', 'R_22',
                'R_12', 'R_21']:
        if key in table_s.colnames:
            table_engine_s[key] = np.ascontiguousarray(
                table_s[key][argsort_pix_s], dtype=np.float64)

    if lens_source_cut is None:
        z_l_max = np.repeat(np.amax(table_l['z']) + 1, len(table_s))
    elif isinstance(lens_source_cut, numbers.Number):
        z_l_max = table_s['z'] - lens_source_cut
    else:
        z_l_max = np.array(lens_source_cut)

    table_engine_s['z_l_max'] = np.ascontiguousarray(
        z_l_max[argsort_pix_s], dtype=np.float64)

    for table, argsort_pix, table_engine in zip(
            [table_l, table_s], [argsort_pix_l, argsort_pix_s],
            [table_engine_l, table_engine_s]):

        z_min = np.amin(table['z'])
        z_max = np.amax(table['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.0001)))

        table_engine['d_com'] = np.ascontiguousarray(interp1d(
            z_interp, cosmology.comoving_transverse_distance(z_interp).to(
                u.Mpc).value)(table['z'])[argsort_pix])

    if table_c is not None and table_n is None:
        z_min = np.amin(table_l['z'])
        z_max = np.amax(table_l['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.001)))
        f_bias_interp = photo_z_dilution_factor(
            z_interp, table_c, cosmology, weighting=weighting,
            lens_source_cut=lens_source_cut)
        f_bias_interp = interp1d(
            z_interp, f_bias_interp, kind='cubic', bounds_error=False,
            fill_value=(f_bias_interp[0], f_bias_interp[-1]))
        table_engine_l['f_bias'] = np.ascontiguousarray(
            f_bias_interp(np.array(table_engine_l['z'])), dtype=np.float64)
        dz_s_interp = mean_photo_z_offset(
            z_interp, table_c=table_c, cosmology=cosmology,
            weighting=weighting)
        dz_s_interp = interp1d(
            z_interp, dz_s_interp, kind='cubic', bounds_error=False,
            fill_value=(dz_s_interp[0], dz_s_interp[-1]))
        table_engine_l['delta z_s'] = np.ascontiguousarray(
            dz_s_interp(np.array(table_engine_l['z'])), dtype=np.float64)

    elif table_c is None and table_n is not None:
        n_bins = table_n['n'].data.shape[1]
        sigma_crit_eff = np.zeros(len(table_l) * n_bins, dtype=np.float64)
        z_mean = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            z_min = np.amin(table_l['z'])
            z_max = min(np.amax(table_l['z']),
                        np.amax(table_n['z'][table_n['n'][:, i] > 0]))
            z_interp = np.linspace(
                z_min, z_max, max(10, int((z_max - z_min) / 0.001)))

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
            z_mean[i] = np.average(table_n['z'], weights=table_n['n'][:, i])
        table_engine_l['sigma_crit_eff'] = np.ascontiguousarray(
            sigma_crit_eff, dtype=np.float64)
        table_engine_s['z_bin'] = np.ascontiguousarray(
            table_s['z_bin'][argsort_pix_s], dtype=int)
        # Overwrite the photometric redshifts in the source table. These
        # redshifts will be used to compute the mean source redshifts for each
        # lens.
        table_engine_s['z'] = np.ascontiguousarray(
            z_mean[table_s['z_bin']][argsort_pix_s], dtype=np.float64)

    elif table_c is not None and table_s is not None:
        raise ValueError('table_c and table_n cannot both be given.')

    # Create arrays that will hold the final results.
    table_engine_r = {}
    n_results = len(table_l) * (len(bins) - 1)

    key_list = ['sum 1', 'sum w_ls', 'sum w_ls e_t', 'sum w_ls z_s',
                'sum w_ls e_t sigma_crit', 'sum w_ls sigma_crit']

    if 'm' in table_s.colnames:
        key_list.append('sum w_ls m')

    if 'e_rms' in table_s.colnames:
        key_list.append('sum w_ls (1 - e_rms^2)')

    if 'm_sel' in table_s.colnames:
        key_list.append('sum w_ls m_sel')

    if (('R_11' in table_s.colnames) and ('R_12' in table_s.colnames) and
            ('R_21' in table_s.colnames) and ('R_22' in table_s.colnames)):
        key_list.append('sum w_ls R_T')

    for key in key_list:
        table_engine_r[key] = np.ascontiguousarray(
            np.zeros(n_results, dtype=(
                np.int64 if key == 'sum 1' else np.float64)))

    z_l = np.array(table_engine_l['z'])
    d_com_l = np.array(table_engine_l['d_com'])

    if not isinstance(bins, u.quantity.Quantity):
        bins = bins * u.Mpc

    try:
        theta_bins = np.tile(bins.to(u.rad).value, len(table_l))
    except UnitConversionError:
        theta_bins = (np.tile(bins.to(u.Mpc).value, len(table_l)) /
                      np.repeat(d_com_l, len(bins))).flatten()
        if not comoving:
            theta_bins *= (1 + np.repeat(z_l, len(bins))).flatten()

    dist_3d_sq_bins = np.minimum(4 * np.sin(theta_bins / 2.0)**2, 2.0)

    # When running in parrallel, replace numpy arrays with shared-memory
    # multiprocessing arrays.
    if n_jobs > 1:
        dist_3d_sq_bins = get_raw_multiprocessing_array(dist_3d_sq_bins)
        for table_engine in [table_engine_l, table_engine_s, table_engine_r]:
            for key in table_engine.keys():
                table_engine[key] = get_raw_multiprocessing_array(
                    table_engine[key])

    # Create a queue that holds all the pixels containing lenses.
    if n_jobs == 1:
        queue = Queue.Queue()
    else:
        queue = mp.Queue()

    for i in range(len(u_pix_l)):
        queue.put(i)

    args = (u_pix_l, n_pix_l, u_pix_s, n_pix_s, dist_3d_sq_bins,
            table_engine_l, table_engine_s, table_engine_r, bins, comoving,
            weighting, nside, queue, progress_bar)

    if n_jobs == 1:
        precompute_engine(*args)
    else:
        processes = []
        for i in range(n_jobs):
            process = mp.Process(target=precompute_engine, args=(*args, ))
            if i == 0:
                args = list(args)
                args[-1] = False
                args = tuple(args)
            process.start()
            processes.append(process)
        for i in range(n_jobs):
            processes[i].join()

    inv_argsort_pix_l = np.argsort(argsort_pix_l)
    for key in table_engine_r.keys():
        table_l[key] = np.array(table_engine_r[key]).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]

    table_l['sum w_ls z_l'] = table_l['z'][:, np.newaxis] * table_l['sum w_ls']

    if 'f_bias' in table_engine_l.keys():
        table_l['sum w_ls sigma_crit f_bias'] = (
            np.array(table_engine_l['f_bias'])[inv_argsort_pix_l][
                :, np.newaxis] * table_l['sum w_ls sigma_crit'])
        table_l['sum w_ls e_t sigma_crit f_bias'] = (
            np.array(table_engine_l['f_bias'])[inv_argsort_pix_l][
                :, np.newaxis] * table_l['sum w_ls e_t sigma_crit'])

    if 'delta z_s' in table_engine_l.keys():
        table_l['sum w_ls z_s'] = (
            table_l['sum w_ls z_s'] - table_l['sum w_ls'] * np.array(
                table_engine_l['delta z_s'])[inv_argsort_pix_l][:, np.newaxis])

    table_l.meta['bins'] = bins
    table_l.meta['comoving'] = comoving
    table_l.meta['H0'] = cosmology.H0.value
    table_l.meta['Ok0'] = cosmology.Ok0
    table_l.meta['Om0'] = cosmology.Om0
    table_l.meta['weighting'] = weighting

    return table_l
