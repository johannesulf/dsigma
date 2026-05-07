"""Module for pre-computing lensing results."""

import multiprocessing as mp
import queue
import warnings

import numpy as np
from astropy import units as u
from astropy.units import UnitConversionError
from astropy_healpix import HEALPix

from . import default_cosmology
from .helpers import interpolate_over_redshift
from .physics import critical_surface_density
from .physics import effective_critical_surface_density
from .precompute_engine import precompute_engine

__all__ = ['mean_photo_z_offset', 'photo_z_dilution_factor', 'precompute']


def photo_z_dilution_factor(z_l, table_c, cosmology, weighting=-2):
    r"""Calculate the photo-z delta sigma bias as a function of lens redshift.

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

    Returns
    -------
    f_bias : float or numpy.ndarray
        The photo-z bias factor, :math:`f_{\rm bias}`, for the lens
        redshift(s).

    """
    z_l_max = table_c['z_l_max']
    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = interpolate_over_redshift(
        cosmology.comoving_transverse_distance, z_l).to(u.Mpc).value
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

    mask = z_l >= z_l_max

    if np.any(np.all(mask, axis=-1)):
        msg = ("Could not find valid calibration sources for some lens "
               "redshifts. The f_bias correction may be undefined.")
        warnings.warn(msg, category=RuntimeWarning, stacklevel=2)

    return (np.sum((w * sigma_crit_phot**weighting) * (~mask), axis=-1) /
            np.sum((w * sigma_crit_phot**(weighting + 1) / sigma_crit_true) *
                   (~mask), axis=-1))


def mean_photo_z_offset(z_l, table_c, cosmology, weighting=-2):
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

    Returns
    -------
    dz : float or numpy.ndarray
        The mean source redshift offset for the lens redshift(s).

    """
    z_l_max = table_c['z_l_max']
    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = interpolate_over_redshift(
        cosmology.comoving_transverse_distance, z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(table_c['z']).to(u.Mpc).value
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

    mask = z_l >= z_l_max

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
        cosmology=None, comoving=True, weighting=-2, nside=256, n_jobs=1,
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
        and critical surface densities. Default is ``None``.
    table_n : astropy.table.Table, optional
        Source redshift distributions. If provided, this will be used to
        compute mean source redshifts and critical surface densities. These
        mean quantities would be used instead the individual photometric
        redshift estimates. The table needs to have a `z` column giving the
        redshift and a `n` column with the :math:`n(z)` for all samples.
        Default is ``None``.
    cosmology : astropy.cosmology or None, optional
        Cosmology to assume for calculations. If ``None``, use
        ``dsigma.default_cosmology``. Default is ``None``.
    comoving : bool, optional
        Whether to use comoving or physical quantities for radial bins (if
        given in physical units) and the excess surface density. Default is
        True.
    weighting : float, optional
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
        Default is -2.
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
    cosmology = default_cosmology if cosmology is None else cosmology

    if cosmology.Ok0 != 0:
        msg = "dsigma does not support non-flat cosmologies."
        raise ValueError(msg)

    if np.any(table_l['z'] < 0):
        msg = "Input lens redshifts must all be non-negative."
        raise ValueError(msg)

    if not isinstance(nside, int) or not np.isin(nside, 2**np.arange(15)):
        msg = f"nside must be a positive power of 2. Received {nside}."
        raise ValueError(msg)

    if not isinstance(n_jobs, int) or n_jobs < 1:
        msg = f"Number of jobs must be positive integer. Received {n_jobs}."
        raise ValueError(msg)

    if table_c is not None and table_n is not None:
        msg = "`table_c` and `table_n` cannot both be given."
        raise ValueError(msg)

    if table_n is not None:
        if 'z' in table_s.colnames:
            msg = ("When providing tomographic source redshift distributions "
                   "via the `table_n` argument, the `z` column is ignored.")
            warnings.warn(msg, category=RuntimeWarning, stacklevel=2)
        if 'z_bin' not in table_s.colnames:
            msg = ("To use source redshift distributions, the source table "
                   "needs to have a `z_bin` column.")
            raise ValueError(msg)
        if (not np.issubdtype(table_s['z_bin'].dtype, np.integer) or
                np.any(table_s['z_bin'] < 0)):
            msg = ("The `z_bin` column in the source table must contain only "
                   "non-negative integers.")
            raise ValueError(msg)
        if np.amax(table_s['z_bin']) > table_n['n'].data.shape[1]:
            msg = ("The source table contains more redshift bins than where "
                   "passed via the `table_n` argument.")
            raise ValueError(msg)
    elif 'z_l_max' in table_s.colnames and np.any(
            table_s['z_l_max'] > table_s['z']):
        msg = ("The maximum lens redshift can never be larger than the source "
               "redshift.")
        raise ValueError(msg)

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

    for f, f_name in zip([np.sin, np.cos], ['sin', 'cos']):
        for table, argsort_pix, table_engine in zip(
                [table_l, table_s], [argsort_pix_l, argsort_pix_s],
                [table_engine_l, table_engine_s]):
            for angle in ['ra', 'dec']:
                table_engine[f'{f_name} {angle}'] =\
                    np.ascontiguousarray(f(np.deg2rad(table[angle]))[
                        argsort_pix])

    for key in ['z', 'z_l_max', 'w', 'e_1', 'e_2', 'm', 'e_rms', 'm_sel',
                'R_11', 'R_22', 'R_12', 'R_21']:
        if key in table_s.colnames:
            table_engine_s[key] = np.ascontiguousarray(
                table_s[key][argsort_pix_s], dtype=np.float64)

    if 'z_bin' in table_s.colnames:
        table_engine_s['z_bin'] = np.ascontiguousarray(
            table_s['z_bin'][argsort_pix_s], dtype=int)

    if 'z_l_max' not in table_engine_s:
        if table_n is not None:
            table_engine_s['z_l_max'] = np.ascontiguousarray(
                np.repeat(np.finfo(np.float64).max, len(table_s)),
                dtype=np.float64)
        else:
            table_engine_s['z_l_max'] = table_engine_s['z']

    for table, argsort_pix, table_engine in zip(
            [table_l, table_s], [argsort_pix_l, argsort_pix_s],
            [table_engine_l, table_engine_s]):
        if 'z' in table.colnames:
            table_engine['d_com'] = np.ascontiguousarray(
                interpolate_over_redshift(
                    cosmology.comoving_transverse_distance,
                    table_engine['z']).to(u.Mpc).value, dtype=np.float64)

    if table_c is not None:
        table_c_copy = table_c.copy()
        if 'z_l_max' not in table_c_copy.colnames:
            table_c_copy['z_l_max'] = table_c_copy['z']
        table_engine_l['f_bias'] = np.ascontiguousarray(
            interpolate_over_redshift(
                photo_z_dilution_factor, table_engine_l['z'], table_c_copy,
                cosmology, weighting=weighting), dtype=np.float64)
        table_engine_l['delta z_s'] = np.ascontiguousarray(
            interpolate_over_redshift(
                mean_photo_z_offset, table_engine_l['z'], table_c_copy,
                cosmology, weighting=weighting), dtype=np.float64)

    if table_n is not None:
        n_bins = table_n['n'].data.shape[1]
        z_ave = np.array([
            np.average(table_n['z'], weights=table_n['n'][:, i]) for i in
            range(n_bins)])
        table_engine_s['z'] = np.ascontiguousarray(
            z_ave[table_engine_s['z_bin']], dtype=np.float64)

        def _inverse_effective_critical_surface_density(z_l, z_bin):
            return 1.0 / effective_critical_surface_density(
                z_l, table_n['z'], table_n['n'][:, z_bin],
                cosmology=cosmology, comoving=comoving)

        sigma_crit_eff_inv = np.zeros(len(table_l) * n_bins, dtype=np.float64)

        for z_bin in range(n_bins):
            sigma_crit_eff_inv[z_bin::n_bins] = interpolate_over_redshift(
                _inverse_effective_critical_surface_density,
                table_engine_l['z'], z_bin)

        with np.errstate(divide='ignore'):
            sigma_crit_eff = np.where(
                sigma_crit_eff_inv > 0, 1.0 / sigma_crit_eff_inv,
                np.finfo(np.float64).max)

        table_engine_l['sigma_crit_eff'] = np.ascontiguousarray(
            sigma_crit_eff, dtype=np.float64)

    # Create arrays that will hold the final results.
    table_engine_r = {}
    n_results = len(table_l) * (len(bins) - 1)

    key_list = ['sum 1', 'sum w_ls', 'sum w_ls e_t', 'sum w_ls z_s',
                'sum w_ls e_t sigma_crit', 'sum w_ls sigma_crit']

    for table_s_key, table_r_key in zip(
        ['m', 'e_rms', 'm_sel', 'R_11'],
        ['sum w_ls m', 'sum w_ls (1 - e_rms^2)', 'sum w_ls m_sel',
         'sum w_ls R_T']):
        if table_s_key in table_s.colnames:
            key_list.append(table_r_key)

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
            for key in table_engine:
                table_engine[key] = get_raw_multiprocessing_array(
                    table_engine[key])

    # Create a queue that holds all the pixels containing lenses.
    q = queue.Queue() if n_jobs == 1 else mp.Queue()
    for i in range(len(u_pix_l)):
        q.put(i)

    args = (u_pix_l, n_pix_l, u_pix_s, n_pix_s, dist_3d_sq_bins,
            table_engine_l, table_engine_s, table_engine_r, bins, comoving,
            weighting, nside, q, progress_bar)

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
    for key in table_engine_r:
        table_l[key] = np.array(table_engine_r[key]).reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]

    table_l['sum w_ls z_l'] = table_l['z'][:, np.newaxis] * table_l['sum w_ls']

    if 'f_bias' in table_engine_l:
        table_l['sum w_ls sigma_crit f_bias'] = (
            np.array(table_engine_l['f_bias'])[inv_argsort_pix_l][
                :, np.newaxis] * table_l['sum w_ls sigma_crit'])
        table_l['sum w_ls e_t sigma_crit f_bias'] = (
            np.array(table_engine_l['f_bias'])[inv_argsort_pix_l][
                :, np.newaxis] * table_l['sum w_ls e_t sigma_crit'])

    if 'delta z_s' in table_engine_l:
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
