import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from . import surveys
from .physics import mpc_per_degree, lens_magnification_shear_bias

__all__ = ['number_of_pairs', 'raw_tangential_shear',
           'raw_excess_surface_density', 'photo_z_dilution_factor',
           'boost_factor', 'scalar_shear_response_factor',
           'matrix_shear_response_factor', 'shear_responsivity_factor',
           'mean_lens_redshift', 'mean_source_redshift',
           'mean_critical_surface_density', 'tangential_shear',
           'excess_surface_density', 'shape_noise_error']


def number_of_pairs(table_l):
    """Compute the number of lens-source pairs per bin.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    n_pairs : numpy array
        The number of lens-source pairs in each radial bin.
    """

    return np.sum(table_l['sum 1'].data, axis=0)


def raw_tangential_shear(table_l):
    """Compute the average tangential shear for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy array
        The raw, uncorrected tangential shear in each radial bin.
    """

    return (np.sum(table_l['sum w_ls e_t'].data *
                   table_l['w_sys'].data[:, None], axis=0) /
            np.sum(table_l['sum w_ls'].data * table_l['w_sys'].data[:, None],
                   axis=0))


def raw_excess_surface_density(table_l):
    """Compute the raw, uncorrected excess surface density for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy array
        The raw, uncorrected excess surface density in each radial bin.
    """

    return (np.sum(table_l['sum w_ls e_t sigma_crit'].data *
                   table_l['w_sys'].data[:, None], axis=0) /
            np.sum(table_l['sum w_ls'].data *
                   table_l['w_sys'].data[:, None], axis=0))


def photo_z_dilution_factor(table_l):
    r"""Compute the photometric redshift bias averaged over the entire catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    f_bias : float
        Photometric redshift bias :math:`f_{\mathrm{bias}}`.
    """

    return (np.sum(table_l['sum w_ls e_t sigma_crit f_bias'].data *
                   table_l['w_sys'].data[:, None], axis=0) /
            np.sum(table_l['sum w_ls e_t sigma_crit'].data *
                   table_l['w_sys'].data[:, None], axis=0))


def boost_factor(table_l, table_r):
    """Compute the boost factor by comparing the number of lens-source pairs
    in real lenses and random lenses.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.

    Returns
    -------
    b : numpy array
        Boost factor in each radial bin.
    """

    return (
        np.sum(table_l['sum w_ls'].data *
               table_l['w_sys'].data[:, None], axis=0) /
        np.sum(table_l['w_sys'].data) /
        np.sum(table_r['sum w_ls'].data *
               table_r['w_sys'].data[:, None], axis=0) *
        np.sum(table_r['w_sys'].data))


def scalar_shear_response_factor(table_l):
    r"""Compute the mean :math:`m` factor such that :math:`\gamma_{\mathrm obs}
    = (1 + m) \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    m : numpy array
        Multiplicative shear bias in each radial bin.
    """

    return (
        np.sum(table_l['sum w_ls m'].data *
               table_l['w_sys'].data[:, None], axis=0) /
        np.sum(table_l['sum w_ls'].data *
               table_l['w_sys'].data[:, None], axis=0))


def matrix_shear_response_factor(table_l):
    r"""Compute the mean tangential response :math:`R_t` factor such that
    :math:`\gamma_{\mathrm obs} = R_t \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r_t : numpy array
        Tangential shear response factor in each radial bin.
    """

    return (
        np.sum(table_l['sum w_ls R_T'] * table_l['w_sys'][:, None],
               axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def shear_responsivity_factor(table_l):
    """Compute the shear responsitivity factor.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r : numpy array
        Shear responsitivity factor in each radial bin.
    """

    return (
        np.sum(table_l['sum w_ls (1 - e_rms^2)'] *
               table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def mean_lens_redshift(table_l):
    """Compute the weighted-average lens redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    z_l : numpy array
        Mean lens redshift in each bin.
    """

    return (
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None] *
               table_l['z'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def mean_source_redshift(table_l):
    """Compute the weighted-average source redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    z_s : numpy array
        Mean source redshift in each bin.
    """

    return (
        np.sum(table_l['sum w_ls z_s'] * table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def mean_critical_surface_density(table_l):
    """Compute the weighted-average critical surface density.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    sigma_crit : numpy array
        Mean critical surface_density in each bin.
    """

    return (
        np.sum(table_l['sum w_ls sigma_crit'] * table_l['w_sys'][:, None],
               axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def lens_magnification_bias(table_l, alpha_l, camb_results,
                            photo_z_dilution_correction=True):
    """Estimate the additive lens magnification bias.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    alpha_l : float
        TBD
    camb_results : camb.results.CAMBdata
        CAMB results object that contains information on cosmology and the
        matter power spectrum.
    photo_z_dilution_correction : boolean, optional
        Whether to correct for photo-z dilution.

    Returns
    -------
    ds_lm : numpy array
        The lens magnification bias in each radial bin.
    """

    z_l = mean_lens_redshift(table_l)
    z_s = mean_source_redshift(table_l)
    sigma_crit = mean_critical_surface_density(table_l)

    if photo_z_dilution_correction:
        sigma_crit = sigma_crit * photo_z_dilution_factor(table_l)

    rp_bins = table_l.meta['rp_bins']
    rp = 2.0 / 3.0 * np.diff(rp_bins**3) / np.diff(rp_bins**2)
    cosmo = FlatLambdaCDM(H0=table_l.meta['H0'], Om0=table_l.meta['Om0'])
    theta = np.deg2rad(rp / mpc_per_degree(
        z_l, cosmology=cosmo, comoving=table_l.meta['comoving']))

    return np.array([lens_magnification_shear_bias(
        theta[i], alpha_l, z_l[i], z_s[i], camb_results) for i in
        range(len(theta))]) * sigma_crit


def tangential_shear(table_l, table_r=None, photo_z_dilution_correction=False,
                     boost_correction=False,
                     scalar_shear_response_correction=False,
                     matrix_shear_response_correction=False,
                     shear_responsivity_correction=False,
                     hsc_selection_bias_correction=False,
                     random_subtraction=False, return_table=False):
    """Compute the total excess surface density signal (including all
    corrections) from precompute results.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.
    boost_correction : boolean, optional
        If true, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided.
    scalar_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
    matrix_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
    shear_responsivity_correction : boolean, optional
        If true, correct for the shear responsivity.
    hsc_selection_bias_correction : boolean, optional
        If true, correct for the multiplicative selection bias in HSC.
    random_subtraction : boolean, optional
        If true, subtract the signal around randoms. This can only be done if
        a random catalog is provided.
    return_table : boolean, optional
        If true, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final excess
        surface density is returned.

    Returns
    -------
    delta_sigma : numpy array or astropy.table.Table
        The excess surface density in each radial bin specified in the
        precomputation phase.
    """

    result = Table()

    result['rp_min'] = table_l.meta['bins'][:-1]
    result['rp_max'] = table_l.meta['bins'][1:]
    result['n_pairs'] = number_of_pairs(table_l)
    result['rp'] = np.sqrt(result['rp_min'] * result['rp_max'])
    result['et_raw'] = raw_tangential_shear(table_l)
    result['et'] = raw_tangential_shear(table_l)
    result['z_l'] = mean_lens_redshift(table_l)
    result['z_s'] = mean_source_redshift(table_l)

    if boost_correction:
        if table_r is None:
            raise Exception('Cannot compute boost factor correction without ' +
                            'results from a random catalog.')
        result['b'] = boost_factor(table_l, table_r)
        result['et'] *= result['b']

    if scalar_shear_response_correction:
        result['1+m'] = 1 + scalar_shear_response_factor(table_l)
        result['et'] /= result['1+m']

    if matrix_shear_response_correction:
        result['R_t'] = matrix_shear_response_factor(table_l)
        result['et'] /= result['R_t']

    if shear_responsivity_correction:
        result['2R'] = 2 * shear_responsivity_factor(table_l)
        result['et'] /= result['2R']

    if hsc_selection_bias_correction:
        result['1+m_sel'] = 1 + surveys.hsc.selection_bias_factor(
            table_l)
        result['et'] *= result['1+m_sel']

    if random_subtraction:
        if table_r is None:
            raise Exception('Cannot subtract random results without ' +
                            'results from a random catalog.')
        result['et_r'] = excess_surface_density(
            table_r, boost_correction=False,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            hsc_selection_bias_correction=hsc_selection_bias_correction,
            random_subtraction=False, return_table=False)
        result['et'] -= result['et_r']

    if not return_table:
        return result['et']

    return result


def excess_surface_density(table_l, table_r=None,
                           photo_z_dilution_correction=False,
                           boost_correction=False,
                           scalar_shear_response_correction=False,
                           matrix_shear_response_correction=False,
                           shear_responsivity_correction=False,
                           hsc_selection_bias_correction=False,
                           random_subtraction=False,
                           return_table=False):
    """Compute the total excess surface density signal (including all
    corrections) from precompute results.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.
    photo_z_dilution_correction : boolean, optional
        If true, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the Precomputation phase.
    boost_correction : boolean, optional
        If true, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided.
    scalar_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
    matrix_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
    shear_responsivity_correction : boolean, optional
        If true, correct for the shear responsivity.
    hsc_selection_bias_correction : boolean, optional
        If true, correct for the multiplicative selection bias in HSC.
    random_subtraction : boolean, optional
        If true, subtract the signal around randoms. This can only be done if
        a random catalog is provided.
    return_table : boolean, optional
        If true, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final excess
        surface density is returned.

    Returns
    -------
    delta_sigma : numpy array or astropy.table.Table
        The excess surface density in each radial bin specified in the
        precomputation phase.
    """

    result = Table()

    result['rp_min'] = table_l.meta['bins'][:-1]
    result['rp_max'] = table_l.meta['bins'][1:]
    result['n_pairs'] = number_of_pairs(table_l)
    result['rp'] = np.sqrt(result['rp_min'] * result['rp_max'])
    result['ds_raw'] = raw_excess_surface_density(table_l)
    result['ds'] = raw_excess_surface_density(table_l)
    result['z_l'] = mean_lens_redshift(table_l)
    result['z_s'] = mean_source_redshift(table_l)

    if boost_correction:
        if table_r is None:
            raise Exception('Cannot compute boost factor correction without ' +
                            'results from a random catalog.')
        result['b'] = boost_factor(table_l, table_r)
        result['ds'] *= result['b']

    if scalar_shear_response_correction:
        result['1+m'] = 1 + scalar_shear_response_factor(table_l)
        result['ds'] /= result['1+m']

    if matrix_shear_response_correction:
        result['R_t'] = matrix_shear_response_factor(table_l)
        result['ds'] /= result['R_t']

    if shear_responsivity_correction:
        result['2R'] = 2 * shear_responsivity_factor(table_l)
        result['ds'] /= result['2R']

    if hsc_selection_bias_correction:
        result['1+m_sel'] = 1 + surveys.hsc.selection_bias_factor(
            table_l)
        result['ds'] *= result['1+m_sel']

    if photo_z_dilution_correction:
        result['f_bias'] = photo_z_dilution_factor(table_l)
        result['ds'] *= result['f_bias']

    if random_subtraction:
        if table_r is None:
            raise Exception('Cannot subtract random results without ' +
                            'results from a random catalog.')
        result['ds_r'] = excess_surface_density(
            table_r, photo_z_dilution_correction=photo_z_dilution_correction,
            boost_correction=False,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            hsc_selection_bias_correction=hsc_selection_bias_correction,
            random_subtraction=False, return_table=False)
        result['ds'] -= result['ds_r']

    if not return_table:
        return result['ds']

    return result


def shape_noise_error(table_l, table_r=None, **kwargs):
    """Analytically estimate the shape noise error for the excess surface
    density. Note that on large scales, there is additional cosmic variance
    error. Thus, jackknife re-sampling should be used for those scales.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.
    kwargs : dict
        Additional keyword arguments to be passed to the
        `excess_surface_density` function. This is used to calculate correction
        factors.

    Returns
    -------
    delta_sigma_error : numpy array
        Shape noise error estimate for the excess surface density.
    """

    kwargs['return_table'] = True
    table = excess_surface_density(table_l, **kwargs)
    correction_factor = table['ds'] / table['ds_raw']

    error = (np.sqrt(np.sum(table_l['sum (w_ls e_t sigma_crit)^2'] *
                            table_l['w_sys'][:, None], axis=0)) /
             np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))

    if table_r is not None:
        table = excess_surface_density(table_r, **kwargs)
        correction_factor = table['ds'] / table['ds_raw']
        error_r = (np.sqrt(np.sum(table_r['sum (w_ls e_t sigma_crit)^2'] *
                                  table_r['w_sys'][:, None], axis=0)) /
                   np.sum(table_r['sum w_ls'] * table_r['w_sys'][:, None],
                          axis=0))
        error = np.sqrt(error**2 + error_r**2)

    return error * correction_factor
