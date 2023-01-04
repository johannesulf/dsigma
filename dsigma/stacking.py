"""Module for stacking lensing results after pre-computation."""

import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from . import surveys
from .physics import mpc_per_degree, lens_magnification_shear_bias
from .physics import critical_surface_density

__all__ = ['number_of_pairs', 'raw_tangential_shear',
           'raw_excess_surface_density', 'photo_z_dilution_factor',
           'boost_factor', 'scalar_shear_response_factor',
           'matrix_shear_response_factor', 'shear_responsivity_factor',
           'mean_lens_redshift', 'mean_source_redshift',
           'tangential_shear', 'excess_surface_density']


def number_of_pairs(table_l):
    """Compute the number of lens-source pairs per bin.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    n_pairs : numpy.ndarray
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
    delta_sigma : numpy.ndarray
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
    delta_sigma : numpy.ndarray
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
    """Compute the boost factor.

    Boost factor is computed by comparing the number of lens-source pairs
    in real lenses and random lenses.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.

    Returns
    -------
    b : numpy.ndarray
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
    r"""Compute the mean shear response.

    The shear response factor :math:`m` is defined such that
    :math:`\gamma_{\mathrm obs} = (1 + m) \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    m : numpy.ndarray
        Multiplicative shear bias in each radial bin.

    """
    return (
        np.sum(table_l['sum w_ls m'].data *
               table_l['w_sys'].data[:, None], axis=0) /
        np.sum(table_l['sum w_ls'].data *
               table_l['w_sys'].data[:, None], axis=0))


def matrix_shear_response_factor(table_l):
    r"""Compute the mean tangential response.

    The tangential shear response factor:math:`R_t` is defined such that
    :math:`\gamma_{\mathrm obs} = R_t \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r_t : numpy.ndarray
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
    r : numpy.ndarray
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
    z_l : numpy.ndarray
        Mean lens redshift in each bin.

    """
    return (
        np.sum(table_l['sum w_ls z_l'] * table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def mean_source_redshift(table_l, photo_z_correction=False):
    """Compute the weighted-average source redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    photo_z_correction : boolean, optional
        By default, this function returns the average photometric source
        redshift. If True and a calibration catalog or source redshift
        distribution has been provided at the precompute stage, estimate the
        intrinsic source redshift distribution. Default is False.

    Returns
    -------
    z_s : numpy.ndarray
        Mean source redshift in each bin.

    """
    if not photo_z_correction:
        key_num = 'sum w_ls z_s'
    else:
        key_num = 'sum w_ls (z_s - delta z_s)'

    return (
        np.sum(table_l[key_num] * table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def lens_magnification_bias(table_l, alpha_l, camb_results,
                            photo_z_correction=True):
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
    photo_z_correction : boolean, optional
        Whether to correct for photo-z dilution and offsets.

    Returns
    -------
    ds_lm : numpy.ndarray
        The lens magnification bias in each radial bin.

    """
    cosmology = FlatLambdaCDM(H0=table_l.meta['H0'], Om0=table_l.meta['Om0'])

    z_l = mean_lens_redshift(table_l)
    z_s = mean_source_redshift(table_l)
    if photo_z_correction:
        z_s_true = mean_source_redshift(table_l, photo_z_correction=True)
        try:
            sigma_crit = critical_surface_density(
                z_l, z_s, cosmology, comoving=table_l.meta['comoving'])
            sigma_crit *= photo_z_dilution_factor(table_l)
        except KeyError:
            sigma_crit = critical_surface_density(
                z_l, z_s_true, cosmology, comoving=table_l.meta['comoving'])
    else:
        sigma_crit = critical_surface_density(
            z_l, z_s, cosmology, comoving=table_l.meta['comoving'])
        z_s_true = z_s

    bins = table_l.meta['bins']

    shear_mode = ('shear_mode' in table_l.meta.keys() and
                  table_l.meta['shear_mode'])

    if shear_mode:
        theta = 2.0 / 3.0 * np.diff(bins**3) / np.diff(bins**2)
    else:
        rp = 2.0 / 3.0 * np.diff(bins**3) / np.diff(bins**2)
        theta = np.deg2rad(rp / mpc_per_degree(
            z_l, cosmology=cosmology, comoving=table_l.meta['comoving']))

    gamma = np.array([lens_magnification_shear_bias(
        theta[i], alpha_l, z_l[i], z_s_true[i], camb_results) for i in
        range(len(theta))])

    if shear_mode:
        return gamma
    else:
        return gamma * sigma_crit


def tangential_shear(table_l, table_r=None, boost_correction=False,
                     scalar_shear_response_correction=False,
                     matrix_shear_response_correction=False,
                     shear_responsivity_correction=False,
                     hsc_selection_bias_correction=False,
                     random_subtraction=False, return_table=False):
    """Compute the mean tangential shear with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is None.
    boost_correction : boolean, optional
        If True, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided. Default is False.
    scalar_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is False.
    matrix_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is False.
    shear_responsivity_correction : boolean, optional
        If True, correct for the shear responsivity. Default is False.
    hsc_selection_bias_correction : boolean, optional
        If True, correct for the multiplicative selection bias in HSC. Default
        is False.
    random_subtraction : boolean, optional
        If True, subtract the signal around randoms. This can only be done if
        a random catalog is provided. Default is False.
    return_table : boolean, optional
        If True, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final tangential
        shearis returned. Default is False.

    Returns
    -------
    e_t : numpy.ndarray or astropy.table.Table
        The tangential shear in each radial bin specified in the precomputation
        phase. If `return_table` is True, will return a table with detailed
        information for each radial bin. The final result is in the column
        `et`.

    Raises
    ------
    ValueError
        If boost or random subtraction correction are requested but no random
        catalog is provided.

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
            raise ValueError('Cannot compute boost factor correction without' +
                             ' results from a random catalog.')
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
            raise ValueError('Cannot subtract random results without ' +
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
        return result['et'].data

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
    """Compute the mean excess surface density with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is None.
    photo_z_dilution_correction : boolean, optional
        If True, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the Precomputation phase.
        Default is False.
    boost_correction : boolean, optional
        If true, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided. Default is False.
    scalar_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is False.
    matrix_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is False.
    shear_responsivity_correction : boolean, optional
        If True, correct for the shear responsivity. Default is False.
    hsc_selection_bias_correction : boolean, optional
        If True, correct for the multiplicative selection bias in HSC. Default
        is False.
    random_subtraction : boolean, optional
        If True, subtract the signal around randoms. This can only be done if
        a random catalog is provided. Default is False.
    return_table : boolean, optional
        If True, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final excess
        surface density is returned. Default is False.

    Returns
    -------
    delta_sigma : numpy.ndarray or astropy.table.Table
        The excess surface density in each radial bin specified in the
        precomputation phase. If `return_table` is True, will return a table
        with detailed information for each radial bin. The final result is in
        the column `ds`.

    Raises
    ------
    ValueError
        If boost or random subtraction correction are requested but no random
        catalog is provided.

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
            raise ValueError('Cannot compute boost factor correction without' +
                             ' results from a random catalog.')
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
            raise ValueError('Cannot subtract random results without ' +
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
        return result['ds'].data

    return result
