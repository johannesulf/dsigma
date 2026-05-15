"""Module for stacking lensing results after pre-computation."""

import numpy as np
from astropy import units as u
from astropy.cosmology import units as cu
from astropy.table import Table
from astropy.units import UnitConversionError

from .physics import lens_magnification_shear_bias, mpc_per_degree

__all__ = ['boost_factor', 'excess_surface_density', 'lens_magnification_bias',
           'matrix_shear_response_factor', 'mean_critical_surface_density',
           'mean_lens_redshift', 'mean_source_redshift', 'number_of_pairs',
           'photo_z_dilution_factor', 'raw_excess_surface_density',
           'raw_tangential_shear', 'scalar_shear_response_factor',
           'shear_responsivity_factor', 'tangential_shear']


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
    gt : numpy.ndarray
        The raw, uncorrected tangential shear in each radial bin.

    """
    return (np.dot(table_l['w_sys'].data, table_l['sum w_ls e_t'].data) /
            np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


def raw_excess_surface_density(table_l):
    """Compute the raw, uncorrected excess surface density for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    ds : numpy.ndarray
        The raw, uncorrected excess surface density in each radial bin.

    """
    return (np.dot(table_l['w_sys'].data,
                   table_l['sum w_ls e_t sigma_crit'].quantity) /
            np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


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
    return (np.dot(table_l['w_sys'].data,
                   table_l['sum w_ls e_t sigma_crit f_bias'].data) /
            np.dot(table_l['w_sys'].data,
                   table_l['sum w_ls e_t sigma_crit'].data))


def boost_factor(table_l, table_r):
    """Compute the boost factor.

    Boost factor is computed by comparing the number of lens-source pairs
    in real lenses and random lenses.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table
        Precompute results for random lenses.

    Returns
    -------
    b : numpy.ndarray
        Boost factor in each radial bin.

    """
    return (
        np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data) /
        np.dot(table_r['w_sys'].data, table_r['sum w_ls'].data) *
        np.sum(table_r['w_sys'].data) / np.sum(table_l['w_sys'].data))


def scalar_shear_response_factor(table_l, selection_bias=False):
    r"""Compute the mean shear response.

    The shear response factor :math:`m` is defined such that
    :math:`\gamma_{\mathrm obs} = (1 + m) \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    selection_bias : bool
        If ``True``, calculate the selection bias :math:`m_\mathrm{sel}`,
        instead. Default is ``False``.

    Returns
    -------
    m : numpy.ndarray
        Multiplicative shear bias in each radial bin.

    """
    m = 'm_sel' if selection_bias else 'm'

    return (np.dot(table_l['w_sys'].data, table_l[f'sum w_ls {m}'].data) /
            np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


def matrix_shear_response_factor(table_l):
    r"""Compute the mean tangential response.

    The tangential shear response factor :math:`R_t` is defined such that
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
    return (np.dot(table_l['w_sys'].data, table_l['sum w_ls R_T'].data) /
            np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


def shear_responsivity_factor(table_l):
    """Compute the shear responsivity factor.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r : numpy.ndarray
        Shear responsivity factor in each radial bin.

    """
    return (
        np.dot(table_l['w_sys'].data, table_l['sum w_ls (1 - e_rms^2)'].data) /
        np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


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
    return (np.dot(table_l['w_sys'].data, table_l['sum w_ls z_l'].data) /
            np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


def mean_source_redshift(table_l):
    """Compute the weighted-average source redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    z_s : numpy.ndarray
        Mean source redshift in each bin.

    """
    return (np.dot(table_l['w_sys'].data, table_l['sum w_ls z_s'].data) /
            np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


def mean_critical_surface_density(table_l, photo_z_dilution_correction=False):
    """Compute the weighted-average (effective) critical surface density.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    photo_z_dilution_correction : bool, optional
        If ``True``, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the precomputation phase.
        Default is ``False``.

    Returns
    -------
    sigma_crit : numpy.ndarray
        Mean (effective) critical surface density.

    """
    if photo_z_dilution_correction:
        key = 'sigma_crit f_bias'
    else:
        key = 'sigma_crit'
    return (
        np.dot(table_l['w_sys'].data, table_l[f'sum w_ls {key}'].quantity) /
        np.dot(table_l['w_sys'].data, table_l['sum w_ls'].data))


def lens_magnification_bias(table_l, alpha_l, sigma_8=0.82, n_s=0.96,
                            photo_z_dilution_correction=False, shear=False):
    r"""Estimate the additive lens magnification bias.

    Note that the assumed cosmology is taken from ``table_l.meta['cosmology']``
    which is added by ``precompute``.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    alpha_l : float
        The response of the lenses to magnification.
    sigma_8 : float, optional
        Scale of fluctuations at :math:`8 h^{-1} \, \mathrm{Mpc}`. Default is
        0.82.
    n_s : float, optional
        Primordial power spectrum index. Default is 0.96.
    photo_z_dilution_correction : bool, optional
        If ``True``, correct the mean critical surface density for photo-z
        biases. Not used if `shear` is ``True``. This should be consistent with
        what is used for calculating the total excess surface density. Default
        is ``False``.
    shear : bool, optional
        If ``True``, return bias of the mean tangential shear. Otherwise,
        return an estimate for the bias of the excess surface density. Default
        is ``False``.

    Returns
    -------
    ds_lm : numpy.ndarray
        The lens magnification bias in each radial bin.

    """
    cosmology = table_l.meta['cosmology']
    bins = table_l.meta['bins']
    comoving = table_l.meta['comoving']
    # Average over bins assuming constant density per area.
    bins = 2.0 / 3.0 * np.diff(bins**3) / np.diff(bins**2)

    z_l = mean_lens_redshift(table_l)
    z_s = mean_source_redshift(table_l)

    try:
        theta = bins.to(u.rad)
    except UnitConversionError:
        theta = (bins / mpc_per_degree(
            z_l, cosmology=cosmology, comoving=comoving)).to(
                u.rad, cu.with_H0(cosmology.H0))

    gt = lens_magnification_shear_bias(
        theta, alpha_l, z_l, z_s, cosmology, sigma_8=sigma_8, n_s=n_s)

    if shear:
        return gt

    return gt * mean_critical_surface_density(
        table_l, photo_z_dilution_correction=photo_z_dilution_correction)


def tangential_shear(table_l, table_r=None, boost_correction=False,
                     scalar_shear_response_correction=False,
                     matrix_shear_response_correction=False,
                     shear_responsivity_correction=False,
                     selection_bias_correction=False,
                     random_subtraction=False, return_table=False):
    """Compute the mean tangential shear with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is ``None``.
    boost_correction : bool, optional
        If ``True``, calculate and apply a boost factor correction. This can
        only be done if a random catalog is provided. Default is ``False``.
    scalar_shear_response_correction : bool, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is ``False``.
    matrix_shear_response_correction : bool, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is ``False``.
    shear_responsivity_correction : bool, optional
        If ``True``, correct for the shear responsivity. Default is ``False``.
    selection_bias_correction : bool, optional
        If ``True``, correct for the multiplicative selection bias in, e.g.,
        HSC. Default is ``False``.
    random_subtraction : bool, optional
        If ``True``, subtract the signal around randoms. This can only be done
        if a random catalog is provided. Default is ``False``.
    return_table : bool, optional
        If ``True``, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final tangential
        shear is returned. Default is ``False``.

    Returns
    -------
    gt : numpy.ndarray or astropy.table.Table
        The tangential shear in each radial bin specified in the precomputation
        phase. If `return_table` is ``True``, will return a table with detailed
        information for each radial bin. The final result is in the column
        `gt`.

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
    result['gt_raw'] = raw_tangential_shear(table_l)
    result['gt'] = raw_tangential_shear(table_l)
    result['z_l'] = mean_lens_redshift(table_l)
    result['z_s'] = mean_source_redshift(table_l)

    if boost_correction:
        if table_r is None:
            msg = ("Cannot compute boost factor correction without results "
                   "from a random catalog.")
            raise ValueError(msg)
        result['b'] = boost_factor(table_l, table_r)
        result['gt'] *= result['b']

    if scalar_shear_response_correction:
        result['1+m'] = 1 + scalar_shear_response_factor(table_l)
        result['gt'] /= result['1+m']

    if matrix_shear_response_correction:
        result['R_t'] = matrix_shear_response_factor(table_l)
        result['gt'] /= result['R_t']

    if shear_responsivity_correction:
        result['2R'] = 2 * shear_responsivity_factor(table_l)
        result['gt'] /= result['2R']

    if selection_bias_correction:
        result['1+m_sel'] = 1 + scalar_shear_response_factor(
            table_l, selection_bias=True)
        result['gt'] /= result['1+m_sel']

    if random_subtraction:
        if table_r is None:
            msg = ("Cannot subtract random results without results from a "
                   "random catalog.")
            raise ValueError(msg)
        result['gt_r'] = tangential_shear(
            table_r, boost_correction=False,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            selection_bias_correction=selection_bias_correction,
            random_subtraction=False, return_table=False)
        result['gt'] -= result['gt_r']

    if not return_table:
        return result['gt'].data

    return result


def excess_surface_density(table_l, table_r=None,
                           photo_z_dilution_correction=False,
                           boost_correction=False,
                           scalar_shear_response_correction=False,
                           matrix_shear_response_correction=False,
                           shear_responsivity_correction=False,
                           selection_bias_correction=False,
                           random_subtraction=False,
                           return_table=False):
    """Compute the mean excess surface density with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is ``None``.
    photo_z_dilution_correction : bool, optional
        If ``True``, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the precomputation phase.
        Default is ``False``.
    boost_correction : bool, optional
        If ``True``, calculate and apply a boost factor correction. This can
        only be done if a random catalog is provided. Default is ``False``.
    scalar_shear_response_correction : bool or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is ``False``.
    matrix_shear_response_correction : bool or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is ``False``.
    shear_responsivity_correction : bool, optional
        If ``True``, correct for the shear responsivity. Default is ``False``.
    selection_bias_correction : bool, optional
        If ``True``, correct for the multiplicative selection bias in, e.g.,
        HSC. Default is ``False``.
    random_subtraction : bool, optional
        If ``True``, subtract the signal around randoms. This can only be done
        if a random catalog is provided. Default is ``False``.
    return_table : bool, optional
        If ``True``, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final excess
        surface density is returned. Default is ``False``.

    Returns
    -------
    ds : numpy.ndarray or astropy.table.Table
        The excess surface density in each radial bin specified in the
        precomputation phase. If `return_table` is ``True``, will return a
        table with detailed information for each radial bin. The final result
        is in the column `ds`.

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
    result['ds_raw'] = raw_excess_surface_density(table_l)
    result['ds'] = raw_excess_surface_density(table_l)
    result['z_l'] = mean_lens_redshift(table_l)
    result['z_s'] = mean_source_redshift(table_l)

    if boost_correction:
        if table_r is None:
            msg = ("Cannot compute boost factor correction without results "
                   "from a random catalog.")
            raise ValueError(msg)
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

    if selection_bias_correction:
        result['1+m_sel'] = 1 + scalar_shear_response_factor(
            table_l, selection_bias=True)
        result['ds'] /= result['1+m_sel']

    if photo_z_dilution_correction:
        result['f_bias'] = photo_z_dilution_factor(table_l)
        result['ds'] *= result['f_bias']

    if random_subtraction:
        if table_r is None:
            msg = ("Cannot subtract random results without results from a "
                   "random catalog.")
            raise ValueError(msg)
        result['ds_r'] = excess_surface_density(
            table_r, photo_z_dilution_correction=photo_z_dilution_correction,
            boost_correction=False,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            selection_bias_correction=selection_bias_correction,
            random_subtraction=False, return_table=False)
        result['ds'] -= result['ds_r']

    if not return_table:
        return result['ds'].quantity

    return result
