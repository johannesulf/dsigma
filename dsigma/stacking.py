import numpy as np
from astropy.table import Table
from . import surveys

__all__ = ['raw_excess_surface_density', 'photo_z_dilution_factor',
           'boost_factor', 'shear_bias_factor',
           'shear_responsivity_factor', 'excess_surface_density']


def raw_excess_surface_density(table_l, rotation=False):
    """Compute the raw, uncorrect excess surface density for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy array
        The raw, uncorrected excess surface density in each radial bin.
    """

    gamma = 'x' if rotation else 't'

    return (np.sum(table_l['sum w_ls e_{} sigma_crit'.format(gamma)] *
                   table_l['w_sys'][:, None], axis=0) /
            np.sum(table_l['sum w_ls'] *
                   table_l['w_sys'][:, None], axis=0))


def photo_z_dilution_factor(table_l):
    """Compute the photometric redshift bias averaged over the entire catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    f : float
        Photometric redshift bias.
    """

    return (np.sum(table_l['calib: sum w_ls w_c'] *
                   table_l['w_sys'][:, None]) /
            np.sum(table_l['calib: sum w_ls w_c sigma_crit_p / sigma_crit_t'] *
                   table_l['w_sys'][:, None]))


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
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['w_sys']) /
        np.sum(table_r['sum w_ls'] * table_r['w_sys'][:, None], axis=0) *
        np.sum(table_r['w_sys']))


def shear_bias_factor(table_l):
    """Compute the multiplicative shear bias.

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
        np.sum(table_l['sum w_ls m'] * table_l['w_sys'][:, None], axis=0) /
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
        np.sum(table_l['sum w_ls (1 - sigma_rms^2)'] *
               table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def excess_surface_density(table_l, table_r=None, rotation=False,
                           photo_z_dilution_correction=False,
                           boost_correction=False,
                           shear_bias_correction=False,
                           shear_responsivity_correction=False,
                           selection_bias_correction=False,
                           random_subtraction=False,
                           return_table=False):
    """Compute the total lensing signal, including all corrections from
    precompute results.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.
    rotation : boolean, optional
        If true, calculate the lensing signal for ellipicities rotated by 45
        deg. Ideally, the result should be consistent with 0.
    photo_z_dilution_correction : boolean, optional
        If true, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the Precomputation phase.
    boost_correction : boolean, optional
        If true, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided.
    shear_bias_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias.
    shear_responsivity_correction : boolean, optional
        If true, correct for the shear responsivity.
    selection_bias_correction : boolean, optional
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

    result['rp_min'] = table_l.meta['rp_bins'][:-1]
    result['rp_max'] = table_l.meta['rp_bins'][1:]
    result['rp'] = np.sqrt(result['rp_min'] * result['rp_max'])
    result['delta sigma_raw'] = raw_excess_surface_density(
        table_l, rotation=rotation)
    result['delta sigma'] = raw_excess_surface_density(
        table_l, rotation=rotation)

    if boost_correction:
        if table_r is None:
            raise Exception('Cannot compute boost factor correction without ' +
                            'results from a random catalog.')
        result['b'] = boost_factor(table_l, table_r)
        result['delta sigma'] *= result['b']

    if shear_bias_correction:
        result['1 + m'] = 1 + shear_bias_factor(table_l)
        result['delta sigma'] /= result['1 + m']

    if shear_responsivity_correction:
        result['2R'] = 2 * shear_responsivity_factor(table_l)
        result['delta sigma'] /= result['2R']

    if selection_bias_correction:
        result['1 + m_sel'] = 1 + surveys.hsc.selection_bias_factor(
            table_l)
        result['delta sigma'] *= result['1 + m_sel']

    if random_subtraction:
        if table_r is None:
            raise Exception('Cannot subtract random results without ' +
                            'results from a random catalog.')
        result['delta sigma_ran'] = excess_surface_density(
            table_r, rotation=rotation,
            photo_z_dilution_correction=False,
            boost_correction=False,
            shear_bias_correction=shear_bias_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            selection_bias_correction=selection_bias_correction,
            random_subtraction=False)
        result['delta sigma'] -= result['delta sigma_ran']

    if photo_z_dilution_correction:
        result['f_bias'] = photo_z_dilution_factor(table_l)
        result['delta sigma'] *= result['f_bias']

    if not return_table:
        return result['delta sigma']

    return result
