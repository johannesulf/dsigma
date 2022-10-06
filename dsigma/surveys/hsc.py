"""Module with functions specific to the Hyper-Suprime Cam survey."""

import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'apply_photo_z_quality_cut',
           'selection_bias_factor']

default_version = 'PDR2'
known_versions = ['PDR2', ]
e_2_convention = 'flipped'


def default_column_keys(version=default_version):
    """Return a dictionary of default column keys.

    Parameters
    ----------
    version : string or None, optional
        Version of the catalog.

    Returns
    -------
    keys : dict
        Dictionary of default column keys.

    Raises
    ------
    ValueError
        If `version` does not correspond to a known catalog version.

    """
    if version == 'PDR2':
        keys = {
            'ra': 'ira',
            'dec': 'idec',
            'z': 'photoz_best',
            'z_low': 'photoz_err68_min',
            'e_1': 'ishape_hsm_regauss_e1',
            'e_2': 'ishape_hsm_regauss_e2',
            'w': 'ishape_hsm_regauss_derived_shape_weight',
            'm': 'ishape_hsm_regauss_derived_shear_bias_m',
            'e_rms': 'ishape_hsm_regauss_derived_rms_e',
            'R_2': 'ishape_hsm_regauss_resolution'}
    else:
        raise ValueError(
            "Unkown version of DES. Supported versions are {}.".format(
                known_versions))

    return keys


def apply_photo_z_quality_cut(table_s, global_photo_z_cuts,
                              specinfo_photo_z_cuts):
    """Apply HSC-specific photo-z cuts to the source catalog.

    Parameters
    ----------
    table_s : astropy.table.Table
        HSC weak lensing source catalog.
    global_photo_z_cuts : string
        Requirements for global photometric redshift quality.
    specinfo_photo_z_cuts : string
        Specinfo photo-z cuts.

    Returns
    -------
    table_s : astropy.table.Table
        Table containing only source passing the cuts.

    Raises
    ------
    ValueError
        If invalid options are passed.

    """
    if global_photo_z_cuts == "basic":
        # Implement ~2-sigma clipping over chi^2_5.
        mask = table_s['frankenz_model_llmin'] < 6.
    elif global_photo_z_cuts == "medium":
        mask = table_s['frankenz_model_llmin'] < 6.
        # Remove sources with overly broad PDFs around `z_best`.
        mask = mask & (table_s['frankenz_photoz_risk_best'] < 0.25)
    elif global_photo_z_cuts == "strict":
        # Similar to `medium`, but stricter.
        mask = table_s['frankenz_model_llmin'] < 6.
        mask = mask & (table_s['frankenz_photoz_risk_best'] < 0.15)
    elif global_photo_z_cuts == "none":
        # No global photo-z cut is applied.
        mask = np.isfinite(table_s['frankenz_photoz_best'])
    else:
        raise ValueError("Invalid global photo-z cuts option "
                         "{}".format(global_photo_z_cuts))

    # Apply specinfo photo-z cuts
    # Should most likely be applied with corresponding redshift cuts.
    if specinfo_photo_z_cuts == "none":
        # No cut
        pass
    elif specinfo_photo_z_cuts == "great":
        # >50% of info comes from non-photo-z sources.
        mask = mask & (table_s['frankenz_model_ptype2'] < 0.5)
    elif specinfo_photo_z_cuts == "good":
        # >10% of info comes from non-photo-z sources.
        mask = mask & (table_s['frankenz_model_ptype2'] < 0.9)
    elif specinfo_photo_z_cuts == "moderate":
        # 10%-50% of info comes from non-photo-z sources.
        mask = mask & (table_s['frankenz_model_ptype2'] >= 0.5)
        mask = mask & (table_s['frankenz_model_ptype2'] < 0.9)
    elif specinfo_photo_z_cuts == "poor":
        # <=50% of info comes from non-photo-z sources.
        mask = mask & (table_s['frankenz_model_ptype2'] >= 0.5)
    elif specinfo_photo_z_cuts == "poorest":
        # <=10% of info comes from non-photo-z sources.
        mask = mask & (table_s['frankenz_model_ptype2'] >= 0.9)
    else:
        raise ValueError("Invalid specinfo photo-z cuts option "
                         "= {}".format(global_photo_z_cuts))

    return table_s[mask]


def selection_bias_factor(table_l):
    """Compute the multiplicative selection bias.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    m_sel : numpy.ndarray
        Multiplicative selection bias in each radial bin.

    """
    return (
        np.sum(table_l['sum w_ls A p(R_2=0.3)'] *
               table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))
