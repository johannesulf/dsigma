"""Precompute delta sigma."""
from functools import partial
from multiprocessing import Pool

import numpy as np

import cosmology

from . import data_structure as ds

from .functions import (mpc_per_degree, get_matches,
                        sigma_crit, get_radial_bins,
                        projection_angle)

__all__ = ["lens_source_z_cuts", "core_delta_sigma_function",
           "delta_sigma_single", "delta_sigma_catalog", "get_inv_photoz_bias",
           "result_per_bin", "source_r2_selection", "photo_z_selection",
           "prepare_photoz_calib"]


def prepare_photoz_calib(cfg):
    """Prepare the photometric calibration catalog.

    Parameters
    ----------
    cfg : dict
        Configuration parameters.

    Returns
    -------
    calib_use : numpy array
        Catalog for photo-z calibration sources.
    """
    if cfg['photoz_calib']['calib']:
        calib = np.load(cfg['photoz_calib']['catalog'])
        calib_use, _ = photo_z_selection(calib, cfg)
    else:
        calib_use = None

    return calib_use


def photo_z_selection(sources, cfg):
    """Apply photo-z selection for the source catalog.

    Parameters
    ----------
    sources : numpy array
        Weak lensing source catalog.

    cfg : dict
        Configuration parameters for pre-compute process.
        Should have the `global_photo_z_cuts` and `specinfo_photo_z_cuts` options.

    Returns
    -------
    photoz_mask: boolen array
        Boolen mask for useful sources after the photo-z selection.
    """
    # Apply global photo-z cuts
    # TODO: Right now the column names are fixed
    global_photo_z_cuts = cfg['photoz']['global_photo_z_cuts']

    if global_photo_z_cuts == "basic":
        # Implement ~2-sigma clipping over chi^2_5
        photoz_mask = sources['frankenz_model_llmin'] < 6.
    elif global_photo_z_cuts == "medium":
        photoz_mask = sources['frankenz_model_llmin'] < 6.
        # Remove sources with overly broad PDFs around `z_best`
        photoz_mask *= sources['frankenz_photoz_risk_best'] < 0.25
    elif global_photo_z_cuts == "strict":
        # Similar to `medium`, but stricter
        photoz_mask = sources['frankenz_model_llmin'] < 6.
        photoz_mask *= sources['frankenz_photoz_risk_best'] < 0.15
    elif global_photo_z_cuts == "none":
        # No global photo-z cut is applied
        photoz_mask = np.isfinite(sources[cfg['photoz']['field']])
    else:
        raise Exception("Invalid global photo-z cuts option "
                        "= {}".format(global_photo_z_cuts))

    # Apply specinfo photo-z cuts
    # Should most likely be applied with corresponding redshift cuts
    specinfo_photo_z_cuts = cfg["photoz"]['specinfo_photo_z_cuts']

    if specinfo_photo_z_cuts == "none":
        # No cut
        pass
    elif specinfo_photo_z_cuts == "great":
        # >50% of info comes from non-photo-z sources.
        photoz_mask *= sources['frankenz_model_ptype2'] < 0.5
    elif specinfo_photo_z_cuts == "good":
        # >10% of info comes from non-photo-z sources.
        photoz_mask *= sources['frankenz_model_ptype2'] < 0.9
    elif specinfo_photo_z_cuts == "moderate":
        # 10%-50% of info comes from non-photo-z sources.
        photoz_mask *= ((sources['frankenz_model_ptype2'] >= 0.5) &
                        (sources['frankenz_model_ptype2'] < 0.9))
    elif specinfo_photo_z_cuts == "poor":
        # <=50% of info comes from non-photo-z sources.
        photoz_mask = sources['frankenz_model_ptype2'] >= 0.5
    elif specinfo_photo_z_cuts == "poorest":
        # <=10% of info comes from non-photo-z sources.
        photoz_mask = sources['frankenz_model_ptype2'] >= 0.9
    else:
        raise Exception("Invalid specinfo photo-z cuts option "
                        "= {}".format(global_photo_z_cuts))

    return sources[photoz_mask], photoz_mask


def get_inv_photoz_bias(z_l, z_s_true, z_s_phot, cosmo, z_weights=1.0,
                        comoving=False):
    """Calculate the inverse of the photo-z bias factor.

    Parameters
    ----------
    z_l : float
        Redshift of the lens.
    z_s_true : float, or numpy array
        The "true" redshift of sources
    z_s_phot : float, or numpy array
        The photometric redshift used for the same source.
    z_weights : float, or numpy array
        Redshift weight.
    cosmos : cosmology.Cosmology object
        Cosmology object from `cosmology` package by Erin Sheldon.
    comoving : boolen, optional
        Flag for using comoving instead of physical unit. Default: False

    Return
    ------
        The inverse of the photo-z bias factor, the `f_bias^{-1}`.

    Notes
    -----
        The definition of f_bias is that: DS_True = DS_Phot * zp_bias
        And:
            zp_bias_num = Sum_{i=1}^{N_s}(weight_s)

            zp_bias_den = Sum_{i=1}^{N_s}(weight_s * sigma_crit_ratio)

        Where, sigma_crit_ratio:

            sigma_crit_ratio = sigma_crit_phot_z / sigma_crit_true_z

    """
    # The denominator
    zp_bias_den = np.sum(
        z_weights *
        sigma_crit(z_l, z_s_phot, cosmo, comoving=comoving) /
        sigma_crit(z_l, z_s_true, cosmo, comoving=comoving), axis=0
        )

    # The numerator
    zp_bias_num = np.sum(z_weights, axis=0)

    return zp_bias_num, zp_bias_den


def source_r2_selection(r2, weight, n_hist=100, check=False, error=False):
    """Apply R2 selection bias correction.

    Calculate edge-selection bias due to resolution cut at R2=0.3 for a
    given dataset, such as a tomographic redshift bin.

    Based on:
        https://github.com/PrincetonUniversity/HSC-WeakLens/blob/master/shape_catalog/selbias/R2_selection_bias.py

    Parameters
    ----------
    r2 : numpy array
        Shape resolution.
    weight : numpy array
        Shape weight.
    n_hist : int, optional
        Number of bins in histogram

    Return
    ------
    """
    if len(r2) <= 1:
        # If not source in the array, just return 0.0
        if error:
            return 0.0, 1.0, np.nan
        return 0.0, 1.0

    if check:
        min_r2, max_r2 = np.min(r2), np.max(r2)

        # No need to check these for HSC shape catalog
        if min_r2 < 0.3:
            raise ValueError("Found R2 value below cutoff: {0}".format(min_r2))
        if max_r2 > 1.0:
            raise ValueError("Found R2 value above cutoff: {0}".format(max_r2))

        if not np.all(np.isfinite(weight)):
            raise ValueError("Non-finite weight")

        if not np.all(weight) >= 0.0:
            raise ValueError("Negative weight")

    # 0.7 because the range of r2 should be between 0.3 and 1.0
    bin_size = 0.7 / n_hist

    # Numerator
    p_edge_num = np.sum(weight[r2 <= 0.3 + bin_size])

    # Denominator
    p_edge_den = (np.sum(weight) * bin_size)

    # See GREAT3-like sims paper
    A, A_err = 0.00865, 0.0026

    if error:
        return (A * p_edge_num), p_edge_den, (A_err * p_edge_num)

    return (A * p_edge_num), p_edge_den


def lens_source_z_cuts(lens, sources, cfg):
    """Remove sources that are in front of the lens.

    Parameters
    ----------
    lens : numpy array
        Information about a single lens.
    sources : numpy array
        Catalog of weak lensing sources.
    cfg : dict
        configuration parameters.

    Return
    ------
        Catalog of useful sources.

    Notes
    -----
    In the configuration file, the user define the column names
    for photometric redshift and error of photo-z.
    User has access to them here using:
        - `sf['z']`: photometric redshift.
        - `sf['z_low']`: lower confident bound of the redshift distribution.

    Photo-z separation cuts:
        zs > zl + 0.1 AND zs_lower_68_bound > zl
    """
    photoz_sep = float(cfg['photoz']["lens_source_z_separation"])

    # Useful fields from the source and lens catalog
    sf, lf = cfg['source_fields'], cfg['lens_fields']

    if photoz_sep >= 0.:
        # Use additional z_err selection when True.
        if cfg['photoz']['z_err_cut']:
            # Make sure that the z_low is the lower bound of the photo-z distribution
            photoz_mask = ((sources[sf['z_low']] > lens[lf['z']]) &
                           (sources[sf['z']] > (lens[lf['z']] + photoz_sep)))
        else:
            # Only use z_separation cut.
            photoz_mask = (sources[sf['z']] >= (lens[lf['z']] + photoz_sep))
    else:
        raise Exception("Invalid lens_source_z_separation "
                        "= {}".format(photoz_sep))
    return sources[photoz_mask]


def result_per_bin(results, matches, per_pair_results, idx, n_hist=100):
    """Gather the result for each radial bin.

    Parameters
    ----------
    results : numpy array
        Structured numpy array for output.
    matches : numpy array
        Cross-match results from `smatch`.
    per_pair_results : numpy array
        Information for each lens-source pair.
    idx : int
        Index of the radial bin.
    n_hist : int, optional
        Numpy of histogram used to calculate resolution selection bias.
        Default: 100

    Returns
    -------

    Notes
    -----
    * For the `sum_num_t` and `sum_num_x`, remember:

        ds_weight =  e_weight * inverse_sigma_crit ** 2

    So for `sum_num_t` and `sum_num_x`, one `inverse_sigma_crit` get cancelled out.

    * Compute sum of k using the same weight for delta_sigma:

        k = sum(ds_weight * bias_m_srcs) / sum(ds_weight)

        -  Note: we don't include the 1 here so that has to included later, to form (1+m)

    * For the responsivity:

        R = sum(ds_weight * (1 - e_rms ** 2.0)) / sum(ds_weight)

        - Also using the same weight

    * For the square term of the shape noise, see:

        https://github.com/msimet/GGLensing/blob/master/deltasigma.py
    """
    # Get the results for lens-sources pairs in each radial bin
    mask_each_bin = per_pair_results['rad_bin'] == idx

    per_pair_each_bin = per_pair_results[mask_each_bin]

    result_bin = results['radial_bins'][idx]

    # Compute the numerator.
    result_bin['sum_num_t'] = (np.sum(per_pair_each_bin['e_t'] *
                                      per_pair_each_bin['e_weight'] *
                                      per_pair_each_bin['inverse_sigma_crit']))
    result_bin['sum_num_x'] = (np.sum(per_pair_each_bin['e_x'] *
                                      per_pair_each_bin['e_weight'] *
                                      per_pair_each_bin['inverse_sigma_crit']))
    result_bin['sum_den'] = np.sum(per_pair_each_bin['ds_weight'])

    # This is used to calculate the naive error for delta_sigma
    result_bin['sum_num_t_sq'] = (np.sum((per_pair_each_bin['e_t'] *
                                          per_pair_each_bin['e_weight'] *
                                          per_pair_each_bin['inverse_sigma_crit']) ** 2))
    result_bin['sum_num_x_sq'] = (np.sum((per_pair_each_bin['e_x'] *
                                          per_pair_each_bin['e_weight'] *
                                          per_pair_each_bin['inverse_sigma_crit']) ** 2))

    # Resolution selection bias
    if len(per_pair_each_bin['r2']) > 1:
        result_bin['sum_num_m_sel'], result_bin['sum_den_m_sel'] = source_r2_selection(
            per_pair_each_bin['r2'], per_pair_each_bin['e_weight'],
            n_hist=n_hist, check=False, error=False)
    else:
        result_bin['sum_num_m_sel'], result_bin['sum_den_m_sel'] = 0.0, 1.0

    # Multiplicative bias (k)
    result_bin['sum_num_k'] = np.sum(per_pair_each_bin['bias_m_srcs'] *
                                     per_pair_each_bin['ds_weight'])

    # Responsivity (R)
    result_bin['sum_num_r'] = np.sum((1 - per_pair_each_bin['e_rms'] ** 2) *
                                     per_pair_each_bin['ds_weight'])

    # Square term about the shape noise.
    result_bin['ds_t_sq'] = np.sum(per_pair_each_bin['e_t'] ** 2 *
                                   per_pair_each_bin['e_weight'])
    result_bin['ds_x_sq'] = np.sum(per_pair_each_bin['e_x'] ** 2 *
                                   per_pair_each_bin['e_weight'])

    # Just for test
    result_bin['e_weight'] = np.nansum(per_pair_each_bin['e_weight'])

    # Number of pairs in each radial bin
    result_bin['n_pairs'] = len(per_pair_each_bin['e_t'])

    # Calculate the sum of lens-source distance in each bin
    result_bin['sum_dist'] = np.sum(matches[mask_each_bin]['dist'])

    # return result_bin
    return


def core_delta_sigma_function(lens, sources, cfg, cosmo, matches, calib=None):
    """Compare delta sigma for each lens-source pair.

    It deals with a single lens galaxy, and all the sources galaxies that
    are matched within a maximum radius required by the user.

    Here, the e1 and e2 measurements are first rotated into g-g lensing frame
    for each lens-source pair.
    Then we group the per-pair results into radial bins.

    Parameters
    ----------
    lens : numpy array
        Information about single lens.
    sources : numpy array
        Catalog for sources.
    cfg : dict
        Configuration parameters.
    cosmo : cosmology.Cosmo
        Cosmology object from `cosmology` object by Erin Sheldon.
    matches : numpy array
        Cross-match results from `smatch` by Erin Sheldon
    calib : numpy array, optional
        Additional photometric redshift calibration catalog. Default: None

    Return
    ------
    results : numpy array
        Pre-compute results for a single lens.

    """
    # Useful fields from the source and lens catalog
    sf, lf = cfg['source_fields'], cfg['lens_fields']

    # Results for each lens-source pair
    per_pair_results = np.zeros(len(sources), dtype=[
        ("e_t", "float64"),
        ("e_x", "float64"),
        ("r2", "float64"),
        ("e_weight", "float64"),
        ("e_rms", "float64"),
        ("ds_weight", "float64"),
        ("bias_m_srcs", "float64"),
        ("inverse_sigma_crit", "float64"),
        ("rad_bin", "int"),
        ])

    # Estimate the inverse of Sigma_crit
    per_pair_results['inverse_sigma_crit'] = (
        1.0 / sigma_crit(
            np.full(len(sources[sf['z']]), lens[lf['z']]), sources[sf['z']],
            cosmo, comoving=cfg['comoving']
            )
        )

    """
    Get the projection angle between each lens-source pair, used to
    project e1 and e2 into the proper reference frame to get the
    tangential and cross shear

    Assumed coordinate frame for source catalog is (X,Y)=(RA, DEC)
    """
    cos2phi, sin2phi = projection_angle(lens[lf['ra']], lens[lf['dec']],
                                        sources[sf['ra']],
                                        sources[sf['dec']])

    # Get the tangential and cross shear term
    per_pair_results['e_t'] = (sources[sf['e1']] * cos2phi +
                               sources[sf['e2']] * sin2phi) * -1.0
    per_pair_results['e_x'] = (sources[sf['e1']] * sin2phi -
                               sources[sf['e2']] * cos2phi) * 1.0

    """
    The weight from the HSC catalog includes the intrinsic dispersion
    of galaxy shape and the shape measurement error.

    ```
    e_weight = (1 / np.sqrt(e_rms ** 2 + sigma_e ** 2))
    ```
    """
    per_pair_results['e_weight'] = sources[sf['weight']]

    # Resolution of the source
    per_pair_results['r2'] = sources[sf['r2']]

    # RMS of the ellipticity. Used for responsivity R
    per_pair_results['e_rms'] = sources[sf['e_rms']]

    """
    # Weight used to calculate delta_sigma

    ```
    ds_weight = e_weight * (inverse_sigma_crit ** 2)
    ```
    """
    per_pair_results['ds_weight'] = (
        per_pair_results['e_weight'] *
        per_pair_results['inverse_sigma_crit'] ** 2)

    # Multiplicative bias
    per_pair_results['bias_m_srcs'] = sources[sf['bias_m']]

    # The radial bin that this pair belongs to
    per_pair_results['rad_bin'] = get_radial_bins(
        matches['dist'], nbins=cfg['binning']['nbins'],
        rmin=cfg['binning']['rmin'], rmax=cfg['binning']['rmax'])

    # Group the per pair results into radial bins.
    results = ds.get_results_arr(cfg)

    # Pass basic information about the lenses to the result array
    results['field'] = lens[lf['field']]
    results['ra'] = lens[lf['ra']]
    results['dec'] = lens[lf['dec']]
    results['z'] = lens[lf['z']]

    # Check the weight of lenses
    try:
        results['weight'] = lens[lf['weight']]
    except KeyError:
        # If there is no `weight`, add a constant one: weight = 1.0
        results['weight'] = 1.0

    # If necessary, estimates the photo-z bias factor
    if calib is None:
        results['zp_bias_num'], results['zp_bias_den'] = 1.0, 1.0
    else:
        # Check to see if redshift weight should be applied
        if cfg['photoz_calib']['z_weight'] is not None:
            z_weights = calib[cfg['photoz_calib']['z_weight']]
        else:
            z_weights = np.full(len(calib[sf['z']]), 1.0)
        # Get the numerator and denominator for the zp_bias factor
        results['zp_bias_num'], results['zp_bias_den'] = get_inv_photoz_bias(
            np.full(len(calib[sf['z']]), lens[lf['z']]),
            calib[cfg['photoz_calib']['z_true']],
            calib[sf['z']], cosmo, z_weights=z_weights,
            comoving=cfg['comoving'])

    # Gather the result for each radius bin into the array
    _ = [result_per_bin(results, matches, per_pair_results, idx,
                        n_hist=cfg['selection']['n_hist'])
         for idx in range(len(results['radial_bins']))]

    return results


def delta_sigma_single(lens, sources, cfg, calib=None):
    """Delta sigma for a single lens and all the sources.

    See delta_sigma_catalog for the reason that the sources are a
    global rather than an argument.

    Parameters
    ----------
    lens : numpy array
        Information about a single lens.
    sources : numpy array
        Catalog of weak lensing sources.
    cfg : dict
        configuration parameters.
    calib : numpy array, optional
        Additional photometric redshift calibration catalog. Default: None

    Return
    ------
        Precompute result for single lens.
    """
    # Get the useful fields in the source and lens catalog.
    sf, lf = cfg['source_fields'], cfg['lens_fields']

    # Apply redshift cuts
    sources_use = lens_source_z_cuts(lens, sources, cfg)

    # Apply the same photo-z cuts to the calibration catalog
    calib_use = lens_source_z_cuts(lens, calib, cfg) if calib is not None else None

    # The cosmology object
    cosmo = cosmology.Cosmo(**cfg['cosmology'])

    # Physical scale in unit of Mpc/degree
    mpc_deg = mpc_per_degree(cosmo, lens[lf['z']], comoving=cfg['comoving'])

    # Find sources in desired angular range
    matches = get_matches(
        lens[lf['ra']], lens[lf['dec']],
        sources_use[sf['ra']], sources_use[sf['dec']],
        mpc_deg=mpc_deg, scale='physical',
        rmin=cfg['binning']['rmin'],
        rmax=cfg['binning']['rmax'])

    # mask to only matched_sources
    sources_match = sources_use[matches['i2']]

    return core_delta_sigma_function(lens, sources_match, cfg, cosmo,
                                     matches, calib=calib_use)


def delta_sigma_catalog(lenses, sources, cfg, calib=None, n_jobs=1):
    """For each lens, calculate delta sigma for all sources.

    Parameters
    ----------
    lenses : numpy array
        Catalog of lenses
    sources : numpy array
        Catalog of sources
    cfg : dict
        Configuration parameters
    n_jobs : int, optional
        Number of jobs to run at the same time. Default: 1
    calib : numpy array, optional
        Additional photometric redshift calibration catalog. Default: None

    Return
    ------
    results : numpy array
        Results of pre-computation.
    lenses_new : numpy array
        New lens catalog sorted by field.
    """
    # Prepare the outputs
    results = []
    lenses_new = np.array([], dtype=lenses.dtype)

    # Get the useful fields in the source and lens catalog.
    sf, lf = cfg['source_fields'], cfg['lens_fields']

    field_list = np.unique(lenses[lf['field']])

    for ii, field in enumerate(field_list):
        """
        The most expensive part of this pipeline is copying/cutting data.

        Therefore we:
        1. Do each field separately so that we can skip the separation cuts.
        2. Save the sources array as a global so that we don't duplicate that
           for each pool call """
        print("\n# Deal with field %3d/%3d - ID: %d" % (ii + 1, len(field_list), field))

        # Lenses in this field
        lenses_field = lenses[lenses[lf['field']] == field]
        lenses_new = np.append(lenses_new, np.array(lenses_field, dtype=lenses.dtype))

        # Sources in this field
        sources_field = np.sort(sources[sources[sf['field']] == field], order=sf['z'])

        print("  Field {0}: {1} lens galaxies and {2} source galaxies"
              .format(field, len(lenses_field), len(sources_field)))

        # Freeze the source catalog and configuration parameters
        delta_sigma_single_map = partial(delta_sigma_single,
                                         sources=sources_field, calib=calib, cfg=cfg)

        if n_jobs != 1:
            with Pool(processes=n_jobs) as p:
                results.extend(p.map(delta_sigma_single_map, lenses_field))
        else:
            results.extend(list(map(delta_sigma_single_map, lenses_field)))

    return results, lenses_new
