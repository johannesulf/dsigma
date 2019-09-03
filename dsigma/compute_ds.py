"""Compute delta sigma profile."""
from functools import partial

import numpy as np

try:
    from joblib import Parallel, delayed
    joblib_available = True
except ImportError:
    joblib_available = False

from . import plots
from . import maskgen
from . import jackknife as jk

__all__ = ['assert_lens_rand_consistent',
           'assert_precompute_catalog_consistent', 'load_data_delta_sigma',
           'reweight_rand_photoz', 'mask_lens_delta_sigma',
           'get_lens_weight', 'get_r_factor', 'get_zp_bias',
           'get_k_factor', 'get_calibration_factor',
           'get_delta_sigma_ratio', 'get_boost_factor',
           'get_delta_sigma_lens_rand', 'get_delta_sigma_npairs',
           'get_delta_sigma_error_simple', 'get_delta_sigma_error_gglens',
           'get_delta_sigma_error_jackknife', 'get_delta_sigma_errors',
           'qa_delta_sigma', 'prepare_compute_ds', 'get_ds_ratio_two_weights',
           'prepare_compute_ds_second', 'get_delta_sigma_ratio_jackknife',
           'effective_redshift']


def prepare_compute_ds(cfg, lens_key='ds_lenses', rand_key='ds_randoms',
                       ext_mask='external_mask', arg_mask='mask_args',
                       lens_cat='lens_catalog', rand_cat='rand_catalog'):
    """Prepare the data for computing delta_sigma.

    * Make sure that the pre-compute results for random and lenses
      share the same cosmology and radial bins.
    * Assign jackknife field if necessary.

    Parameters
    ----------
    cfg : dict
        Configuration parameters for computeDS.
    lens_key : str, optional
        The key in the configuration file for pre-compute
        results for lenses.
        Default: 'ds_lenses'
    rand_key : str, optional
        The key in the configuration file for pre-compute
        results for randoms. (return None if random is not available)
        Default: 'ds_randoms'
    external_mask: str, optional
        Key name for the external mask array.  Default: 'external_mask'
    arg_mask: str, optional
        Key name for lens mask argument.  Default: 'mask_args'
    lens_cat : str, optional
        Key for the lens catalog. Default: "lens_catalog".
    rand_cat : str, optional
        Key for the random catalog. Default: "rand_catalog".

    Returns
    -------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Catalog of lenses.
    rand_data : numpy array
        Catalog of random objects.
    radial_bins : numpy array
        1-D array for radial bins.
    cosmology : cosmology.Cosmo
        Cosmology object for Erin Sheldon's `cosmology` package.
    """
    # Load the data: now need to load lens catalog, pre-compute
    # results for lens and random (if necessary)
    lens_ds, rand_ds, lens_data, rand_data = load_data_delta_sigma(
        cfg, lens_key=lens_key, rand_key=rand_key,
        lens_cat=lens_cat, rand_cat=rand_cat)

    # Make sure the pre-compute results and the lens catalog are consistent
    assert_precompute_catalog_consistent(lens_ds, lens_data)
    # Make sure the pre-compute results and the random catalog are consistent
    if rand_ds is not None:
        assert_precompute_catalog_consistent(rand_ds, rand_data)

    # Get the cosmology
    cosmology = lens_ds['cosmology']

    # Get the radial bins using bin center
    radial_bins = lens_ds['radial_bins']

    if rand_ds:
        assert_lens_rand_consistent(lens_ds, rand_ds)
        lens_ds, rand_ds = lens_ds["delta_sigma"], rand_ds["delta_sigma"]
    else:
        lens_ds, rand_ds = lens_ds["delta_sigma"], None

    # Show the effective redshift of the lens
    lens_z_eff = effective_redshift(lens_ds)
    print("# The effective redshift of the lenses: %8.6f" % lens_z_eff)

    # Show the effective redshift of the rando (if available)
    if rand_ds is not None:
        rand_z_eff = effective_redshift(rand_ds)
        print("# The effective redshift of the randoms: %8.6f" % rand_z_eff)

    # Assign the jackknife regions if necessary
    # TODO: Need to have a threshold: when N_lens < N_threshold
    # stop doing Jackknife resampling, using bootstrap instead
    if rand_ds is not None:
        lens_ds, rand_ds = jk.add_jackknife_both(lens_ds, rand_ds,
                                                 cfg['njackknife_fields'])
    else:
        lens_ds = jk.add_jackknife_field(lens_ds, cfg['njackknife_fields'])

    # Select a subsample of lenses if necessary
    lens_ds_use, lens_data_use = mask_lens_delta_sigma(
        lens_ds, lens_data, cfg, ext_mask=ext_mask, arg_mask=arg_mask)

    return lens_ds_use, rand_ds, lens_data_use, rand_data, radial_bins, cosmology


def prepare_compute_ds_second(cfg, lens_ds, rand_ds, radial_bins, cosmology):
    """Prepare the second dataset for computeDS.

    Parameters
    ----------
    cfg : dict
        Configuration parameters for computeDS.
    lens_ds : numpy array
        Pre-compute results for selected lenses.
    rand_ds : numpy array
        Pre-compute results for random objects.
    radial_bins : numpy array
        1-D array for radial bins.
    cosmology : cosmology.Cosmo
        Cosmology object for Erin Sheldon's `cosmology`.

    Returns
    -------
    """
    # If necessary, prepare the second pre-compute lensing results
    if cfg["ds_lenses_2"] is not None:
        # Assuming the radial bins and the cosmology should be the same
        (lens_ds_2, rand_ds_2, lens_data_2, rand_data_2,
         radial_bins_2, cosmology_2) = prepare_compute_ds(
             cfg, lens_key='ds_lenses_2', rand_key='ds_randoms_2',
             lens_cat='lens_catalog_2', rand_cat='rand_catalog_2',
             ext_mask='external_mask_2', arg_mask='mask_args_2')

        # Make sure the two pre-compute results share the same radial bins
        if not np.all(cosmology == cosmology_2):
            print("Different cosmology for the two pre-compute results !")

        if not np.all(radial_bins == radial_bins_2):
            raise ValueError("Different radial bins for the two "
                             "pre-compute results")

        # Show the effective redshift of the lens
        lens_z_eff = effective_redshift(lens_ds_2)
        print("# The effective redshift of the lenses: %8.6f" % lens_z_eff)

        # Show the effective redshift of the rando (if available)
        if rand_ds is not None:
            rand_z_eff = effective_redshift(rand_ds_2)
            print("# The effective redshift of the randoms: %8.6f" % rand_z_eff)

        # Make sure these two results share the same jackknife fields
        if (rand_ds is None) and (rand_ds_2 is None):
            lens_ds, lens_ds_2 = jk.add_jackknife_both(
                lens_ds, lens_ds_2, cfg['njackknife_fields'])
        elif (rand_ds is not None) and (rand_ds_2 is not None):
            (lens_ds, rand_ds,
             lens_ds_2, rand_ds_2) = jk.add_jackknife_both(
                 lens_ds, rand_ds, cfg['njackknife_fields'],
                 lens_ds_2=lens_ds_2, rand_ds_2=rand_ds_2)
        else:
            # TODO: Now we only allow both using or not using random catalogs
            raise Exception("Two pre-compute results should " +
                            "have consistent options for random subtraction")

        return lens_ds_2, rand_ds_2, lens_data_2, rand_data_2

    return None, None, None, None


def assert_precompute_catalog_consistent(lens_ds, lens_data):
    """Make sure the pre-compute results for lens are consistent with the
       lens catalog.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses
    lens_data : numpy array
        Lens catalog
    """
    # TODO: This assumes 'ra', 'dec' as column names.
    if not np.all(lens_ds['delta_sigma']['ra'] == lens_data['ra']):
        raise ValueError("Pre-compute results for lens and lens catalog " +
                         "are not consistent !")

    if not np.all(lens_ds['delta_sigma']['dec'] == lens_data['dec']):
        raise ValueError("Pre-compute results for lens and lens catalog " +
                         "are not consistent !")


def assert_lens_rand_consistent(lenses, randoms):
    """Consistency check.

    Make sure the lenses and randoms share the same cosmology
    and radial_bins.

    Parameters
    ----------
    lenses : numpy array
        Pre-compute results for lenses
    randoms : numpy array
        Pre-compute results for randoms
    """
    # Make sure we are comparing individual cosmological parameters
    try:
        lens_cosmo = lenses['cosmology'].item()
        rand_cosmo = randoms['cosmology'].item()
    except ValueError:
        lens_cosmo = lenses['cosmology']
        rand_cosmo = randoms['cosmology']

    if not np.all(lens_cosmo == rand_cosmo):
        raise ValueError("Lenses and randoms computed with " +
                         "different cosmology")

    if not np.all(lenses["radial_bins"] == randoms["radial_bins"]):
        raise ValueError("Lenses and randoms computed with " +
                         "different radial bins")


def load_data_delta_sigma(cfg, lens_key='ds_lenses', rand_key='ds_randoms',
                          lens_cat='lens_catalog', rand_cat='rand_catalog'):
    """Load the data for computing delta_sigma.

    Parameters
    ----------
    cfg : dict
        Configuration file
    lens_key : str, optional
        The key in the configuration file for pre-compute results for lenses.
        Default: lens_ds
    rand_key : str, optional
        The key in the configuration file for pre-compute results for randoms.
        (return None if random is not available)
        Default: rand_ds
    lens_cat : str, optional
        Key for the lens catalog. Default: "lens_catalog".
    rand_cat : str, optional
        Key for the random catalog. Default: "rand_catalog".

    Returns
    -------
    lens_ds : numpy array
        Pre-compute results for selected lenses.
    rand_ds : numpy array
        Pre-compute results for random objects.
    lens_data : numpy array
        Catalog of selected lenses.
    rand_data : numpy array
        Catalog of random objects.
    """
    lens_ds = np.load(cfg[lens_key], allow_pickle=True)
    lens_data = np.load(cfg[lens_cat], allow_pickle=True)

    if rand_key in cfg.keys():
        rand_ds = np.load(cfg[rand_key], allow_pickle=True)
        rand_data = np.load(cfg[rand_cat], allow_pickle=True)
    else:
        rand_ds, rand_data = None, None

    return lens_ds, rand_ds, lens_data, rand_data


def reweight_rand_photoz(lens_z, rand_z, nbins=10, qa=False, prefix=None):
    """Re-weight the photo-z for randoms.

    Re-weight the randoms based on the photometric redshift
    distributions for lenses and randoms.

    Parameters
    ----------
    lens_z : numpy array
        Redshift of selected lenses.
    rand_z : numpy array
        Redshift of random objects.
    nbins : int, optional
        Number of bins used in histogram.
    qa : boolen, optional
        Make QA plots about the reweight. Default: False
    prefix : str, optional
        Prefix for the QA plots. Default: None

    Return
    ------
    rand_zweight : numpy array
        Redshift weights for randoms.
    """
    # Get the redshift ranges of both lenses and randoms
    z_both = np.concatenate([lens_z, rand_z])
    z_min, z_max = z_both.min(), z_both.max()
    z_bins = np.linspace(z_min, z_max, nbins + 1)

    # Create the histograms for both samples in the same redshift range
    lens_zhist, _ = np.histogram(
        lens_z, bins=nbins, range=(z_min, z_max), density=True)
    rand_zhist, _ = np.histogram(
        rand_z, bins=nbins, range=(z_min, z_max), density=True)

    # Calculate the redshift weights in each redshift bins
    z_weights = np.nan_to_num((lens_zhist / rand_zhist), 0.0)

    # Assign redshift weights to randoms
    rand_zidx = np.digitize(rand_z, z_bins, right=False)
    rand_zweight = np.zeros(len(rand_z))

    for ii, ww in enumerate(z_weights):
        rand_zweight[rand_zidx == (ii + 1)] = ww

    if qa:
        # Get the redshift bin centers
        z_center = (z_bins[:-1] + z_bins[1:]) / 2

        # This QA figures are meant for debug
        plots.qa_rand_photoz_weight(z_center, z_weights,
                                    qa_prefix=prefix)

        plots.qa_lens_rand_zhist(lens_z, rand_z, rand_zweight,
                                 qa_prefix=prefix)

    return rand_zweight


def mask_lens_delta_sigma(lens_ds, lens_data, cfg,
                          ext_mask='external_mask', arg_mask='mask_args'):
    """Mask for lenses and randoms.

    Using the user defined mask to select subsample of lenses.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    lens_data : numpy array
        Catalog of lenses.
    cfg : dict
        Configuration parameters for computeDS.
    ext_mask : str, optional
        Key name of external mask file in the configuration dict.
        Default: 'external_mask'
    arg_mask : str, optional
        Key name of the mask argument in the configuration dict.
        Default: 'mask_args'

    Return
    ------
    lens_ds : numpy array
        Pre-compute results for selected lenses.
    lens_data : numpy array
        Catalog of selected lenses.
    """
    # Creat an empty lens mask
    lens_mask = np.full(len(lens_data), True)

    if ext_mask in cfg:
        if (cfg[ext_mask] != 'None') and (cfg[ext_mask] != ''):
            ext_mask = np.load(cfg[ext_mask], allow_pickle=True)
            assert len(ext_mask) == len(lens_data)

            lens_mask = (lens_mask & np.asarray(ext_mask, dtype=bool))

            print("\n# Sample: %s provides %d galaxies" % (cfg[ext_mask],
                                                           sum(lens_mask)))

    if (arg_mask in cfg) and (cfg[arg_mask] is not ''):
        lens_mask = (lens_mask & maskgen.compute_mask(
            lens_data, cfg[arg_mask]))
        print("\n# Sample: %s provides %d galaxies" % (cfg[arg_mask],
                                                       sum(lens_mask)))

    return lens_ds[lens_mask], lens_data[lens_mask]


def get_zp_bias(lens_ds, weights=1.0):
    """Estimate the photometric redshift bias using the pre-compute results.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses (or randoms).
    weights : float or numpy array, optional
        Additional weights. Default: 1.0

    Return
    ------
        The photo-z bias factor.
    """
    return ((weights * lens_ds['zp_bias_num']).sum(axis=0) /
            (weights * lens_ds['zp_bias_den']).sum(axis=0))


def get_lens_weight(lens_ds, lens_data=None, weight_field='weight', second_weight=1.0):
    """Get the user defined weights for lens or random objects.

    Allow a second weight, for example, to re-weight the random
    objects for their redshift distribution.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lens.
    lens_data : numpy array, optional
        Lens catalog.
    weight : string, optional
        Column name of lens weight. Default: 'weight'
    second_weight : numpy array, optional
        Numpy array for secondary lens weight. Default: 1.0

    Return
    ------
    weights : numpy array
        Final weight for lenses.
    """
    # Get the weight
    if weight_field is None:
        weights = lens_ds['weight']
    else:
        weights = lens_data[weight_field]

    # Check the weight has the correct shape
    n_objs, n_bins = (lens_ds['radial_bins']['sum_num_t']).shape
    assert len(weights) == n_objs, "Wrong size for the weights"

    # Assign weight to each lens in each radial bin
    return np.repeat(weights * second_weight, n_bins).reshape((n_objs, n_bins))


def get_r_factor(ds_r_bins, weights=1.):
    """Responsivity R.

    Calculate the responsivity correction factor, R.

    Parameters
    ----------
    ds_r_bins : numpy array
        Pre-compute results in different radial bins.
    weights : float or numpy array, optional
        Optional weights used in calculation. Default: 1.0

    Return
    ------
    r_factor : numpy array
        R-factor for forming the final DeltaSigma singal.
    """
    # Sum of the responsivity
    #r_factor = (np.nansum((ds_r_bins['sum_num_r'] * weights), axis=0) /
    #            np.nansum((ds_r_bins['sum_den'] * weights), axis=0))
    # Empty bins could be 0/0 == nan. Convert that to 0.
    return np.nan_to_num(
        ((ds_r_bins['sum_num_r'] * weights).sum(axis=0) /
         (ds_r_bins['sum_den'] * weights).sum(axis=0)), 0.0)


def get_m_sel_factor(ds_r_bins, weights=1.):
    """Resolution selection bias.

    Parameters
    ----------
    ds_r_bins : numpy array
        Pre-compute results in different radial bins.
    weights : float or numpy array, optional
        Optional weights used in calculation. Default: 1.0

    Return
    ------
    m_sel_factor : numpy array
        Resolution seection bias for forming the final DeltaSigma signal.

    Notes
    -----
        TODO: Should the lens weights be applied here
    """
    # Empty bins could be 0/0 == nan. Convert that to 0.
    return np.nan_to_num(
        ((ds_r_bins['sum_num_m_sel'] * weights).sum(axis=0) /
         (ds_r_bins['sum_den_m_sel'] * weights).sum(axis=0)), 0.0)


def get_k_factor(ds_r_bins, weights=1):
    """Multiplicative factor k.

    Calculate the multiplicative correction factor, k.

    Parameters
    ----------
    ds_r_bins : numpy array
        Pre-compute results in different radial bins.
    weights : float or numpy array, optional
        Optional weights used in calculation. Default: 1.0

    Return
    ------
    k_factor : numpy array
        Multiplicative correction factor for forming the final DeltaSigma signal.
    """
    # Sum of the multiplicative bias
    # Empty bins could be 0/0 == nan. Convert that to 0.
    return np.nan_to_num(
        ((ds_r_bins['sum_num_k'] * weights).sum(axis=0) /
         (ds_r_bins['sum_den'] * weights).sum(axis=0)), 0.0)


def get_calibration_factor(ds_r_bins, weights=1, selection_bias=False):
    """Calibration factor.

    Calculate the calibration factor for delta_sigma.

    Parameters
    ----------
    ds_r_bins : numpy array
        Pre-compute results in different radial bins.
    weights : float or numpy array, optional
        Optional weights used in calculation. Default: 1.0
    selection_bias : boolen, optional
        Whether to include the correction for resolution selection bias.
        Default: False

    Return
    ------
    calibration_factor: numpy array
        Calibration factors for forming final DeltaSigma signal.
    r_factor: numpy array
        R term in the calibration factors.
    k_factor: numpy array
        k-term in the calibration factors.
    m_factor: numpy array
        Resolution selection bias in calibration factors
    """
    r_factor = get_r_factor(ds_r_bins, weights=weights)
    k_factor = get_k_factor(ds_r_bins, weights=weights)

    if selection_bias:
        # Should include the lens weights
        m_factor = get_m_sel_factor(ds_r_bins, weights=weights)
    else:
        # If not necessary, just assign 0.0 everywhere.
        m_factor = np.zeros(len(r_factor))

    # Do not do 1.0 / calibration_factor here
    calibration_factor = 2.0 * r_factor * (1.0 + k_factor) * (1.0 + m_factor)

    return calibration_factor, r_factor, k_factor, m_factor


def get_delta_sigma_ratio(ds_r_bins, weights=1, selection_bias=False):
    """Calculate the delta sigma term based on pre-compute results.

    This applies to both the lenses and the randoms.
    For randoms sometimes an additional weight is necessary to match
    the photometric redshift distributions.  This can be applied
    through the `weights` parameter.

    Parameters
    ----------
    ds_r_bins : numpy array
        Pre-compute results in different radial bins.
    weights : float or numpy array, optional
        Optional weights used in calculation. Default: 1.0
    selection_bias : boolen, optional
        Whether to include the correction for resolution selection bias.
        Default: False

    Return
    ------
    dsigma_t : numpy array
        The tangential term for the DeltaSigma singal.
    dsigma_x : numpy array
        The cross term for the DeltaSigma singal.
    calibration_factor : numpy array
        The calibration factor for the DeltaSigma singal.
    r_factor: numpy array
        R term in the calibration factors.
    k_factor: numpy array
        k-term in the calibration factors.
    m_factor: numpy array
        Resolution selection bias in calibration factors

    Notes:
    ------

    - At this point, the `sum_num_t` and `sum_num_x` have been weighted by
    the `e_weight * inverse_sigma_crit`.
    - `sum_den` is the sum of `ds_weight`, which is `e_weight * inverse_sigma_crit ** 2`.
    - `r_factor` and `r_factor` have been weighted by the shape weight.
    """
    # Form the delta sigma; Empty bins could be 0/0 == nan. Convert that to 0.
    # The tangential term
    dsigma_t = np.nan_to_num(
        (ds_r_bins['sum_num_t'] * weights).sum(axis=0) /
        (ds_r_bins['sum_den'] * weights).sum(axis=0), 0.0)

    # The cross term
    dsigma_x = np.nan_to_num(
        (ds_r_bins['sum_num_x'] * weights).sum(axis=0) /
        (ds_r_bins['sum_den'] * weights).sum(axis=0), 0.0)

    calibration_factor, r_factor, k_factor, m_factor = get_calibration_factor(
        ds_r_bins, weights=weights, selection_bias=selection_bias
        )

    return (dsigma_t, dsigma_x, calibration_factor,
            r_factor, k_factor, m_factor)


def get_boost_factor(lens_bins, rand_bins, rand_weights=1.0, lens_weights=1.0):
    """Boost correction factor.

    TODO: This still needs a lot of tests, still have question
          about the weights

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_weights : numpy array, optional
        Weights for lenses. Default: None
    rand_weights : numpy array, optional
        Weights for randes. Default: None

    Return
    ------
    boost_factor : numpy array
        Boost factor corrections.
    """
    return (
        (lens_bins['sum_den'] * lens_weights).sum(axis=0) * rand_weights.sum(axis=0)
        ) / (
            (rand_bins['sum_den'] * rand_weights).sum(axis=0) * lens_weights.sum(axis=0)
        )


def get_delta_sigma_lens_rand(lens_ds, rand_ds, lens_data, rand_data,
                              use_boost=False,
                              lens_mask=None, rand_mask=None,
                              rand_zweight_nbins=10, qa=False,
                              prefix=None, selection_bias=False,
                              weight_field='weight', same_weight_rand=True,
                              output_array=None, only_dsigma=False):
    """Construct delta sigma.

    Assemble the formal delta sigma signal with the random
    part subtracted.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Lens catalog.
    rand_data : numpy array
        Random catalog.
    lens_mask : boolen array, optional
        Mask for selecting useful lenses. Default: None
    rand_mask : boolen array, optional
        Mask for selection useful randoms. Default: None
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias.
        Default: False
    rand_zweight_nbins: int, optional
        Number of bins when reweighting the redshift of randoms.
    qa : boolen, optional
        Flag for generating QA plots. Default: True
    prefix : str, optional
        Prefix for output QA plots. Default: None
    output_array : structured numpy array, optional
        If available, put the necessary outputs into this array instead of
        outputting them individually.
    only_dsigma : bool, optional
        Only return the final DeltaSigma profile.
        Useful for the bootstrap or jackknife resampling. Default: False
    weight_field : string, optional
        Column name of the lens weight. Default: 'weight'
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True

    Returns
    -------
    output_array : structured numpy array, optional
        Updated output results for DeltaSigma profiles.
    rand_zweight : numpy array
        Redshift weights for randoms.

    Notes
    -----

    The procedures are:
        - Gather the useful lenses and randoms (if available), along with the
          weights for each lens.
        - Calculate the all the useful factors for forming the DeltaSigma
          signal for lenses.
        - If randoms are available.
            * Assign weight to randoms based on the redshift distribution of lenses.
            * Calculate the all the useful factors for forming the DeltaSigma
          signal for randoms.
            * Calculate the boost factor correction.
            * Form the final DeltaSigma signal.
        - If randoms are not available.
            * Form the final DeltaSigma signal.
        - Save the results to an output array, or return them individually.
    """
    # Allow a mask for jackknife resampling and other application
    lens_ds = lens_ds[lens_mask] if lens_mask is not None else lens_ds
    lens_data = lens_data[lens_mask] if lens_mask is not None else lens_data
    rand_ds = rand_ds[rand_mask] if rand_mask is not None else rand_ds
    rand_data = rand_data[rand_mask] if rand_mask is not None else rand_data

    # Get the photometric redshift bias factor for lenses
    zp_bias_lens = get_zp_bias(lens_ds)

    # Assign weights for lenses in all radial bins
    lens_weights = get_lens_weight(lens_ds, lens_data=lens_data, weight_field=weight_field)

    # Get the delta sigma and calibration factors for lenses
    (lens_dsigma_t, lens_dsigma_x, lens_calib,
     lens_r, lens_k, lens_m) = get_delta_sigma_ratio(
         lens_ds['radial_bins'], weights=lens_weights,
         selection_bias=selection_bias)

    if rand_ds is not None:
        # Here should match the redshift distributions between lenses
        # and randoms, provide a new weight for the randoms
        rand_zweight = reweight_rand_photoz(lens_ds['z'], rand_ds['z'],
                                            nbins=rand_zweight_nbins,
                                            qa=qa, prefix=prefix)

        # Get the user define weights for randoms
        # Includes the weight for photo-z
        if same_weight_rand:
            weight_field_rand = weight_field
        else:
            weight_field_rand = 'weight'

        rand_weights = get_lens_weight(
            rand_ds, lens_data=rand_data, weight_field=weight_field_rand,
            second_weight=rand_zweight)

        # Get the photometric redshift bias factor for lenses
        zp_bias_rand = get_zp_bias(rand_ds, weights=rand_zweight)

        # Get the delta sigma and calibration factor for randoms
        (rand_dsigma_t, rand_dsigma_x, rand_calib,
         rand_r, rand_k, rand_m) = get_delta_sigma_ratio(
             rand_ds['radial_bins'], weights=rand_weights,
             selection_bias=selection_bias)

        # Get the boost correction factor
        boost_factor = get_boost_factor(
            lens_ds['radial_bins'], rand_ds['radial_bins'],
            rand_weights=rand_weights, lens_weights=lens_weights)

        if use_boost:
            print("# Boost factor applied !")
            delta_sigma_lr = (
                lens_dsigma_t / lens_calib * boost_factor -
                rand_dsigma_t / rand_calib) * zp_bias_lens
        else:
            delta_sigma_lr = (lens_dsigma_t / lens_calib -
                              rand_dsigma_t / rand_calib) * zp_bias_lens
    else:
        rand_weights = None
        zp_bias_rand = 0.0
        boost_factor = np.ones(len(lens_dsigma_t), dtype="float")
        rand_dsigma_t = np.zeros(len(lens_dsigma_t), dtype="float")
        rand_dsigma_x = np.zeros(len(lens_dsigma_t), dtype="float")
        rand_r = np.ones(len(lens_dsigma_t), dtype="float")
        rand_k = np.zeros(len(lens_dsigma_t), dtype="float")
        rand_m = np.zeros(len(lens_dsigma_t), dtype="float")
        rand_calib = np.ones(len(lens_dsigma_t), dtype="float")
        # Final DeltaSigma profile without random subtraction
        delta_sigma_lr = lens_dsigma_t * lens_calib * zp_bias_lens

    if only_dsigma:
        return delta_sigma_lr

    if output_array is not None:
        output_array["dsigma_lr"] = delta_sigma_lr
        output_array["lens_dsigma_t"] = lens_dsigma_t
        output_array["lens_dsigma_x"] = lens_dsigma_x
        output_array["lens_r"] = lens_r
        output_array["lens_k"] = lens_k
        output_array["lens_m"] = lens_m
        output_array["lens_calib"] = lens_calib
        output_array["rand_dsigma_t"] = rand_dsigma_t
        output_array["rand_dsigma_x"] = rand_dsigma_x
        output_array["rand_r"] = rand_r
        output_array["rand_k"] = rand_k
        output_array["rand_m"] = rand_m
        output_array["rand_calib"] = rand_calib
        output_array["boost_factor"] = boost_factor
        output_array['zp_bias_lens'] = np.full(len(lens_dsigma_t), zp_bias_lens)
        output_array['zp_bias_rand'] = np.full(len(lens_dsigma_t), zp_bias_rand)

        return output_array, lens_weights, rand_weights

    # No, don't do this, this is a bad idea!
    return (delta_sigma_lr, boost_factor, lens_weights, rand_weights,
            lens_dsigma_t, lens_dsigma_x, lens_r, lens_k, lens_m,
            rand_dsigma_t, rand_dsigma_x, rand_r, rand_k, rand_m)


def get_delta_sigma_npairs(lens_ds, rand_ds,
                           lens_weights=None, rand_weights=None, dsigma_output=None):
    """Count lensing pairs.

    Return the number of lens-source and lens-random pairs
    in each radial bin, taking into account that some of the
    randoms are zero-weighted.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_weights : numpy array, optional
        Lens weights.  Default: None
    rand_weights : numpy array, optional
        Random weights.  Default: None
    dsigma_output : structured numpy array, optional
        Array that kept the output results for DeltaSigma profile. Default: None

    Return
    ------
    dsigma_output : structured numpy array
        Updated output array with N_pair information.
    """
    if lens_weights is not None:
        if lens_weights.ndim != 2:
            raise Exception("# Something wrong with the lens weight!")
        lw = lens_weights
        assert len(lw) == len(lens_ds)
    else:
        lw = 1.0

    lens_npairs = (lens_ds['radial_bins']['n_pairs']).sum(axis=0)
    lens_npairs_eff = (lens_ds['radial_bins']['n_pairs'] * lw).sum(axis=0)

    try:
        r_mean_mpc = (lens_ds['radial_bins']['sum_dist']).sum(axis=0) / lens_npairs
    except (ValueError, KeyError):
        r_mean_mpc = lens_npairs * np.nan

    if rand_ds is not None:
        if rand_weights is not None:
            if rand_weights.ndim != 2:
                raise Exception("# Something wrong with the random weight!")
            rw = rand_weights
            assert len(rw) == len(rand_ds)
        else:
            rw = 1.0

        rand_npairs = (rand_ds['radial_bins']['n_pairs']).sum(axis=0)
        rand_npairs_eff = (rand_ds['radial_bins']['n_pairs'] * rw).sum(axis=0)

        # Also get the same mean distance for randoms in each bin
        try:
            r_mean_mpc_rand = (rand_ds['radial_bins']['sum_dist']).sum(axis=0) / rand_npairs
        except (ValueError, KeyError):
            r_mean_mpc_rand = lens_npairs * np.nan
    else:
        rand_npairs = lens_npairs * 0
        rand_npairs_eff = lens_npairs * 0
        r_mean_mpc_rand = lens_npairs * np.nan

    if dsigma_output is not None:
        dsigma_output['r_mean_mpc'] = r_mean_mpc
        dsigma_output['r_mean_mpc_rand'] = r_mean_mpc_rand
        dsigma_output['lens_npairs'] = lens_npairs
        dsigma_output['lens_npairs_eff'] = lens_npairs_eff
        dsigma_output['rand_npairs'] = rand_npairs
        dsigma_output['rand_npairs_eff'] = rand_npairs_eff

        return dsigma_output

    return r_mean_mpc, lens_npairs, lens_npairs_eff, rand_npairs, rand_npairs_eff


def get_delta_sigma_error_simple(lens_ds, rand_ds=None, calib=1.0, calib_rand=1.0,
                                 lens_weights=1.0, rand_weights=1.0):
    """Shape noise error for delta sigma.

    Calculate the erorr of delta_sigma using simple shape noise.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array, optional
        Pre-compute results for randoms. Default: None
    calib : numpy array, optional
        Calibration factors for lenses. Default: 1.0
    calib_rand : numpy array, optional
        Calibration factors for randoms. Default: 1.0
    lens_weights : numpy array, optional
        Lens weight. Default: 1.0
    rand_weights : numpy array, optional
        Random weight. Default: 1.0

    Return
    ------
        Simple error estimates of DeltaSigma profile.
    """
    ds_lens_err = (1.0 / np.sqrt(
        np.nansum(
            (lens_ds['radial_bins']['sum_den'] * lens_weights), axis=0)))

    if rand_ds is not None:
        ds_rand_err = (1.0 / np.sqrt(
            np.nansum(
                (rand_ds['radial_bins']['sum_den'] * rand_weights), axis=0)))

        return np.sqrt((calib * ds_lens_err) ** 2.0 + (calib_rand * ds_rand_err))

    return calib * ds_lens_err


def get_delta_sigma_error_gglens(lens_ds, rand_ds=None, calib=1.0, calib_rand=1.0,
                                 lens_weights=1.0, rand_weights=1.0):
    """Shape noise error for delta sigma.

    Calculate the erorr of delta_sigma using Melanie's code.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array, optional
        Pre-compute results for randoms. Default: None
    calib : numpy array, optional
        Calibration factors for lenses. Default: 1.0
    calib_rand : numpy array, optional
        Calibration factors for randoms. Default: 1.0
    lens_weights : numpy array, optional
        Lens weights.  Default: 1.0
    rand_weights : numpy array, optional
        Random weight. Default: 1.0

    Return
    ------
        Error estimates of DeltaSigma using Melanie Simet's algorithm.
    """
    lens_err_num = np.sqrt(
        np.nansum((lens_ds['radial_bins']['sum_num_t_sq'] * lens_weights), axis=0))
    lens_err_den = np.nansum((lens_ds['radial_bins']['sum_den'] * lens_weights),
                             axis=0)

    lens_err = calib * (lens_err_num / lens_err_den)

    if rand_ds is not None:
        rand_err_num = np.sqrt(
            np.nansum((rand_ds['radial_bins']['sum_num_t_sq'] * rand_weights), axis=0))
        rand_err_den = np.nansum(
            (rand_ds['radial_bins']['sum_den'] * rand_weights),
            axis=0)

        rand_err = calib_rand * (rand_err_num / rand_err_den)

        return np.sqrt(lens_err ** 2.0 + rand_err ** 2.0)

    return lens_err


def get_jk_profile(field, lens_ds, rand_ds, lens_data, rand_data,
                   use_boost=False, selection_bias=False, rand_zweight_nbins=10,
                   weight_field='weight', same_weight_rand=True):
    """Get a single jackknife profile."""
    lens_mask = (lens_ds['jk_field'] != field)
    rand_mask = (rand_ds['jk_field'] != field) if rand_ds is not None else None

    return get_delta_sigma_lens_rand(
        lens_ds, rand_ds, lens_data, rand_data, lens_mask=lens_mask, rand_mask=rand_mask,
        use_boost=use_boost, qa=False, selection_bias=selection_bias,
        rand_zweight_nbins=rand_zweight_nbins, only_dsigma=True,
        weight_field=weight_field, same_weight_rand=same_weight_rand)


def get_delta_sigma_error_jackknife(lens_ds, rand_ds, lens_data, rand_data,
                                    use_boost=False, selection_bias=False,
                                    weight_field='weight', n_jobs=1,
                                    rand_zweight_nbins=10,
                                    same_weight_rand=True):
    """Jackknife error for delta sigma.

    Calculate the erorr of delta_sigma using jackknife resampling.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Lens catalog.
    rand_data : numpy array
        Random catalog.
    n_jobs : int, optional
        Number of processors to run on using joblib. Default: 1
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias.
        Default: False
    weight_field : string, optional
        Column name of the lens weight.  Default: 'weight'
    rand_zweight_nbins: int, optional
        Number of bins when reweighting the redshift of randoms.
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True

    Return
    ------
    jackknife_errors : numpy array
        Error estimates of DeltaSigma profile using Jackknife resampling.
    jk_samples : numpy array
        DeltaSigma profiles from all the Jackknife sub-fields.
    """
    jk_fields = set(lens_ds['jk_field'])

    # For test purpose
    n_jobs = 10
    if (n_jobs > 1) and joblib_available:
        get_jk = partial(get_jk_profile,
                         lens_ds=lens_ds, rand_ds=rand_ds,
                         lens_data=lens_data, rand_data=rand_data,
                         use_boost=use_boost,
                         selection_bias=selection_bias,
                         same_weight_rand=same_weight_rand,
                         rand_zweight_nbins=rand_zweight_nbins,
                         weight_field=weight_field)

        jk_samples = Parallel(
            n_jobs=n_jobs, backend='threading')(
                delayed(get_jk)(field) for field in jk_fields)
    else:
        jk_samples = np.asarray(
            [get_jk_profile(
                field, lens_ds, rand_ds, lens_data, rand_data,
                use_boost=use_boost,
                selection_bias=selection_bias,
                same_weight_rand=same_weight_rand,
                rand_zweight_nbins=rand_zweight_nbins,
                weight_field=weight_field) for field in jk_fields]
        )

    jackknife_errors = np.sqrt(np.var(jk_samples, axis=0) * (len(jk_samples) - 1))

    return jackknife_errors, jk_samples


def get_delta_sigma_errors(lens_ds, rand_ds, lens_data, rand_data, cfg,
                           lens_weights=None, rand_weights=None,
                           dsigma_output=None, n_jobs=1,
                           same_weight_rand=True):
    """Gather different error estimates for DeltaSigma profiles.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Lens catalog.
    rand_data : numpy array
        Random catalog.
    cfg : dict
        Configuration parameters.
    dsigma_output : structured numpy array, optional
        Array that kept the output results for DeltaSigma profile. Default: None
    n_jobs : int, optional
        Number of processors to run on using joblib. Default: 1
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True
    lens_weights : numpy array, optional
        Lens weight. Default: 1.0
    rand_weights : numpy array, optional
        Random weight. Default: 1.0

    Return
    ------
    dsigma_output : structured numpy array
        Updated output array with error estimates.
    jk_samples : numpy array
        DeltaSigma profiles from all the Jackknife sub-fields.
    """
    # Lens weight
    if lens_weights is None:
        lens_weights = get_lens_weight(
            lens_ds, lens_data=lens_data, weight_field=cfg['lens_weight'])

    if rand_weights is None and rand_ds is not None:
        if same_weight_rand:
            weight_field = cfg['lens_weight']
        else:
            weight_field = 'weight'
        rand_weights = get_lens_weight(
            rand_ds, lens_data=rand_data, weight_field=weight_field)

    # Get the naive shape errors
    delta_sigma_err_1 = get_delta_sigma_error_simple(
        lens_ds, rand_ds=rand_ds, calib=dsigma_output['lens_calib'],
        calib_rand=dsigma_output['rand_calib'],
        lens_weights=lens_weights, rand_weights=rand_weights)

    # Get the shape error from Melanie's gglens code
    delta_sigma_err_2 = get_delta_sigma_error_gglens(
        lens_ds, rand_ds=rand_ds, calib=dsigma_output['lens_calib'],
        calib_rand=dsigma_output['rand_calib'],
        lens_weights=lens_weights, rand_weights=rand_weights)

    # Get the jackknife errors
    if cfg['njackknife_fields'] < 3:
        print("# Can not calculate Jackknife error: n_jackknife_field < 3!")
        jackknife_errors, jackknife_samples = delta_sigma_err_1, None
    else:
        jackknife_errors, jackknife_samples = get_delta_sigma_error_jackknife(
            lens_ds, rand_ds, lens_data, rand_data,
            use_boost=cfg['boost_factor'],
            rand_zweight_nbins=cfg['rand_zweight_nbins'],
            selection_bias=cfg['selection_bias'],
            weight_field=cfg['lens_weight'],
            same_weight_rand=cfg['same_weight_rand'],
            n_jobs=n_jobs)
        print("#    Number of useful jackknife regions: %d" % len(jackknife_samples))

    if dsigma_output is not None:
        dsigma_output['dsigma_err_1'] = delta_sigma_err_1
        dsigma_output['dsigma_err_2'] = delta_sigma_err_2
        dsigma_output['dsigma_err_jk'] = jackknife_errors

        return dsigma_output, jackknife_samples

    return delta_sigma_err_1, delta_sigma_err_2, jackknife_errors


def get_delta_sigma_ratio_jackknife(lens_ds, rand_ds, lens_data, rand_data,
                                    lens_ds_2, rand_ds_2, lens_data_2, rand_data_2,
                                    n_bins, use_boost=False, selection_bias=False,
                                    rand_zweight_nbins=10, same_weight_rand=True,
                                    use_diff=False):
    """
    Using the jackknife to measure the ratio between the delta sigma
    profiels from two pre-compute results.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Lens catalog.
    rand_data : numpy array
        Random catalog.
    lens_ds_2 : numpy array
        Pre-compute results for second set of lenses.
    rand_ds_2 : numpy array
        Pre-compute results for second set of randoms.
    lens_data_2 : numpy array
        Secondary lens catalog.
    rand_data_2 : numpy array
        Secondary random catalog.
    n_bins : int
        Number of radial bins.
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias.
        Default: False
    rand_zweight_nbins: int, optional
        Number of bins when reweighting the redshift of randoms.
    weight_field : string, optional
        Column name of the lens weight.  Default: 'weight'
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True
    use_diff : boolen, optional
        Return the differences between two DeltaSigma profiles instead of the ratio.

    Return
    ------
    jk_ratio_avg : numpy array
        Average ratio of the two DeltaSigma profiles.
    jk_ratio_err : numpy array
        Error on the average ratio of the two profiles.
    jk_ratio : numpy array
        All the ratios of DeltaSigma profiles in different Jackknife fields.
    jk_dsig1 : numpy array
        All the DeltaSigma profiles for the first sample.
    jk_dsig2 : numpy array
        All the DeltaSigma profiles for the second sample.
    """
    jk_fields = set(lens_ds['jk_field'])
    jk_dsig1 = np.zeros((len(jk_fields), n_bins), dtype="float64")
    jk_dsig2 = np.zeros((len(jk_fields), n_bins), dtype="float64")
    jk_ratio = np.ones((len(jk_fields), n_bins), dtype="float64")

    for ii, field in enumerate(jk_fields):
        lens_mask = (lens_ds['jk_field'] != field)
        lens_mask_2 = (lens_ds_2['jk_field'] != field)

        rand_mask = (rand_ds['jk_field'] != field) if rand_ds is not None else None
        rand_mask_2 = (rand_ds_2['jk_field'] != field) if rand_ds_2 is not None else None

        jk_dsig1[ii] = get_delta_sigma_lens_rand(
            lens_ds, rand_ds, lens_data, rand_data, only_dsigma=True,
            lens_mask=lens_mask, rand_mask=rand_mask,
            use_boost=use_boost, qa=False,
            selection_bias=selection_bias,
            rand_zweight_nbins=rand_zweight_nbins,
            weight_field=None,
            same_weight_rand=same_weight_rand)

        jk_dsig2[ii] = get_delta_sigma_lens_rand(
            lens_ds_2, rand_ds_2, lens_data_2, rand_data_2, only_dsigma=True,
            lens_mask=lens_mask_2, rand_mask=rand_mask_2,
            use_boost=use_boost, qa=False,
            selection_bias=selection_bias,
            rand_zweight_nbins=rand_zweight_nbins,
            weight_field=None,
            same_weight_rand=same_weight_rand)

        if use_diff:
            jk_ratio[ii] = jk_dsig2[ii] - jk_dsig1[ii]
        else:
            jk_ratio[ii] = jk_dsig2[ii] / jk_dsig1[ii]

    jk_ratio_avg = np.mean(jk_ratio, axis=0)
    jk_ratio_err = np.sqrt(np.var(jk_ratio, axis=0) * (len(jk_ratio) - 1))

    return jk_ratio_avg, jk_ratio_err, jk_ratio, jk_dsig1, jk_dsig2


def get_ds_ratio_two_weights(lens_ds, rand_ds, lens_data, rand_data, cfg, n_bins,
                             use_boost=False, selection_bias=False, use_diff=False,
                             rand_zweight_nbins=10, same_weight_rand=True):
    """
    Using the jackknife to measure the ratio between the delta sigma
    profiels using two different lens weights.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Lens catalog.
    rand_data : numpy array
        Random catalog.
    cfg : dict
        Configuration parameters.
    n_bins : int
        Number of radial bins.
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias.
        Default: False
    rand_zweight_nbins: int, optional
        Number of bins when reweighting the redshift of randoms.
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True
    use_diff : boolen, optional
        Return the differences between two DeltaSigma profiles instead of the ratio.

    Return
    ------
    jk_ratio_avg : numpy array
        Average ratio of the two DeltaSigma profiles.
    jk_ratio_err : numpy array
        Error on the average ratio of the two profiles.
    jk_ratio : numpy array
        All the ratios of DeltaSigma profiles in different Jackknife fields.
    jk_dsig1 : numpy array
        All the DeltaSigma profiles for the first sample.
    jk_dsig2 : numpy array
        All the DeltaSigma profiles for the second sample.
    """
    jk_fields = set(lens_ds['jk_field'])
    jk_dsig1 = np.zeros((len(jk_fields), n_bins), dtype="float64")
    jk_dsig2 = np.zeros((len(jk_fields), n_bins), dtype="float64")
    jk_ratio = np.ones((len(jk_fields), n_bins), dtype="float64")

    weight_1, weight_2 = cfg['lens_weight'], cfg['lens_weight_2']
    print("# Now get the ratio of DSigma profiles using %s and %s" % (
        weight_1, weight_2))
    for ii, field in enumerate(jk_fields):
        lens_mask = (lens_ds['jk_field'] != field)
        rand_mask = (rand_ds['jk_field'] != field) if rand_ds is not None else None

        jk_dsig1[ii] = get_delta_sigma_lens_rand(
            lens_ds, rand_ds, lens_data, rand_data, only_dsigma=True,
            lens_mask=lens_mask, rand_mask=rand_mask,
            use_boost=use_boost, qa=False,
            selection_bias=selection_bias,
            rand_zweight_nbins=rand_zweight_nbins,
            weight_field=weight_1,
            same_weight_rand=same_weight_rand)

        jk_dsig2[ii] = get_delta_sigma_lens_rand(
            lens_ds, rand_ds, lens_data, rand_data, only_dsigma=True,
            lens_mask=lens_mask, rand_mask=rand_mask,
            use_boost=use_boost, qa=False,
            selection_bias=selection_bias,
            rand_zweight_nbins=rand_zweight_nbins,
            weight_field=weight_2,
            same_weight_rand=same_weight_rand)

        if use_diff:
            jk_ratio[ii] = jk_dsig2[ii] - jk_dsig1[ii]
        else:
            jk_ratio[ii] = jk_dsig2[ii] / jk_dsig1[ii]

    print("#    Jackknife the ratios ...")
    jk_ratio_avg = np.mean(jk_ratio, axis=0)
    jk_ratio_err = np.sqrt(np.var(jk_ratio, axis=0) * (len(jk_ratio) - 1))

    return jk_ratio_avg, jk_ratio_err, jk_ratio, jk_dsig1, jk_dsig2


def qa_delta_sigma(dsigma_output, prefix=None, random=False,
                   jackknife_samples=None):
    """Quality assurance plots.

    Create a series of QA plots for the detla_sigma results.

    Parameters
    ----------
    dsigma_output : structured numpy array
        Final output of DeltaSigma profiles.
    prefix : str, optional
        Prefix of output figures.  Default: None.
    random : bool, optional
        Whether random is available. Default: False
    jackknife_samples: numpy array, optional
        All DeltaSigma profiles from Jackknife fields.

    Return
    ------
        Generate a series of QA plots and save as PNG files.

    Notes
    -----
    """
    # Generate an output figure for diagnose.
    # R x DeltaSigma profile
    rdsig_prefix = prefix + '_qa_r_dsigma_profile' if prefix is not None else 'qa_r_dsigma_profile'
    plots.plot_r_delta_sigma(dsigma_output, qa_prefix=rdsig_prefix)

    # DeltaSigma profiles
    dsig_prefix = prefix + '_qa_dsigma_profile' if prefix is not None else 'qa_dsigma_profile'
    plots.plot_delta_sigma(dsigma_output, samples=jackknife_samples, qa_prefix=dsig_prefix)

    # Calibration factors
    calib_prefix = prefix + '_qa_calib_factors' if prefix is not None else 'qa_calib_factors'
    plots.plot_r_calibration_factors(dsigma_output, random=random, qa_prefix=calib_prefix)

    # Number of lens-source pairs
    npair_prefix = prefix + '_qa_n_pairs' if prefix is not None else 'qa_n_pairs'
    plots.plot_r_npairs(dsigma_output, random=random, qa_prefix=npair_prefix)

    # Check the cross-term
    cross_prefix = prefix + '_qa_cross_profile' if prefix is not None else 'qa_cross_profile'
    plots.plot_r_dsigma_tx(dsigma_output, random=random, qa_prefix=cross_prefix)

    boost_prefix = prefix + '_qa_boost_profile' if prefix is not None else 'qa_boost_profile'
    plots.plot_r_boost_factor(dsigma_output, qa_prefix=boost_prefix)


def effective_redshift(ds_pre):
    """Compute the effective redshift for lens or random.

    Parameters
    ----------
    ds_pre : ndarray
        Pre-computed DS results.
    """
    w_ls = np.sum(ds_pre['radial_bins']['sum_den'], axis=1)

    return np.sum(
        ds_pre['z'] * ds_pre['weight'] * w_ls) / np.sum(ds_pre['weight'] * w_ls)
