"""Estimates of covariance matrix using bootstrap resampling."""
import numpy as np

from . import functions as f
from . import compute_ds as ds
from . import data_structure as data

try:
    from joblib import Parallel, delayed
    joblib_available = True
except ImportError:
    joblib_available = False


__all__ = ['get_delta_sigma_bootstrap', 'form_delta_sigma_single',
           'gather_delta_sigma_factors', 'form_delta_sigma_all',
           'get_bootstrap_samples_dsigma', 'prep_dsigma_jk',
           'form_dsig_single_bootstrap', 'form_ratio_single_bootstrap',
           'get_bootstrap_samples_ratio', 'get_bootstrap_samples_two_weights']


def prep_dsigma_jk(lens_ds_use, rand_ds, lens_data_use, rand_data, jk_id,
                   weight_field='weight', z_weight=None, z_bins=10,
                   same_weight_rand=True):
    """Prepare all the components to form DeltaSigma profile in one JK region.

    Parameters
    ----------
    lens_ds_use : numpy array
        Precompute results for selected lenses.
    rand_ds : numpy array
        Precompute results for randoms.
    lens_data_use : numpy array
        Lens catalog for the selected ones.
    rand_data : numpy array
        Random catalog.
    jk_id : int
        Index of the Jackknife region.
    weight_field : str, optional
        Column name for the lens weight. Default: 'weight'
    z_weight : array, optional
        Photo-z weight for randoms. Default: None
    z_bins : int, optional
        Number of bins to re-weight the photo-z of randoms. Default: 10
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True

    Return
    ------
        Array that contains all components to form the final DeltaSigma signal.

    """
    lens_ds_jk = lens_ds_use[lens_ds_use['jk_field'] == jk_id]
    lens_data_jk = lens_data_use[lens_ds_use['jk_field'] == jk_id]
    lens_rbins = lens_ds_jk['radial_bins']

    # Assign weights for lenses in all radial bins
    lens_weight = ds.get_lens_weight(
        lens_ds_jk, lens_data=lens_data_jk, weight_field=weight_field)

    if rand_ds is not None:
        rand_ds_jk = rand_ds[rand_ds['jk_field'] == jk_id]
        rand_rbins = rand_ds_jk['radial_bins']
        rand_data_jk = rand_data[rand_ds['jk_field'] == jk_id]

        if z_weight is not None:
            rand_zweight = z_weight[rand_ds['jk_field'] == jk_id]
        else:
            rand_zweight = ds.reweight_rand_photoz(
                lens_ds_jk['z'], rand_ds_jk['z'], nbins=z_bins, qa=False)

        # Random should be able to have its own weight
        if same_weight_rand:
            weight_field_rand = weight_field
        else:
            weight_field_rand = 'weight'

        rand_weight = ds.get_lens_weight(
            rand_ds_jk, lens_data=rand_data_jk,
            weight_field=weight_field_rand, second_weight=rand_zweight)
    else:
        rand_rbins, rand_weight = None, 1.0

    return gather_delta_sigma_factors(
        lens_rbins, rand_rbins, lens_weight=lens_weight, rand_weight=rand_weight)


def form_dsig_single_bootstrap(jk_samples, selection_bias=True, zp_bias=1.0,
                               use_boost=False):
    """Generate a DeltaSigma profile from bootstrap resampling all the Jackknife regions.

    Parameters
    ----------
    jk_samples : list or numpy array
        List of structured array to compute DeltaSigma information for each Jackknife field.
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias. Default: True
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    zp_bias : float, optional
        Photo-z selection bias. Default: 0.0

    Returns
    -------
        DeltaSigma singal from a bootstrapped resampling of the JK regions.
    """
    return form_delta_sigma_all(
        [jk_samples[ii] for ii in np.random.choice(
            np.arange(len(jk_samples)), len(jk_samples))],
        selection_bias=selection_bias, zp_bias=zp_bias, use_boost=use_boost)


def form_ratio_single_bootstrap(jk_samples_1, jk_samples_2,
                                selection_bias=True, zp_bias_1=1.0, zp_bias_2=1.0,
                                use_boost=False, use_diff=False):
    """Generate a DeltaSigma profile from bootstrap resampling all the Jackknife regions.

    Parameters
    ----------
    jk_samples_1 : list or numpy array
        The first list of structured array to compute DeltaSigma information for
        each Jackknife field.
    jk_samples_2 : list or numpy array
        The second list of structured array to compute DeltaSigma information for
        each Jackknife field.
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias. Default: True
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    zp_bias_1, zp_bias_2 : float, optional
        Photo-z selection bias. Default: 0.0
    use_diff : boolen, optional
        Return the differences between two DeltaSigma profiles instead of the ratio.

    Returns
    -------
        DeltaSigma_2 / DeltaSigma_1 for the bootstrap sample
    """
    # Get the bootstrap index. Make sure both samples use the same bootstrapped sample.
    boot = np.random.choice(np.arange(len(jk_samples_1)), len(jk_samples_1))

    if use_diff:
        return (
            form_delta_sigma_all(
                [jk_samples_2[ii] for ii in boot],
                selection_bias=selection_bias, zp_bias=zp_bias_2, use_boost=use_boost) -
            form_delta_sigma_all(
                [jk_samples_1[ii] for ii in boot],
                selection_bias=selection_bias, zp_bias=zp_bias_1, use_boost=use_boost))

    return (
        form_delta_sigma_all(
            [jk_samples_2[ii] for ii in boot],
            selection_bias=selection_bias, zp_bias=zp_bias_2, use_boost=use_boost) /
        form_delta_sigma_all(
            [jk_samples_1[ii] for ii in boot],
            selection_bias=selection_bias, zp_bias=zp_bias_1, use_boost=use_boost))


def get_bootstrap_samples_two_weights(lens_ds, rand_ds, lens_data, rand_data,
                                      weight_1, weight_2, n_boots=5000, z_bins=10,
                                      selection_bias=True, use_boost=False,
                                      same_weight_rand=True, use_diff=False):
    """Get the DeltaSigma profiles and the covariance matrix using
    bootstrap resampling.

    Parameters
    ----------
    lens_ds : numpy array
        Precompute results for selected lenses.
    rand_ds : numpy array
        Precompute results for randoms.
    lens_data : numpy array
        Lens catalog for the selected ones.
    rand_data : numpy array
        Random catalog.
    weight_1, weight_2 : str, optional
        Column name for the lens weight. Default: 'weight'
    z_bins : int, optional
        Number of bins to re-weight the photo-z of randoms. Default: 10
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias. Default: True
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    n_boots : int, optional
        Number of Bootstrap sampling to run. Default: 5000
    n_jobs : int, optional
        Number of processors to run on. Default: 1
    boxsize : int, optional
        Size of the boxcar smoothing kernel to use on the correlation matrix.
        Default: 1
    trunc : float, optional
        Truncate the correlation matrix at this value.  Default: 0.2
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True
    use_diff : boolen, optional
        Return the differences between two DeltaSigma profiles instead of the ratio.

    Return
    ------
    dsig_all : numpy array
        The final DeltaSigma profile of the entire sample.
    dsig_boot : list of numpy arrays
        DeltaSigma profiles of all the
    jk_samples : numpy array
        Array that contains all components to form the final DeltaSigma signal.

    """
    # List of all the available Jackknife fields.
    jk_fields = list(set(lens_ds['jk_field']))

    # Estimate the photo-z selection bias
    zp_bias = ds.get_zp_bias(lens_ds)

    # Estimate the redshift weight for randoms
    # TODO: We can also do this for each Jackknife region.
    if rand_ds is not None:
        z_weight = ds.reweight_rand_photoz(
            lens_ds['z'], rand_ds['z'], nbins=z_bins, qa=False)
    else:
        z_weight = None

    # Gather necessary information for each Jackknife region
    # TODO: Here, the order of the Jackknife region is important
    # TODO: Can have a function to prepare both.
    jk_samples_1 = [prep_dsigma_jk(
        lens_ds, rand_ds, lens_data, rand_data, jk_id,
        same_weight_rand=same_weight_rand,
        weight_field=weight_1, z_weight=z_weight) for jk_id in jk_fields]

    jk_samples_2 = [prep_dsigma_jk(
        lens_ds, rand_ds, lens_data, rand_data, jk_id,
        same_weight_rand=same_weight_rand,
        weight_field=weight_2, z_weight=z_weight) for jk_id in jk_fields]

    # The final DeltaSigma profile for all the Jackknife regions combined
    dsig_all_1 = form_delta_sigma_all(
        jk_samples_1, selection_bias=selection_bias, zp_bias=zp_bias)

    dsig_all_2 = form_delta_sigma_all(
        jk_samples_2, selection_bias=selection_bias, zp_bias=zp_bias)

    # The DeltaSigma profiles for the bootstrap samples.
    ratio_boot = [form_ratio_single_bootstrap(
        jk_samples_1, jk_samples_2, selection_bias=selection_bias,
        zp_bias_1=zp_bias, zp_bias_2=zp_bias, use_diff=use_diff,
        use_boost=use_boost) for ii in np.arange(n_boots)]

    return dsig_all_1, dsig_all_2, ratio_boot, jk_samples_1, jk_samples_2


def get_bootstrap_samples_ratio(lens_ds_1, rand_ds_1, lens_ds_2, rand_ds_2,
                                lens_data_1, lens_data_2, rand_data_1, rand_data_2,
                                n_boots=5000, z_bins=10, selection_bias=True,
                                same_weight_rand=True, use_boost=False, use_diff=False):
    """Get the DeltaSigma profiles and the covariance matrix using
    bootstrap resampling.

    Parameters
    ----------
    lens_ds_1, lens_ds_2 : numpy array
        Precompute results for selected lenses.
    rand_ds_1, rand_ds_2 : numpy array
        Precompute results for randoms.
    lens_data_1, lens_data_2 : numpy array
        Lens catalog for the selected ones.
    weight_field : str, optional
        Column name for the lens weight. Default: 'weight'
    z_bins : int, optional
        Number of bins to re-weight the photo-z of randoms. Default: 10
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias. Default: True
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    n_boots : int, optional
        Number of Bootstrap sampling to run. Default: 5000
    n_jobs : int, optional
        Number of processors to run on. Default: 1
    boxsize : int, optional
        Size of the boxcar smoothing kernel to use on the correlation matrix.
        Default: 1
    trunc : float, optional
        Truncate the correlation matrix at this value.  Default: 0.2
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True
    use_diff : boolen, optional
        Return the differences between two DeltaSigma profiles instead of the ratio.

    Return
    ------
    dsig_all : numpy array
        The final DeltaSigma profile of the entire sample.
    dsig_boot : list of numpy arrays
        DeltaSigma profiles of all the
    jk_samples : numpy array
        Array that contains all components to form the final DeltaSigma signal.

    """
    # List of all the available Jackknife fields.
    jk_fields_1 = list(set(lens_ds_1['jk_field']))
    jk_fields_2 = list(set(lens_ds_2['jk_field']))
    assert np.all(
        np.sort(np.asarray(jk_fields_1)) == np.sort(np.asarray(jk_fields_2)))

    # Estimate the photo-z selection bias
    zp_bias_1 = ds.get_zp_bias(lens_ds_1)
    zp_bias_2 = ds.get_zp_bias(lens_ds_2)

    # Estimate the redshift weight for randoms
    # TODO: We can also do this for each Jackknife region.
    if rand_ds_1 is not None:
        z_weight_1 = ds.reweight_rand_photoz(
            lens_ds_1['z'], rand_ds_1['z'], nbins=z_bins, qa=False)
    else:
        z_weight_1 = 1.

    if rand_ds_2 is not None:
        z_weight_2 = ds.reweight_rand_photoz(
            lens_ds_2['z'], rand_ds_2['z'], nbins=z_bins, qa=False)
    else:
        z_weight_2 = 1.

    # Gather necessary information for each Jackknife region
    # TODO: Here, the order of the Jackknife region is important
    # TODO: Can have a function to prepare both.
    jk_samples_1 = [prep_dsigma_jk(
        lens_ds_1, rand_ds_1, lens_data_1, rand_data_1, jk_id,
        same_weight_rand=same_weight_rand,
        weight_field=None, z_weight=z_weight_1) for jk_id in jk_fields_1]

    jk_samples_2 = [prep_dsigma_jk(
        lens_ds_2, rand_ds_2, lens_data_2, rand_data_2, jk_id,
        same_weight_rand=same_weight_rand,
        weight_field=None, z_weight=z_weight_2) for jk_id in jk_fields_2]

    # The final DeltaSigma profile for all the Jackknife regions combined
    dsig_all_1 = form_delta_sigma_all(
        jk_samples_1, selection_bias=selection_bias, zp_bias=zp_bias_1)

    dsig_all_2 = form_delta_sigma_all(
        jk_samples_2, selection_bias=selection_bias, zp_bias=zp_bias_2)

    # The DeltaSigma profiles for the bootstrap samples.
    ratio_boot = [form_ratio_single_bootstrap(
        jk_samples_1, jk_samples_2, selection_bias=selection_bias,
        zp_bias_1=zp_bias_1, zp_bias_2=zp_bias_2, use_diff=use_diff,
        use_boost=use_boost) for ii in np.arange(n_boots)]

    return dsig_all_1, dsig_all_2, ratio_boot, jk_samples_1, jk_samples_2


def get_bootstrap_samples_dsigma(lens_ds_use, rand_ds, lens_data_use, rand_data,
                                 n_boots=5000, n_jobs=1, same_weight_rand=True,
                                 z_bins=10, selection_bias=True,
                                 weight_field='weight', use_boost=False):
    """Get the DeltaSigma profiles and the covariance matrix using
    bootstrap resampling.

    Parameters
    ----------
    lens_ds_use : numpy array
        Precompute results for selected lenses.
    rand_ds : numpy array
        Precompute results for randoms.
    lens_data_use : numpy array
        Lens catalog for the selected ones.
    weight_field : str, optional
        Column name for the lens weight. Default: 'weight'
    z_bins : int, optional
        Number of bins to re-weight the photo-z of randoms. Default: 10
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias. Default: True
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    n_boots : int, optional
        Number of Bootstrap sampling to run. Default: 5000
    n_jobs : int, optional
        Number of processors to run on. Default: 1
    boxsize : int, optional
        Size of the boxcar smoothing kernel to use on the correlation matrix.
        Default: 1
    trunc : float, optional
        Truncate the correlation matrix at this value.  Default: 0.2
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True

    Return
    ------
    dsig_all : numpy array
        The final DeltaSigma profile of the entire sample.
    dsig_boot : list of numpy arrays
        DeltaSigma profiles of all the
    jk_samples : numpy array
        Array that contains all components to form the final DeltaSigma signal.

    """
    # List of all the available Jackknife fields.
    jk_fields = list(set(lens_ds_use['jk_field']))

    # Estimate the photo-z selection bias
    # TODO: Do we want to estimate this for each Jackknife region
    zp_bias = ds.get_zp_bias(lens_ds_use)

    # Estimate the redshift weight for randoms
    # TODO: We can also do this for each Jackknife region.
    if rand_ds is not None:
        z_weight = ds.reweight_rand_photoz(
            lens_ds_use['z'], rand_ds['z'], nbins=z_bins, qa=False)
    else:
        z_weight = None

    # Gather necessary information for each Jackknife region
    if (n_jobs > 1) and joblib_available:
        jk_samples = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(prep_dsigma_jk)(
                lens_ds_use, rand_ds, lens_data_use, rand_data, jk_id,
                same_weight_rand=same_weight_rand,
                weight_field=weight_field, z_weight=z_weight) for jk_id in jk_fields)
    else:
        jk_samples = [prep_dsigma_jk(
            lens_ds_use, rand_ds, lens_data_use, rand_data, jk_id,
            same_weight_rand=same_weight_rand,
            weight_field=weight_field, z_weight=z_weight) for jk_id in jk_fields]

    # The final DeltaSigma profile for all the Jackknife regions combined
    dsig_all = form_delta_sigma_all(
        jk_samples, selection_bias=selection_bias, zp_bias=zp_bias)

    # The DeltaSigma profiles for the bootstrap samples.
    dsig_boot = [form_dsig_single_bootstrap(
        jk_samples, selection_bias=selection_bias,
        zp_bias=zp_bias, use_boost=use_boost) for ii in np.arange(n_boots)]

    return dsig_all, dsig_boot, jk_samples


def form_delta_sigma_all(ds_list, cross=False, zp_bias=1.0, debug=False,
                         selection_bias=True, use_boost=False,
                         only_lens=False, only_rand=False):
    """Form the final calibrated DeltaSigma signal.

    Parameters
    ----------
    ds_arr : structured numpy array
        Information for forming the DeltaSigma signal in a single field.
    cross : boolen, optional
        Calculate the signal for cross-term instead of the tangential term. Default: False
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias. Default: True
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    zp_bias : float, optional
        Photo-z selection bias. Default: 0.0
    only_lens: boolen, optional
        Only return signal for lenses.
    only_rand: boolen, optional
        Only return signal for randoms.

    Returns
    -------
    dsigma_final: numpy array
        Calibrated DeltaSigma signal.

    """
    # The tangential lensing signals
    if not cross:
        ds_lens = (
            np.asarray([ds['ds_t_num_lens'] for ds in ds_list]).sum(axis=0) /
            np.asarray([ds['ds_den_lens'] for ds in ds_list]).sum(axis=0))
        ds_rand = (
            np.asarray([ds['ds_t_num_rand'] for ds in ds_list]).sum(axis=0) /
            np.asarray([ds['ds_den_rand'] for ds in ds_list]).sum(axis=0))
    else:
        ds_lens = (
            np.asarray([ds['ds_x_num_lens'] for ds in ds_list]).sum(axis=0) /
            np.asarray([ds['ds_den_lens'] for ds in ds_list]).sum(axis=0))
        ds_rand = (
            np.asarray([ds['ds_x_num_rand'] for ds in ds_list]).sum(axis=0) /
            np.asarray([ds['ds_den_rand'] for ds in ds_list]).sum(axis=0))

    # Calibration terms
    calib_lens = (
        2.0 * (np.asarray([ds['r_num_lens'] for ds in ds_list]).sum(axis=0) /
               np.asarray([ds['ds_den_lens'] for ds in ds_list]).sum(axis=0)) *
        (1.0 + (np.asarray([ds['k_num_lens'] for ds in ds_list]).sum(axis=0) /
                np.asarray([ds['ds_den_lens'] for ds in ds_list]).sum(axis=0))))
    calib_rand = (
        2.0 * (np.asarray([ds['r_num_rand'] for ds in ds_list]).sum(axis=0) /
               np.asarray([ds['ds_den_rand'] for ds in ds_list]).sum(axis=0)) *
        (1.0 + (np.asarray([ds['k_num_rand'] for ds in ds_list]).sum(axis=0) /
                np.asarray([ds['ds_den_rand'] for ds in ds_list]).sum(axis=0))))

    if selection_bias:
        calib_lens *= (1.0 + (
            np.asarray([ds['m_num_lens'] for ds in ds_list]).sum(axis=0) /
            np.asarray([ds['m_den_lens'] for ds in ds_list]).sum(axis=0)))
        calib_rand *= (1.0 + (
            np.asarray([ds['m_num_rand'] for ds in ds_list]).sum(axis=0) /
            np.asarray([ds['m_den_rand'] for ds in ds_list]).sum(axis=0)))

    # In case there are NaN caused by 0/0
    ds_lens = np.nan_to_num(ds_lens, 0.0)
    ds_rand = np.nan_to_num(ds_rand, 0.0)
    calib_lens = np.nan_to_num(calib_lens, 1.0)
    calib_rand = np.nan_to_num(calib_rand, 1.0)

    if use_boost:
        boost = (
            (np.asarray([ds['ds_den_lens'] for ds in ds_list]).sum(axis=0) *
             np.asarray([ds['w_rand'] for ds in ds_list]).sum(axis=0)) /
            (np.asarray([ds['ds_den_rand'] for ds in ds_list]).sum(axis=0) *
             np.asarray([ds['w_lens'] for ds in ds_list]).sum(axis=0)))
    else:
        boost = 1.0

    if debug:
        print("# ds_lens", ds_lens)
        print("# calib_lens", calib_lens)
        print("# ds_rand", ds_rand)
        print("# calib_rand", calib_rand)
        print("# boost", boost)

    if only_lens:
        return (ds_lens * boost / calib_lens) * zp_bias

    if only_rand:
        return (ds_rand / calib_rand) * zp_bias

    return ((ds_lens * boost / calib_lens) - (ds_rand / calib_rand)) * zp_bias


def form_delta_sigma_single(ds_arr, cross=False, zp_bias=1.0,
                            selection_bias=True, use_boost=False,
                            only_lens=False, only_rand=False):
    """Form the final calibrated DeltaSigma signal.

    Parameters
    ----------
    ds_arr : structured numpy array
        Information for forming the DeltaSigma signal in a single field.
    cross : boolen, optional
        Calculate the signal for cross-term instead of the tangential term. Default: False
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias. Default: True
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False
    zp_bias : float, optional
        Photo-z selection bias. Default: 0.0
    only_lens: boolen, optional
        Only return signal for lenses.
    only_rand: boolen, optional
        Only return signal for randoms.

    Returns
    -------
    dsigma_final: numpy array
        Calibrated DeltaSigma signal.

    Notes
    -----

    For lens and random, the tangential DeltaSigma sigma is:

        ds_t_lens = ds_t_num_lens / ds_den_lens
        ds_t_rand = ds_t_num_rand / ds_den_rand

    The cross term signal is:

        ds_x_lens = ds_x_num_lens / ds_den_lens
        ds_x_rand = ds_x_num_rand / ds_den_rand

    The R-term in te calibration factor is:

        r_lens = r_num_lens / ds_den_lens
        r_rand = r_num_rand / ds_den_rand

    The k-term in the calibration factor is:

        k_lens = k_num_lens / ds_den_lens
        k_rand = k_num_rand / ds_den_rand

    The optional m-term in the calibration factor is:

        m_lens = m_num_lens / ds_den_lens
        m_rand = m_num_rand / ds_den_lens

    The optional boost correction factor is:

        boost = (ds_den_lens * w_rand) / (ds_den_rand * w_lens)

    The final calibration term is:

        calib_lens = 1. / (2 * r_lens * (1 + k_lens) * (1 + m_lens))
        calib_rand = 1. / (2 * r_rand * (1 + k_rand) * (1 + m_rand))

    The final DeltaSigma signal afte correction:

        dsigma = (ds_t_lens * calib_lens * boost - ds_t_rand * calib_rand) * zp_bias

    """
    # The tangential lensing signals
    if not cross:
        ds_lens = ds_arr['ds_t_num_lens'] / ds_arr['ds_den_lens']
        ds_rand = ds_arr['ds_t_num_rand'] / ds_arr['ds_den_rand']
    else:
        ds_lens = ds_arr['ds_x_num_lens'] / ds_arr['ds_den_lens']
        ds_rand = ds_arr['ds_x_num_rand'] / ds_arr['ds_den_rand']

    # Calibration terms
    calib_lens = (
        2.0 * (ds_arr['r_num_lens'] / ds_arr['ds_den_lens']) *
        (1.0 + ds_arr['k_num_lens'] / ds_arr['ds_den_lens']))
    calib_rand = (
        2.0 * (ds_arr['r_num_rand'] / ds_arr['ds_den_rand']) *
        (1.0 + ds_arr['k_num_rand'] / ds_arr['ds_den_rand']))

    if selection_bias:
        calib_lens *= (1.0 + ds_arr['m_num_lens'] / ds_arr['m_den_lens'])
        calib_rand *= (1.0 + ds_arr['m_num_rand'] / ds_arr['m_den_rand'])

    if use_boost:
        boost = (
            (ds_arr['ds_den_lens'] * ds_arr['w_rand']) / (ds_arr['ds_den_rand'] * ds_arr['w_lens']))
    else:
        boost = 1.0

    if only_lens:
        return (ds_lens * boost / calib_lens) * zp_bias

    if only_rand:
        return (ds_rand * boost / calib_rand) * zp_bias

    return ((ds_lens * boost / calib_lens) - (ds_rand / calib_rand)) * zp_bias


def gather_delta_sigma_factors(lens_rbins, rand_rbins, lens_weight=1.0, rand_weight=1.0):
    """Gather all the information to calculate the DeltaSigma profile in a single field.

    Parameters
    ----------
    lens_rbins : numpy array
        Pre-compute results for lenses.
    rand_rbins : numpy array
        Pre-compute results for randoms.
    lens_weight : numpy array or float, optional
        Lens weight.  Default: 1.0
    rand_weight : numpy array or float, optional
        Lens weight.  Default: 1.0

    Returns
    ------
    ds_out : structured numpy array
        Output information for the DeltaSigma signal in a single field.
    """
    # Get the output array
    # TODO: check this
    ds_out = data.get_ds_out_field(len(lens_rbins['sum_num_t'][0]))

    # DeltaSigma profile for lens
    # The tangential term
    ds_out['ds_t_num_lens'] = (lens_rbins['sum_num_t'] * lens_weight).sum(axis=0)
    # The cross term
    ds_out['ds_x_num_lens'] = (lens_rbins['sum_num_x'] * lens_weight).sum(axis=0)
    # The denominator
    ds_out['ds_den_lens'] = (lens_rbins['sum_den'] * lens_weight).sum(axis=0)

    # The calibration term for lens
    # The numerator for R term
    ds_out['r_num_lens'] = (lens_rbins['sum_num_r'] * lens_weight).sum(axis=0)
    # The numerator for k term
    ds_out['k_num_lens'] = (lens_rbins['sum_num_k'] * lens_weight).sum(axis=0)
    # Optional m term
    ds_out['m_num_lens'] = (lens_rbins['sum_num_m_sel'] * lens_weight).sum(axis=0)
    ds_out['m_den_lens'] = (lens_rbins['sum_den_m_sel'] * lens_weight).sum(axis=0)

    # The sum of the lens weights
    ds_out['w_lens'] = np.sum(lens_weight, axis=0)
    # The sum of the lens weights
    ds_out['w_rand'] = np.sum(lens_weight, axis=0)

    # Number of pairs
    ds_out['lens_npairs'] = ((lens_rbins['n_pairs'] * lens_weight).sum(axis=0) /
                             np.sum(lens_weight))

    if rand_rbins is not None:
        # DeltaSigma profile for random
        # The tangential term
        ds_out['ds_t_num_rand'] = (rand_rbins['sum_num_t'] * rand_weight).sum(axis=0)
        # The cross term
        ds_out['ds_x_num_rand'] = (rand_rbins['sum_num_x'] * rand_weight).sum(axis=0)
        # The denominator
        ds_out['ds_den_rand'] = (rand_rbins['sum_den'] * rand_weight).sum(axis=0)

        # The calibration term for random
        # The numerator for R term
        ds_out['r_num_rand'] = (rand_rbins['sum_num_r'] * rand_weight).sum(axis=0)
        # The numerator for k term
        ds_out['k_num_rand'] = (rand_rbins['sum_num_k'] * rand_weight).sum(axis=0)
        # Optional m term
        ds_out['m_num_rand'] = (rand_rbins['sum_num_m_sel'] * rand_weight).sum(axis=0)
        ds_out['m_den_rand'] = (rand_rbins['sum_den_m_sel'] * rand_weight).sum(axis=0)

        # Number of pairs
        ds_out['rand_npairs'] = ((rand_rbins['n_pairs'] * rand_weight).sum(axis=0) /
                                 np.sum(rand_weight))

    return ds_out


def get_delta_sigma_bootstrap(lens_ds, rand_ds, lens_data, lens_mask=None,
                              weight_field='weight', selection_bias=True,
                              rand_zweight_nbins=10, use_boost=False):
    """Prepare the precompute results for bootstrap resampling.

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Lens catalog.
    lens_mask : boolen array, optional
        Mask for selecting useful lenses. Default: None
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias.
        Default: False
    rand_zweight_nbins: int, optional
        Number of bins when reweighting the redshift of randoms.
    weight_field : string, optional
        Column name of the lens weight. Default: 'weight'
    use_boost : boolen, optional
        Flag to turn on boost factor correction. Default: False

    Returns
    -------

    Notes
    -----
        Deprecated !
    """
    if lens_mask is not None:
        lens_ds_use, lens_data_use = lens_ds[lens_mask], lens_data[lens_mask]
    else:
        lens_ds_use, lens_data_use = lens_ds, lens_data

    # Generate field ID array for one bootstrap sample
    boot = f.bootstrap_fields(list(set(lens_ds_use['jk_field'])))

    # Prepare the precompute results for bootstrapped lens
    lens_boot = np.concatenate(
        [lens_ds_use[lens_ds_use['jk_field'] == field] for field in boot])
    lens_data_boot = np.concatenate(
        [lens_data_use[lens_ds_use['jk_field'] == field] for field in boot])

    # Get the photometric redshift bias factor for lenses
    zp_bias_lens = ds.get_zp_bias(lens_boot)

    # Assign weights for lenses in all radial bins
    lens_weights = ds.get_lens_weight(lens_boot, lens_data_boot, weight_field=weight_field)

    # Get the delta sigma and calibration factors for lenses
    (lens_dsigma_t, _, lens_calib, _, _, _) = ds.get_delta_sigma_ratio(
        lens_boot['radial_bins'], weights=lens_weights,
        selection_bias=selection_bias)

    if rand_ds is not None:
        # Prepare the precompute results for bootstrapped randoms
        rand_boot = np.concatenate([rand_ds[rand_ds['jk_field'] == field] for field in boot])

        # Here should match the redshift distributions between lenses
        # and randoms, provide a new weight for the randoms
        rand_zweight = ds.reweight_rand_photoz(lens_boot['z'], rand_boot['z'],
                                               nbins=rand_zweight_nbins, qa=False)

        # Get the user define weights for randoms
        # Includes the weight for photo-z
        rand_weights = ds.get_lens_weight(rand_boot, second_weight=rand_zweight)

        # Get the delta sigma and calibration factor for randoms
        (rand_dsigma_t, _, rand_calib, _, _, _) = ds.get_delta_sigma_ratio(
            rand_boot['radial_bins'], weights=rand_weights,
            selection_bias=selection_bias)

        if use_boost:
            # Get the boost correction factor
            boost_factor = ds.get_boost_factor(
                lens_boot['radial_bins'], rand_boot['radial_bins'],
                rand_weights=rand_weights, lens_weights=lens_weights)

            print("# Boost factor applied !")
            return (lens_dsigma_t / lens_calib * boost_factor -
                    rand_dsigma_t / rand_calib) * zp_bias_lens

        return (lens_dsigma_t / lens_calib - rand_dsigma_t / rand_calib) * zp_bias_lens

    # Final DeltaSigma profile without random subtraction
    return lens_dsigma_t * lens_calib * zp_bias_lens
