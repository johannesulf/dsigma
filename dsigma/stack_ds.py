"""Stack delta sigma signal."""
import pickle

from functools import partial

import numpy as np

from . import jackknife as jk
from . import compute_ds as ds
from . import data_structure as data

__all__ = ["form_delta_sigma", "stack_delta_sigma", "batch_delta_sigma"]


def form_delta_sigma(lens_ds, rand_ds, lens_data, rand_data, radial_bins,
                     boost_factor=False, selection_bias=False, weight_field='weight',
                     rand_zweight_nbins=10, save=False, qa=False,
                     same_weight_rand=True, output_prefix='delta_sigma'):
    """Delta sigma measurements.

    The key equation behind this calculation is:

      delta_sigma_lr = lens_dismga_t * boost_factor - rand_sigma_t

      and

      calibration_factor = 1.0 / (2.0 * r_factor * (1.0 + k_factor))

    Parameters
    ----------
    lens_ds : numpy array
        Pre-compute results for lenses.
    rand_ds : numpy array
        Pre-compute results for randoms.
    lens_data : numpy array
        Lens catalog.
    rand_data : numpy array
        Lens catalog.
    radial_bins : numpy array
        Radial bins for the DeltaSigma profile.
    boost_factor : boolen, optional
        Flag to turn on boost factor correction. Default: False
    selection_bias: boolen, optional
        Flag to include correction of resolution selection bias.
        Default: False
    rand_zweight_nbins: int, optional
        Number of bins when reweighting the redshift of randoms.
    weight_field : string, optional
        Column name of the lens weight.  Default: 'weight'
    save: boolen, optional
        Whether to save the results as `.npz` file. Default: False
    qa: boolen, optional
        Whether to generate QA plots. Default: False
    output_prefix: str, optional
        Prefix for output.  Default: 'delta_sigma'
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True

    Return
    ------
    dsigma_output : structured numpy array
        Output array for information related to DeltaSigma profile
    """
    # Have an empty array for output computeDS results.
    dsigma_output = data.get_ds_output_array(radial_bins)

    # This is the main results (core function)
    dsigma_output, lens_weights, rand_weights = ds.get_delta_sigma_lens_rand(
        lens_ds, rand_ds, lens_data, rand_data, output_array=dsigma_output,
        selection_bias=selection_bias, use_boost=boost_factor,
        rand_zweight_nbins=rand_zweight_nbins, qa=qa,
        prefix=output_prefix, weight_field=weight_field,
        same_weight_rand=same_weight_rand)

    # Get the naive shape errors
    dsigma_output['dsigma_err_1'] = ds.get_delta_sigma_error_simple(
        lens_ds, rand_ds=rand_ds, calib=dsigma_output['lens_calib'],
        calib_rand=dsigma_output['rand_calib'],
        lens_weights=lens_weights, rand_weights=rand_weights)

    # Get the shape error from Melanie's gglens code
    dsigma_output['dsigma_err_2'] = ds.get_delta_sigma_error_gglens(
        lens_ds, rand_ds=rand_ds, calib=dsigma_output['lens_calib'],
        calib_rand=dsigma_output['rand_calib'],
        lens_weights=lens_weights, rand_weights=rand_weights)

    # Get the jackknife errors
    if len(list(set(lens_ds['jk_field']))) < 3:
        print("# Can not calculate Jackknife error: n_jackknife_field < 3!")
        jackknife_errors, jackknife_samples = dsigma_output['dsigma_err_1'], None
    else:
        jackknife_errors, jackknife_samples = ds.get_delta_sigma_error_jackknife(
            lens_ds, rand_ds, lens_data, rand_data,
            same_weight_rand=same_weight_rand, use_boost=boost_factor,
            selection_bias=selection_bias, rand_zweight_nbins=rand_zweight_nbins,
            weight_field=weight_field)
        print("#    Number of useful jackknife regions: %d" % len(jackknife_samples))

    dsigma_output['dsigma_err_jk'] = jackknife_errors

    # Get the number of pairs for lenses and randoms
    dsigma_output = ds.get_delta_sigma_npairs(
        lens_ds, rand_ds, lens_weights=lens_weights, rand_weights=rand_weights,
        dsigma_output=dsigma_output)

    # Save the results
    if save:
        outfile = output_prefix + '_dsigma'
        outfile = outfile + '_with_random' if (rand_ds is not None) else outfile
        np.savez(outfile, delta_sigma=dsigma_output,
                 jackknife_samples=jackknife_samples, lens_data=lens_data,
                 lens_weight=lens_weights, rand_weights=rand_weights)

    # Generate an output figure for diagnose.
    if qa:
        ds.qa_delta_sigma(dsigma_output, prefix=output_prefix,
                          jackknife_samples=jackknife_samples,
                          random=(rand_ds is not None))

    return dsigma_output, lens_weights, rand_weights


def stack_delta_sigma(lens_ds, lens_data, radial_bins, rand_ds=None, rand_data=None,
                      lens_mask=None, boost_factor=False, selection_bias=False,
                      weight_field='weight', rand_zweight_nbins=10, save=False, qa=False,
                      same_weight_rand=True, output_prefix='delta_sigma'):
    """Stacked Delta Sigma signals using pre-compute result.

    Parameters
    ----------
    lens_ds: numpy array
        Precomputed lensing results for leses.
    lens_data : numpy array
        Lens catalog.
    rand_ds: numpy array, optional
        Precomputed lensing results for random.  Default: None
    rand_data : numpy array
        Random catalog.
    lens_mask: boolen array, optional
        Boolen array for selecting lens to stacking.
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True

    Return
    ------
    dsigma_output : structured numpy array
        Output array for information related to DeltaSigma profile
    """
    # Select a subsample of lenses if mask is available
    if lens_mask is not None:
        assert len(lens_mask) == len(lens_ds)
        print("#    Sample: %d / %d galaxies" % (sum(lens_mask), len(lens_ds)))
        lens_ds_use, lens_data_use = lens_ds[lens_mask], lens_data[lens_mask]
    else:
        lens_ds_use, lens_data_use = lens_ds, lens_data

    dsigma_out, lens_weights, rand_weights = form_delta_sigma(
        lens_ds_use, rand_ds, lens_data_use, rand_data, radial_bins,
        boost_factor=boost_factor, selection_bias=selection_bias,
        weight_field=weight_field, rand_zweight_nbins=rand_zweight_nbins,
        same_weight_rand=same_weight_rand,
        save=save, qa=qa, output_prefix=output_prefix)

    return dsigma_out, lens_weights, rand_weights


def stack_with_mask(lens_ds, lens_data, rand_ds, rand_data, radial_bins, lens_mask,
                    boost_factor=False, selection_bias=False, weight_field='weight',
                    rand_zweight_nbins=10, save=False, qa=False,
                    same_weight_rand=True, output_prefix='delta_sigma'):
    """Simple wrapper of stack_delta_sigma, help multiprocessing."""
    # Select a subsample of lenses if mask is available
    if lens_mask is not None:
        assert len(lens_mask) == len(lens_ds), "\n# Wrong mask size!"
        print("#    Sample: %d / %d galaxies" % (sum(lens_mask), len(lens_ds)))
        lens_ds_use, lens_data_use = lens_ds[lens_mask], lens_data[lens_mask]
    else:
        lens_ds_use, lens_data_use = lens_ds, lens_data

    dsigma_out, _, _ = form_delta_sigma(
        lens_ds_use, rand_ds, lens_data_use, rand_data, radial_bins,
        boost_factor=boost_factor, selection_bias=selection_bias,
        weight_field=weight_field, rand_zweight_nbins=rand_zweight_nbins,
        same_weight_rand=same_weight_rand, save=save, qa=qa,
        output_prefix=output_prefix)

    return dsigma_out


def batch_delta_sigma(lens_pre,
                      lens_data,
                      mask_list,
                      rand_pre=None,
                      rand_data=None,
                      pickle_output=None,
                      njackknife_fields=31,
                      boost_factor=False,
                      selection_bias=False,
                      weight_field='weight',
                      rand_zweight_nbins=10,
                      same_weight_rand=True,
                      save=False,
                      qa=False,
                      output_prefix='delta_sigma'):
    """Stacked Delta Sigma signals using pre-compute result.

    Parameters
    ----------
    lens_pre: numpy array
        Precomputed lensing results for leses.
    lens_data : numpy array
        Lens catalog.
    rand_pre: numpy array
        Precomputed lensing results for random objects.
    rand_data : numpy array
        Random catalog.
    mask_list: list of boolen array
        List of lens masks.
    rand_ds: numpy array, optional
        Precomputed lensing results for random.  Default: None
    lens_mask: boolen array, optional
        Boolen array for selecting lens to stacking.
    njackknife_fields: int, optional
        Number of sub-regions for Jackknife resampling.  Default: 41
    pickle_output: str, optional
        Name of the output pickle file.  Default: None
    same_weight_rand : bool, optional
        Use the same weight column for random or not.  Default: True

    Return
    ------
    results : list of structured numpy array
        Stacked DeltaSigma results for each selection of lens.
    """
    # Make sure the pre-compute results and the lens catalog are consistent
    # TODO: this step takes long time
    ds.assert_precompute_catalog_consistent(lens_pre, lens_data)

    # Get the pre-compute deltaSigma data, radial bins
    lens_ds, radial_bins = lens_pre['delta_sigma'], lens_pre['radial_bins']

    rand_ds = rand_pre["delta_sigma"] if rand_pre is not None else None

    # If available, make sure the random and lens have consistent pre-compute settings.
    if rand_pre is not None:
        ds.assert_lens_rand_consistent(lens_pre, rand_pre)
        if rand_data is None:
            raise Exception("# Need the random object catalog!")
        else:
            assert len(rand_data) == len(rand_ds)

    # Assign the jackknife regions if necessary
    if rand_ds is not None:
        lens_ds, rand_ds = jk.add_jackknife_both(lens_ds, rand_ds, njackknife_fields)
    else:
        lens_ds = jk.add_jackknife_field(lens_ds, njackknife_fields)

    single_stack = partial(stack_with_mask,
                           lens_ds=lens_ds, lens_data=lens_data,
                           rand_ds=rand_ds, rand_data=rand_data,
                           radial_bins=radial_bins,
                           boost_factor=boost_factor,
                           selection_bias=selection_bias,
                           weight_field=weight_field,
                           rand_zweight_nbins=rand_zweight_nbins,
                           same_weight_rand=same_weight_rand,
                           save=save, qa=qa,
                           output_prefix=output_prefix)

    # We need to know the order of the mask, so do not use multiprocessing
    results = [single_stack(lens_mask=mask) for mask in mask_list]

    if pickle_output is not None:
        pickle.dump(results, open(pickle_output, 'wb'))

    return results
