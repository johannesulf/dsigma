#!/usr/bin/env python3
"""Script to run computeDS procedure."""

import argparse

import numpy as np

from dsigma import config
from dsigma import compute_ds as ds
from dsigma.data_structure import get_ds_output_array


def run(config_file):
    """Stack the delta_sigma signals for selected galaxies.

    Parameters
    ----------
    config_file : str
        Name of the configuration file.

    Return
    ------
        Saved results for stacked DeltaSigma profile computation.

    Notes
    -----

    The key equation behind this calculation is:

        delta_sigma_lr = lens_dismga_t * boost_factor - rand_sigma_t

      and

        calibration_factor = 1.0 / (2.0 * r_factor * (1.0 + k_factor))

      if `selection_bias` is used:

        calibration_factor = 1.0 / (2.0 * r_factor * (1.0 + k_factor) * (1.0 + m_sel))

    Structure of the output:
        - r_mpc : radial bin center in Mpc.
        - r_mean_mpc: mean lens-source distance in one radius bin, in unit of Mpc.
        - r_mean_mpc_rand: mean random-source distance in one radius bin, in unit of Mpc.
        - dsigma_lr : fiducial delta_sigma signal, with random subtraction.
        - dsigma_err_1: simply variance of the weighted mean.
        - dsigma_err_2: delta_sigma error based on Melanie's gglens.
        - dsigma_err_jk: jackknife delta_sigma error.
        - lens_dsigma_t: tangential delta_sigma for lens without any correction.
        - lens_dsigma_x: cross delta_sigma for lens without any correction.
        - lens_r: responsivity correction factor for lens.
        - lens_k: multiplicative bias for lens.
        - lens_m: resolution selection bias for lens.
        - lens_calib: calibration factors for lens.
        - rand_dsigma_t: tangential delta_sigma for random without any correction.
        - rand_dsigma_x: cross delta_sigma for random without any correction.
        - rand_r: responsivity correction factor for random.
        - rand_k: multiplicative bias for random.
        - rand_m: resolution selection bias for random.
        - rand_calib: calibration factors for random.
        - boost_factor: boost correction factor for lens.
        - lens_npairs: number of lens-source pairs in each radial bin.
        - rand_npairs: number of random-source pairs in each radial bin.
        - rand_npairs_eff: number of random-source pairs after re-weight.
    """
    # Load the configuration file.
    print("# Running computeDS for %s\n" % args.config)

    ds_cfg = config.config_computeds(config_file)

    # Prepare the data for computeDS
    (lens_ds, rand_ds, lens_data, rand_data,
     radial_bins, cosmology) = ds.prepare_compute_ds(ds_cfg)

    # Have an empty array for output computeDS results.
    dsigma_output = get_ds_output_array(radial_bins)

    # This is the main results (core function)
    dsigma_output, lens_weights, rand_weights = ds.get_delta_sigma_lens_rand(
        lens_ds, rand_ds, lens_data, rand_data, output_array=dsigma_output,
        selection_bias=ds_cfg['selection_bias'], use_boost=ds_cfg['boost_factor'],
        rand_zweight_nbins=ds_cfg['rand_zweight_nbins'], qa=True,
        prefix=ds_cfg['output_prefix'], same_weight_rand=ds_cfg['same_weight_rand'],
        weight_field=ds_cfg['lens_weight'])

    # Gather three different error estimates of the DeltaSigma profiles
    dsigma_output, jackknife_samples = ds.get_delta_sigma_errors(
        lens_ds, rand_ds, lens_data, rand_data, ds_cfg, dsigma_output=dsigma_output,
        n_jobs=ds_cfg['n_jobs'])

    # Get the number of pairs for lenses and randoms
    dsigma_output = ds.get_delta_sigma_npairs(
        lens_ds, rand_ds, lens_weights=lens_weights, rand_weights=rand_weights,
        dsigma_output=dsigma_output)

    # Save the results
    outfile = ds_cfg['output_prefix'] + '_dsigma'
    outfile = outfile + '_with_random' if (rand_ds is not None) else outfile
    np.savez(outfile, delta_sigma=dsigma_output, cosmology=cosmology,
             jackknife_samples=jackknife_samples, config=ds_cfg,
             lens_weights=lens_weights, rand_weights=rand_weights)

    # Generate an output figure for diagnose.
    ds.qa_delta_sigma(dsigma_output, prefix=ds_cfg['output_prefix'],
                      jackknife_samples=jackknife_samples,
                      random=(rand_ds is not None))

    # -------- Compare to the DeltaSigma profile using a different lens weight ------- #
    # Jackknife the ratio of two DeltaSigma profiles using two different lens weights.
    if ds_cfg['lens_weight_2'] is not None:
        (jk_ratio_avg, jk_ratio_err, jk_ratios,
         jk_dsig1, jk_dsig2) = ds.get_ds_ratio_two_weights(
             lens_ds, rand_ds, lens_data, rand_data, ds_cfg,
             len(dsigma_output['dsigma_lr']), use_diff=ds_cfg['ratios']['diff'],
             use_boost=False, selection_bias=False,
             rand_zweight_nbins=10)

        # Save the ratios to another file
        if ds_cfg['ratios']['diff']:
            outfile += '_wdiff'
        else:
            outfile += '_wratio'

        np.savez(outfile, rad=radial_bins, cosmology=cosmology,
                 diff_avg=jk_ratio_avg, diff_var=jk_ratio_err, diff_arr=jk_ratios,
                 dsig_1=jk_dsig1, dsig_2=jk_dsig2, config=ds_cfg)

    # --------------- Compare to a secondary pre-compute result ----------------- #
    # Jackknife the ratio between the two pre-compute results
    lens_ds_2, rand_ds_2, lens_data_2, rand_data_2 = ds.prepare_compute_ds_second(
        ds_cfg, lens_ds, rand_ds, radial_bins, cosmology)

    if lens_ds_2 is not None:
        print("# Compare the ratio of DeltaSigma profile with another pre-compute result.")
        (jk_ratio_avg, jk_ratio_err, jk_ratios,
         jk_dsig1, jk_dsig2) = ds.get_delta_sigma_ratio_jackknife(
             lens_ds, rand_ds, lens_data, rand_data,
             lens_ds_2, rand_ds_2, lens_data_2, rand_data_2,
             len(dsigma_output['dsigma_lr']),
             use_boost=ds_cfg['boost_factor'], use_diff=ds_cfg['ratios']['diff'],
             selection_bias=ds_cfg['selection_bias'],
             rand_zweight_nbins=ds_cfg['rand_zweight_nbins'])

        # Save the ratios to another file
        # Save the ratios to another file
        if ds_cfg['ratios']['diff']:
            outfile += '_diff'
        else:
            outfile += '_ratio'

        np.savez(outfile, rad=radial_bins, cosmology=cosmology,
                 diff_avg=jk_ratio_avg, diff_var=jk_ratio_err, diff_arr=jk_ratios,
                 dsig_1=jk_dsig1, dsig_2=jk_dsig2, config=ds_cfg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config', type=str,
        help=('location of the configuration file ' +
              'that specifies input/output/cosmology'))

    args = parser.parse_args()

    run(args.config)
