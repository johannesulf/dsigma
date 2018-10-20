#!/usr/bin/env python3
"""Script to run computeDS procedure."""

import argparse

import numpy as np

from dsigma import config
from dsigma import plots as pl
from dsigma import functions as fu
from dsigma import compute_ds as ds
from dsigma import covariance as cov


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

    """
    # Parse the configuration file.
    print("# Running computeDS for %s\n" % args.config)
    ds_cfg = config.config_computeds(config_file)

    # ------------------- Deal with the primary pre-compute result ------------------- #
    # Prepare the data for computeDS
    # Selection of useful lenses is applied in this step too
    (lens_ds_1, rand_ds_1, lens_data_1, rand_data_1,
     radial_bins, cosmology) = ds.prepare_compute_ds(ds_cfg)

    # Gather the Bootstrap samples
    dsig_all, dsig_boot, dsig_jk = cov.get_bootstrap_samples_dsigma(
        lens_ds_1, rand_ds_1, lens_data_1, rand_data_1,
        n_boots=ds_cfg['covariance']['n_boots'], n_jobs=ds_cfg['n_jobs'],
        z_bins=ds_cfg['rand_zweight_nbins'], selection_bias=ds_cfg['selection_bias'],
        weight_field=ds_cfg['lens_weight'], use_boost=ds_cfg['boost_factor'],
        same_weight_rand=ds_cfg['same_weight_rand'])

    # Get the correlation and covariance matrix
    cor_org, cor_trunc, cov_trunc = fu.smooth_cov(
        dsig_boot, boxsize=ds_cfg['covariance']['boxsize'],
        trunc=ds_cfg['covariance']['trunc'])

    # Save the results
    outfile = ds_cfg['output_prefix'] + '_ds_cov'
    outfile = outfile + '_with_random' if (rand_ds_1 is not None) else outfile

    np.savez(outfile, r_mpc=radial_bins, config=ds_cfg, delta_sigma=dsig_all,
             dsigma_boot=dsig_boot, dsigma_jackknife=dsig_jk,
             cor_org=cor_org, cov_trunc=cov_trunc)

    # Generate an output figure for diagnose.
    pl.plot_corr_matrix(np.log10(radial_bins), cor_trunc, qa_prefix=ds_cfg['output_prefix'])
    # ------------------- Deal with the primary pre-compute result ------------------- #

    # --------------- Compare to a secondary pre-compute result ----------------- #
    # Jackknife the ratio between the two pre-compute results
    if ds_cfg['ds_lenses_2'] is not None:
        print("\n# Compare the ratio of DeltaSigma profile with another pre-compute result.")
        lens_ds_2, rand_ds_2, lens_data_2, rand_data_2 = ds.prepare_compute_ds_second(
            ds_cfg, lens_ds_1, rand_ds_1, radial_bins, cosmology)

        dsig_1, dsig_2, ratios, jk_1, jk_2 = cov.get_bootstrap_samples_ratio(
            lens_ds_1, rand_ds_1, lens_ds_2, rand_ds_2,
            lens_data_1, lens_data_2, rand_data_1, rand_data_2,
            n_boots=ds_cfg['covariance']['n_boots'], use_boost=ds_cfg['boost_factor'],
            selection_bias=ds_cfg['selection_bias'], use_diff=ds_cfg['ratios']['diff'],
            same_weight_rand=ds_cfg['same_weight_rand'])

        # Original covariance matrix
        cov_org = np.cov(np.asarray(ratios), rowvar=False)

        # Truncated covariance matrix
        cor_ratio_org, cor_ratio_trunc, cov_ratio_trunc = fu.smooth_cov(
            ratios, boxsize=ds_cfg['covariance']['boxsize'],
            trunc=ds_cfg['covariance']['trunc'])

        # If the ratios are all 1.0, the covarance matrix will be weird
        if not np.all(np.isfinite(cov_ratio_trunc)):
            print("\n# There is NaN in the covarance matrix !")
            err_use = np.sqrt(np.diag(cov_org))
        else:
            err_use = cov_ratio_trunc

        # Average ratio and error based on the diagnoal terms of the covariance matrix
        avg_ratio, err_ratio = np.nanmean(ratios, axis=0), np.sqrt(np.diag(cov_ratio_trunc))

        # Estimate the average ratios
        f1, f1_err = fu.fit_avg_ratio(radial_bins, avg_ratio, err_use,
                                      rmin=ds_cfg['ratios']['r1'], rmax=ds_cfg['ratios']['r3'])
        f2, f2_err = fu.fit_avg_ratio(radial_bins, avg_ratio, err_use,
                                      rmin=ds_cfg['ratios']['r1'], rmax=ds_cfg['ratios']['r2'])
        f3, f3_err = fu.fit_avg_ratio(radial_bins, avg_ratio, err_use,
                                      rmin=ds_cfg['ratios']['r2'], rmax=ds_cfg['ratios']['r3'])
        print("\n# Average ratios:")
        print("#     %4.1f - %4.1f Mpc : %6.3f +/- %6.3f" % (
            ds_cfg['ratios']['r1'], ds_cfg['ratios']['r3'], f1, f1_err))
        print("#     %4.1f - %4.1f Mpc : %6.3f +/- %6.3f" % (
            ds_cfg['ratios']['r1'], ds_cfg['ratios']['r2'], f2, f2_err))
        print("#     %4.1f - %4.1f Mpc : %6.3f +/- %6.3f" % (
            ds_cfg['ratios']['r2'], ds_cfg['ratios']['r3'], f3, f3_err))

        # Save the results
        if ds_cfg['ratios']['diff']:
            outfile = ds_cfg['output_prefix'] + '_ds_cov_diff'
        else:
            outfile = ds_cfg['output_prefix'] + '_ds_cov_ratio'

        np.savez(outfile, r_mpc=radial_bins, config=ds_cfg,
                 dsigma_1=dsig_1, dsigma_2=dsig_2, ratio_boot=ratios,
                 dsigma_jk_1=jk_1, dsigma_jk_2=jk_2,
                 avg_ratio=avg_ratio, err_ratio=err_ratio,
                 cor_org=cor_ratio_org, cov_trunc=cov_ratio_trunc)

        # Generate an output figure for diagnose.
        pl.plot_corr_matrix(np.log10(radial_bins), cor_ratio_trunc, qa_prefix=outfile)
    # --------------- Compare to a secondary pre-compute result ----------------- #

    # -------- Compare to the DeltaSigma profile using a different lens weight ------- #
    # Jackknife the ratio of two DeltaSigma profiles using two different lens weights.
    if ds_cfg['lens_weight_2'] is not None:
        dsig_1, dsig_2, wratios, jk_1, jk_2 = cov.get_bootstrap_samples_two_weights(
            lens_ds_1, rand_ds_1, lens_data_1, rand_data_1,
            ds_cfg['lens_weight'], ds_cfg['lens_weight_2'],
            n_boots=ds_cfg['covariance']['n_boots'], use_boost=ds_cfg['boost_factor'],
            selection_bias=ds_cfg['selection_bias'], use_diff=ds_cfg['ratios']['diff'],
            same_weight_rand=ds_cfg['same_weight_rand'])

        cor_wratio_org, cor_wratio_trunc, cov_wratio_trunc = fu.smooth_cov(
            wratios, boxsize=ds_cfg['covariance']['boxsize'],
            trunc=ds_cfg['covariance']['trunc'])

        # Average ratio and error based on the diagnoal terms of the covariance matrix
        avg_wratio, err_wratio = np.nanmean(wratios, axis=0), np.sqrt(np.diag(cov_wratio_trunc))

        # Estimate the average ratios
        f1, f1_err = fu.fit_avg_ratio(radial_bins, avg_wratio, cov_wratio_trunc,
                                      rmin=ds_cfg['ratios']['r1'], rmax=ds_cfg['ratios']['r3'])
        f2, f2_err = fu.fit_avg_ratio(radial_bins, avg_wratio, cov_wratio_trunc,
                                      rmin=ds_cfg['ratios']['r1'], rmax=ds_cfg['ratios']['r2'])
        f3, f3_err = fu.fit_avg_ratio(radial_bins, avg_wratio, cov_wratio_trunc,
                                      rmin=ds_cfg['ratios']['r2'], rmax=ds_cfg['ratios']['r3'])
        print("\n# Average ratios:")
        print("#     %4.1f - %4.1f Mpc : %6.3f +/- %6.3f" % (
            ds_cfg['ratios']['r1'], ds_cfg['ratios']['r3'], f1, f1_err))
        print("#     %4.1f - %4.1f Mpc : %6.3f +/- %6.3f" % (
            ds_cfg['ratios']['r1'], ds_cfg['ratios']['r2'], f2, f2_err))
        print("#     %4.1f - %4.1f Mpc : %6.3f +/- %6.3f" % (
            ds_cfg['ratios']['r2'], ds_cfg['ratios']['r3'], f3, f3_err))

        # Save the results
        if ds_cfg['ratios']['diff']:
            outfile = ds_cfg['output_prefix'] + '_ds_cov_wdiff'
        else:
            outfile = ds_cfg['output_prefix'] + '_ds_cov_wratio'

        np.savez(outfile, r_mpc=radial_bins, config=ds_cfg,
                 dsigma_1=dsig_1, dsigma_2=dsig_2, ratio_boot=wratios,
                 dsigma_jk_1=jk_1, dsigma_jk_2=jk_2,
                 avg_ratio=avg_wratio, err_ratio=err_wratio,
                 cor_org=cor_wratio_org, cov_trunc=cov_wratio_trunc)

        # Generate an output figure for diagnose.
        pl.plot_corr_matrix(np.log10(radial_bins), cor_wratio_trunc, qa_prefix=outfile)
    # -------- Compare to the DeltaSigma profile using a different lens weight ------- #


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config', type=str,
        help=('location of the configuration file ' +
              'that specifies input/output/cosmology'))

    args = parser.parse_args()

    run(args.config)
