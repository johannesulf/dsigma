#!/usr/bin/env python3
"""Script to run stackDS procedure."""

import pickle
import argparse

import numpy as np

from dsigma import config
from dsigma import compute_ds as ds
from dsigma.stack_ds import batch_delta_sigma


def run(config_file, mask_file, pickle_output='dsigma_stack'):
    """Get the stacked DeltaSigma profiles for different selections of lens.

    Parameters
    ----------
    config_file : str
        Name of the configuration file.
    mask_file : str
        Name of the '.npy' file that stores the list of lens masks.
    n_jobs: int, optional
        Number of jobs to run on.  Default: 1

    Return
    ------
        Saved results for stacked DeltaSigma profile computation.
    """
    # Load the configuration file.
    print("# Running computeDS for %s\n" % args.config)

    ds_cfg = config.config_computeds(config_file)

    # Load the data: now need to load lens catalog, pre-compute
    # results for lens and random (if necessary)
    lens_pre, rand_pre, lens_data, rand_data = ds.load_data_delta_sigma(ds_cfg)

    # Load the mask array
    mask_arr = np.load(mask_file)

    try:
        n_lens, n_mask = len(mask_arr[0]), len(mask_arr)
    except TypeError:
        n_lens, n_mask = len(mask_arr), 1

    # Make sure the mask and data are compatablie
    assert n_lens == len(lens_data), "# Lens data and the mask are not consistent!"

    print("\n# Find %d lens mask array(s)" % n_mask)
    if n_mask == 1:
        mask_arr = [mask_arr]

    results = batch_delta_sigma(
        lens_pre, lens_data, mask_arr, 
        rand_pre=rand_pre, rand_data=rand_data,
        njackknife_fields=ds_cfg['njackknife_fields'],
        boost_factor=ds_cfg['boost_factor'],
        selection_bias=ds_cfg['selection_bias'],
        weight_field=ds_cfg['lens_weight'],
        rand_zweight_nbins=ds_cfg['rand_zweight_nbins'],
        same_weight_rand=ds_cfg['same_weight_rand'],
        save=False, qa=False)

    # Save the results as a pickle file
    pickle.dump(results, open(pickle_output, 'wb'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config', type=str, help=('Configuration file name.'))
    parser.add_argument(
        'mask', type=str, help=('File name for list of lens masks.'))
    parser.add_argument(
        '-p', dest='n_jobs', type=int,
        default=1, help="Number of jobs to run at once.")
    parser.add_argument(
        '-o', dest='output', type=str,
        default='dsigma_stack', help="Output file name.")

    args = parser.parse_args()

    run(args.config, args.mask, n_jobs=args.n_jobs, pickle_output=args.output)
