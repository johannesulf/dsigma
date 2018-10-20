#!/usr/bin/env python3

import os
import argparse

import numpy as np

from dsigma import config
from dsigma import functions as f
from dsigma.precompute_ds import (delta_sigma_catalog, photo_z_selection,
                                  prepare_photoz_calib)


def run(config_file, n_jobs=1):
    """Precompute delta sigma signal for a lens catalog.

    Parameters
    ----------
    config_file : string
        Name of the configuration file in `.yaml` format.
    n_jobs : int
        Number of jobs to run at the same time.

    """
    # Update the configuration parameters
    cfg = config.config_precompute(config_file)

    # Load the lens, source catalogs. They should be in `.npy` format.
    lenses, sources = np.load(cfg['lens_catalog']), np.load(cfg['source_catalog'])

    # Apply global and specinfo photo-z seleciton
    sources_use, photoz_mask = photo_z_selection(sources, cfg)

    # Optional photometric redshift calibration
    calib = prepare_photoz_calib(cfg)

    # Perform pre-compute for the lens catalog
    results, lenses_new = delta_sigma_catalog(lenses, sources_use, cfg,
                                              calib=calib, n_jobs=n_jobs)

    # Save the output file
    np.savez(cfg['outfile'], delta_sigma=np.array(results),
             radial_bins=f.get_radial_bin_centers(cfg['binning']),
             cosmology=cfg['cosmology'], photoz_mask=photoz_mask)

    # Update the lens catalog
    np.save(os.path.splitext(cfg['lens_catalog'])[0] + '_new.npy', lenses_new)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config', type=str,
        help='Configuration file that specifies input/output/cosmology.')
    parser.add_argument(
        '-p', dest='n_jobs', type=int, metavar="n_jobs",
        default=1, help="Number of jobs to run at once.")

    args = parser.parse_args()

    print("# Running precompute for %s\n" % args.config)

    run(args.config, n_jobs=args.n_jobs)
