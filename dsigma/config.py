"""Configuration parameters."""

import os
import yaml

from . import data_structure as ds

__all__ = ["config_precompute"]

def config_precompute(config_file):
    """Prepare configurations for pre-compute process.

    Read configuration parameters from an input .yaml file.

    Parameters
    ----------
    config_file : string
        Name of the configuration file in `.yaml` format.

    Return
    ------
        config : dict
            Updated configuration parameters

    """
    # Read the configuration file
    cfg = yaml.load(open(config_file))

    # Check input catalogs
    if not os.path.isfile(cfg['source_catalog']):
        raise Exception("# Cannot find the source catalog: %s" % cfg['source_catalog'])

    if not os.path.isfile(cfg['lens_catalog']):
        raise Exception("# Cannot find the lens catalog: %s" % cfg['lens_catalog'])

    if 'outfile' not in cfg:
        cfg['outfile'] = 'delta_sigma.npz'

    # data structure of the source catalog
    if 'version' not in cfg:
        cfg["version"] = 's16a'

    # Get a dictionary for the useful fields in source catalog
    source_fields = ds.get_source_fields(cfg["version"])

    # Get a dictionary for the useful fields in the source catalog
    lens_fields = ds.get_lens_fields(cfg)

    # Cosmology related parameters
    if 'H0' not in cfg["cosmology"]:
        cfg["cosmology"]["H0"] = 70.0

    if 'omega_m' not in cfg["cosmology"]:
        cfg["cosmology"]["omega_m"] = 0.3

    if 'omega_l' not in cfg["cosmology"]:
        cfg["cosmology"]["omega_l"] = 1.0 - cfg["cosmology"]["omega_m"]
    print("#    Adopt cosmology (%5.2f, %4.2f, %4.2f)" % (
        cfg['cosmology']['H0'], cfg['cosmology']['omega_m'], cfg['cosmology']['omega_l']))

    if 'omega_k' not in cfg["cosmology"]:
        cfg["cosmology"]["omega_k"] = 0.0

    if 'flat' not in cfg["cosmology"]:
        cfg["cosmology"]["flat"] = True

    if 'comoving' not in cfg:
        cfg["comoving"] = False
    if cfg['comoving']:
        print("#    Using comoving instead of physical coordinates!")

    # Parameters about binning
    if 'rmin_mpc' not in cfg['binning']:
        cfg["binning"]["rmin_mpc"] = 0.1

    if 'rmax_mpc' not in cfg['binning']:
        cfg["binning"]["rmax_mpc"] = 20.0

    if 'nbins' not in cfg['binning']:
        cfg["binning"]["nbins"] = 11

    # We use just rmin and rmax later
    cfg["binning"]["rmin"] = cfg["binning"]["rmin_mpc"]
    cfg["binning"]["rmax"] = cfg["binning"]["rmax_mpc"]

    if 'n_hist' not in cfg['selection']:
        cfg['selection']['n_hist'] = 100

    # Photometric redshift related parameters
    if 'lens_source_z_separation' not in cfg['photoz']:
        cfg['photoz']['lens_source_z_separation'] = 0.1

    if 'z_err_cut' not in cfg['photoz']:
        cfg['photoz']['z_err_cut'] = True

    if 'global_photo_z_cuts' not in cfg['photoz']:
        cfg['photoz']['global_photo_z_cuts'] = 'medium'

    if 'specinfo_photo_z_cuts' not in cfg['photoz']:
        cfg['photoz']['specinfo_photo_z_cuts'] = 'none'

    if 'field' not in cfg['photoz']:
        cfg['photoz']['field'] = 'frankenz_photoz_best'

    # Error of the photo-z
    # Note: when using Frankenz or other HSC photo-z catalog this is often
    #       the 1-sigma or 2-sigma lower boundary of the photo-z PDF.
    #       Not the uncertainty of photo-z itself
    #       Please make sure the `error_type` field reflects the nature
    #       of the uncertainty.
    if 'error_field' not in cfg['photoz']:
        cfg['photoz']['error_field'] = 'frankenz_photoz_err68_min'

    # Type of photo-z uncertainty
    #   z_low: error will be treated as the lower boundary of the photo-z PDF
    #   sigma: uncertainty of the photo-z
    if 'error_type' not in cfg['photoz']:
        cfg['photoz']['error_type'] = 'z_low'

    # How to convert the sigma into z_low
    #   z_low = z_photo - sigma / sigma_to_zlow
    if 'sigma_to_zlow' not in cfg['photoz']:
        cfg['photoz']['sigma_to_zlow'] = 2

    # Photometric redshift calibration
    if 'calib' not in cfg['photoz_calib']:
        cfg['photoz_calib']['calib'] = False

    if cfg['photoz_calib']['calib']:
        print('#    Include photo-z bias correction.')
    else:
        print("#    Do not include photo-z bias correction")

    if cfg['photoz_calib']['calib']:
        # Check the calibration catalog
        if not os.path.isfile(cfg['photoz_calib']['catalog']):
            raise IOError("# Can not find calibration catalog")

        if 'z_true' not in cfg['photoz_calib']:
            cfg['photoz_calib']['z_true'] = 'z_true'

        # If no z_weight is provided, just don't use z_weight
        if (('z_weight' not in cfg['photoz_calib']) or
                (cfg['photoz_calib']['z_weight'] == 'None') or
                ((cfg['photoz_calib']['z_weight']).strip() == '')):
            cfg['photoz_calib']['z_weight'] = None

    # Update information about redshift
    source_fields['z'] = cfg['photoz']['field']
    source_fields['z_err'] = cfg['photoz']['error_field']
    source_fields['z_err_type'] = cfg['photoz']['error_type']
    source_fields['z_err_factor'] = cfg['photoz']['sigma_to_zlow']

    # Check that error field is valid
    # Note: if no photo-z error field is provided, we will just treat the
    #       photo-z itsefl as a "lower boundary".
    #       In the photo-z cut later, this is basically: z_s > z_l
    if source_fields['z_err'] == "":
        source_fields['z_err_type'] = 'z_low'
        source_fields['z_err'] = source_fields['z']

    if source_fields['z_err'] != source_fields['z'] and \
       source_fields['z_err'] != "frankenz_photoz_err68_min" and \
       source_fields['z_err'] != "frankenz_photoz_err95_min":
        raise Exception("Invalid z_err option {}".format(source_fields['z_err']))

    if source_fields['z_err_type'] != "z_low" and \
       source_fields['z_err_type'] != "sigma":
        raise Exception("Invalid z_err_type option {}".format(source_fields['z_err_type']))

    cfg['source_fields'] = source_fields
    cfg['lens_fields'] = lens_fields

    return cfg


def config_computeds(config_file):
    """Prepare the configuration file for computeDS.

    Parameters
    ----------
    config_file : str
        Name of the configuration file.

    Return
    ------
    ds_cfg : dict
        Updated configuration file.
    """
    ds_cfg = yaml.load(open(config_file))

    # Primary pre-compute results for lenses
    if not os.path.isfile(ds_cfg['ds_lenses']):
        raise Exception("# Cannot find pre-compute result for lenses: %s" % ds_cfg['ds_lenses'])

    # Primary pre-compute results for randoms.  Optional.
    if 'ds_randoms' in ds_cfg:
        if not os.path.isfile(ds_cfg['ds_randoms']):
            raise Exception(
                "# Cannot find pre-compute result for randoms: %s" % ds_cfg['ds_randoms'])

    # Primary lens catalog
    if not os.path.isfile(ds_cfg['lens_catalog']):
        raise Exception("# Cannot find lens catalog: %s" % ds_cfg['lens_catalog'])

    # Secondary pre-compute results for lensese. Optional
    if 'ds_lenses_2' not in ds_cfg:
        ds_cfg['ds_lenses_2'] = None
    elif (ds_cfg['ds_lenses_2']).strip().lower() == 'none':
        ds_cfg['ds_lenses_2'] = None
    else:
        print("#    Will compare with DSigma profiles from: %s" % ds_cfg['ds_lenses_2'])
        if not os.path.isfile(ds_cfg['ds_lenses_2']):
            raise Exception(
                "# Cannot find pre-compute result for lenses: %s" % ds_cfg['ds_lenses_2'])

    # Secondary lens catalog. Optional.
    if 'lens_catalog_2' not in ds_cfg:
        # If no second lens catalog is provided, use the same lens catalog.
        ds_cfg['lens_catalog_2'] = ds_cfg['lens_catalog']
    elif (ds_cfg['lens_catalog_2']).strip().lower() == 'none':
        ds_cfg['lens_catalog_2'] = ds_cfg['lens_catalog']
    else:
        if not os.path.isfile(ds_cfg['lens_catalog_2']):
            raise Exception(
                "# Cannot find lens catalog: %s" % ds_cfg['lens_catalog_2'])

    # Secondary pre-compute results for randoms. Optional
    if 'ds_randoms_2' not in ds_cfg:
        ds_cfg['ds_randoms_2'] = None
    elif (ds_cfg['ds_randoms_2']).strip().lower() == 'none':
        ds_cfg['ds_randoms_2'] = None
    else:
        if not os.path.isfile(ds_cfg['ds_randoms_2']):
            raise Exception(
                "# Cannot find pre-compute result for randoms: %s" % ds_cfg['ds_randoms_2'])
        # Secondary rand catalog. Optional.
        if 'rand_catalog_2' not in ds_cfg:
            # If no second lens catalog is provided, use the same lens catalog.
            ds_cfg['rand_catalog_2'] = ds_cfg['rand_catalog']
        elif (ds_cfg['rand_catalog_2']).strip().lower() == 'none':
            ds_cfg['rand_catalog_2'] = ds_cfg['rand_catalog']
        else:
            if not os.path.isfile(ds_cfg['rand_catalog_2']):
                raise Exception(
                    "# Cannot find lens catalog: %s" % ds_cfg['rand_catalog_2'])

    # Lens weight
    if 'lens_weight' not in ds_cfg:
        ds_cfg['lens_weight'] = 'weight'
    print('#    Will use "%s" as lens weight' % ds_cfg['lens_weight'])

    # Secondary lens weight to compare with.
    if 'lens_weight_2' not in ds_cfg:
        ds_cfg['lens_weight_2'] = None
    elif (ds_cfg['lens_weight_2']).strip().lower() == 'none':
        ds_cfg['lens_weight_2'] = None
    else:
        print("#    Will compare with DSigma profiles using weight: %s" % ds_cfg['lens_weight_2'])

    # Assign default values for parameters:
    if 'output_prefix' not in ds_cfg:
        output_prefix, _ = os.path.splitext(ds_cfg['lens_catalog'])
        ds_cfg['output_prefix'] = output_prefix

    # Number of Jackknife fields
    if 'njackknife_fields' not in ds_cfg:
        ds_cfg['njackknife_fields'] = 41

    # Whether to apply boost factor correction
    if 'boost_factor' not in ds_cfg:
        ds_cfg['boost_factor'] = False

    if ds_cfg['boost_factor']:
        print("#    Will apply boost factor correction.")

    # Wether to apply R2 selection bias correction.
    if 'selection_bias' not in ds_cfg:
        ds_cfg['selection_bias'] = False

    if ds_cfg['selection_bias']:
        print("#    Will apply R2 selection bias correction.")

    if 'comoving' not in ds_cfg:
        ds_cfg['comoving'] = False

    # This is only useful when random catalog is used
    if 'rand_zweight_nbins' not in ds_cfg:
        ds_cfg['rand_zweight_nbins'] = 10

    # Use default 'weight' column or use the specified weight column for random
    if 'same_weight_rand' not in ds_cfg:
        ds_cfg['same_weight_rand'] = True

    # Number of processors to run on, mostly for the Jackknife error
    if 'n_jobs' not in ds_cfg:
        ds_cfg['n_jobs'] = 1

    # For estimating the covariance matrix of the DeltaSigma profile
    if 'covariance' not in ds_cfg:
        ds_cfg['covariance'] = {'boxsize': 1, 'trunc': 0.2, 'n_boots': 5000}
    else:
        if 'boxsize' not in ds_cfg['covariance']:
            ds_cfg['covariance']['boxsize'] = 1
        if 'trunc' not in ds_cfg['covariance']:
            ds_cfg['covariance']['trunc'] = 0.2
        if 'n_boots' not in ds_cfg['covariance']:
            ds_cfg['covariance']['n_boots'] = 10000

    # For estimating the average offset of the ratio between two DeltaSigma profiles.
    if 'ratios' not in ds_cfg:
        ds_cfg['ratios'] = {'diff': False, 'r1': 0.1, 'r2': 1.0, 'r3': 10.0}
    else:
        if 'diff' not in ds_cfg['ratios']:
            ds_cfg['ratios']['diff'] = False
        if 'r1' not in ds_cfg['ratios']:
            ds_cfg['ratios']['r1'] = 0.1
        if 'r2' not in ds_cfg['ratios']:
            ds_cfg['ratios']['r2'] = 1.0
        if 'r3' not in ds_cfg['ratios']:
            ds_cfg['ratios']['r3'] = 10.

    return ds_cfg
