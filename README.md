# `dsigma`: A simple pure-python galaxy-galaxy lensing pipeline



# Setup

```
git clone git@github.com:alexieleauthaud/DeltaSigmaPipelineHSC.git
make install_deps
```
Also make sure that you have the [data](#data) in the correct format.

----

# Prepare the input data

----

# Pre-compute for each lens-source pair: `runPrecomputeDS.py`

----

# Forming the final DeltaSigma signal: `runComputeDS.py`

* Basic usage: 
    - `runComputeDS.py config.yaml` 
    - `runCompuveCov.py config.yaml`

* Once the pre-computed results for lenses (and randoms) are ready, you can use the `computeDS.py` and `computeCov.py` scripts to form the final DeltaSigma signal and its uncertainty. These scripts also provide the possibility to test the ratio of DeltaSigma signals between two sets of pre-computed results or using two different weights for lenses (and randoms). 
* These two scripts can use the same configuration file. The differences are:
    1. `runComputeDS.py` will form the final DeltaSigma profile of the selected lenses and estimate its uncertainties using just the shape noise or using Jackknife resampling. 
    2. `runComputeCov.py` will first gather all the factors to form the DeltaSigma profile in each of the Jackknife regions. Then it will bootstrap resampling all the Jackknife regions and form DeltaSigma profiles for all the bootstrapped samples.  It will use such a sample to estimate a mean DeltaSigma profile and will use it to estimate the covariance matrix of the DeltaSigma profile. 
* When comparing two DeltaSigma profiles, there are two options:
    1. Based on the same pre-computed results for lenses (and randoms), but use two different lens weights in the lens (and random) catalogs. In this case, you need to inform the code the column name of the second lens (or random) weight: `lens_weight_2`. 
    2. Based on two sets of pre-computed results.  In this case, you need to inform the code the files for the second set of results with a `_2` suffix. (`ds_lenses_2`, `ds_randoms_2`, `lens_catalog_2`, `rand_catalog_2`, `mask_args_2`)

---- 

## Configuration File

Here is the explanation for the configuration parameters. An example file can be found in `config/sample_compute.yaml`. 

### Input files

* `ds_lenses`: The pre-computed result for lenses in `.npz` format.  The output of the `runPrecompute.py` process.
* `lens_catalog`: Lens catalog in `.npy` format.
* `ds_randoms`: [Optional] The pre-computed result for randoms.
* `rand_catalog`: [Optional; but should come together with `ds_randoms`] Random catalogs in `.npy` format.
* For the second set of pre-computed results, add a `_2` suffix to the keyname.
     - Please make sure the two pre-computed results share the same radial bins and the same cosmology.
     - If `lens_catalog_2` (`rand_catalog_2`) is not available, will use `lens_catalog` (`rand_catalog`). 

### Lens selection

* `mask_args` and `mask_args_2`: [Optional] String that describes the criteria to select useful lenses.  Example: `logm_10/>/11:logm_10/</12`. Default: None
* `external_mask`: [Optional] Boolen mask array stored in `.npy` format.  Default: None
* These two selections can be applied at the same time.

### Lens weight

* `lens_weight`: [Optional] Column name of the lens weight in the lens catalog. Default: `weight`
* `lens_weight_2`: [Optional] Column name of the second lens weight to compare with.
* `rand_zweight_nbins`: [Optional] Number of redshift bins to be used to re-weight the redshift of random objects, so that the lenses and randoms can share the same redshift distribution. Default: 10
* `same_weight_random`: [Optional] Whether the random catalog has the same weight information. When not available, only will use the default `weight` column in the random catalog.  Default: True
    - For example, if your lens and random catalogs both have a weight for PSF: `wpsf`.  Set `same_weight_random: True` will make sure both lenses and randoms are weighted by `wpsf`.  In some cases, the randoms do not have the same weight information as in the lens catalog.

### Others

* `output_prefix`: [Optional] Name the output results.
* `njackknife_fields`: [Optional] Number of the Jackknife regions to be assigned to lenses (and randoms).  Default: 41
* `selection_bias`: [Optional] Whether to include the `R2` selection bias in the signal.  Please see Mandelbaum et al. (2018) for more details. Normally speaking, the impact of this bias is tiny (<1% level). Default: False
* `boost_factor`: [Optional] Whether to apply the boost factor correction. Default: False
* `n_jobs`: [Optional] Number of processors to run on.  Require the `joblib` library. Default: 1
    - This only speeds things up a little for forming the DeltaSigma signal in each of the Jackknife regions.

### Ratio of two DeltaSigma profiles

* When second weight column is present, or when the second set of pre-computed results are available, the script `runComputeCov.py` will try to estimate the average ratio in three different radius ranges defined by `[r1, r2]`, `[r2, r3]`, and `[r1, r3]`. The code fits the average ratio taking the covariance matrix into account.  Default: 

```
ratios:
    r1: 0.1
    r2: 1.0
    r3: 10.0
```

### Bootstrap resampling. 

* `covariance`: Only useful for `runComputeCov.py` when computing the covariance matrix using bootstrap resampling method.  
* `n_boots`: Number of bootstrap samples to use. Default: 5000
* Example: 
```
covariance: 
    n_boots: 5000
```

----

## Outputs

* Assume `output_prefix: massive`

### `runComputeDS.py`

* A summary of the DeltaSigma profiles for the primary pre-computed result: `massive_dsigma.npz` or `massive_dsigma_with_random.npz` when a random catalog is available.  The data available in this compressed file are:
    - `dsigma_output`: structured array as a summary of the DeltaSigma results. Will describe later. 
    - `cosmology`: A dictionary of key cosmology parameters. 
    - `config`: A dictionary for all the configuration parameters. 
    - `jackknife_samples`: The DeltaSigma profiles in each Jackknife regions.
    - `lens_weights` and `rand_weights`: Final weights for lenses and randoms.  (For debugging, will be removed later).

* When comparing with a second lens weight, the results will be in `massive_dsigma_wratio.npz`:
    - `rad`: Array for radial bin centers, in the unit of Mpc.
    - `cosmology`: A dictionary of key cosmology parameters. 
    - `config`: A dictionary for all the configuration parameters. 
    - `diff_avg`: The average ratio between the two DeltaSigma profiles. 
    - `diff_var`: The uncertainty of the ratio. 
    - `dsig_1` and `dsig_2`: The DeltaSigma profiles using `lens_weight` and `lens_weight_2` in each of the Jackknife regions. The ratio is in the format of `dsig_2 / dsig_1`. 
    - `diff_arr`: The ratios of two DeltaSigma profiles in each of the Jackknife region.

* When comparing with the second set of pre-computed results, the summary file is `massive_dsigma_ratio.npz`:
    - The output format is the same with `massive_dsigma_wratio.npz`.
    
### `runComputeCov.py`

* The summary of the DeltaSigma profiles for the first set of pre-computed results: `massive_ds_cov.npz`. Available data are: 
    - `r_mpc`: Radial bin centers in the unit of Mpc.
    - `config`: Configuration parameters in a dictionary. 
    - `delta_sigma`: The stacked DeltaSigma profile for all lenses.
    - `disgma_boot`: The DeltaSigma profiles for each of the bootstrap samples.
    - `cov_trunc`: Covariance matrix for the DeltaSigma profile.

* When comparing with another lens weight.  The output is `massive_ds_cov_wratio.npz`:
    - `r_mpc`: Radial bin centers in the unit of Mpc.
    - `config`: Configuration parameters in a dictionary. 
    - `dsigma_1` and `dsigma_2`: The stacked DeltaSigma profile for all lenses using `lens_weight` and `lens_weight_2`.
    - `ratio_boot`: The ratio of DeltaSigma profiles using different weights for each of the bootstrap samples.
    - `avg_ratio`: The average ratio of two profiles.
    - `err_ratio`: The uncertainty of the ratio using the diagonal term of the covariance matrix. 
    - `cov_trunc`: Covariance matrix for the ratio of the DeltaSigma profile.

* When comparing with the second set of pre-computed results.  The output is `massive_ds_cov_ratio.npz`:
    - The format is the same as in the `massive_ds_cov_wratio.npz` one.
