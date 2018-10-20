# DeltaSigmaPipelineHSC

A collection of scripts forming a pipeline to measure DS with HSC catalogs.

Assumed coordinate from is (X,Y) = (ra,dec)

## Setup
```
git clone git@github.com:alexieleauthaud/DeltaSigmaPipelineHSC.git
make install_deps
```
Also make sure that you have the [data](#data) in the correct format.

## Pipeline scripts

### pre-process_source_catalog.py
* Run with `./pre-process_source_catalog.py`. **Makes assumptions about the locations of files - check the code**
* Takes the HSC weak lensing catalog and the photoz files and combines them into a single file including only the columns we care about.

### precomputeDS.py
* Run with`./precomputeDS.py <CONFIG_FILE> [-p parallization]`
    - `p` is the number of cores used in computation.
* Computes delta sigma for each of the lenses and saves a file with
  the DS per lens.
* Sample config files are provided, see [lenses](config/sample_lens_precompute.yaml), [randoms](config/sample_random_precompute.yaml).

Photoz quality cuts are called in precomputeDS but precomputeDS mainly
calls delta_sigma_catalog which is under delta_sigma.py for the bulk of the calculation.
* Global photoz quality cut options are `basic`, `medium`, and `strict`.
* Redshift error options used for lens-source separation are `frankenz_photoz_err95_min`, `frankenz_photoz_err68_min`, or the same as the field used for the redshift (i.e. no error). If not provided, this defaults to the field used for the redshift.
* Specinfo photoz cut options are `great` (top 1/3rd), `good` (top 2/3rds), `moderate` (middle 1/3rd), `poor` (lower 2/3rds), `poorest` (lowest 1/3rd), and `none` (no cut).

### computeDS.py
Once a precompute file has been made, run this to form DS for any lens selection
* Run with `./computeDS.py <CONFIG_FILE>`
* Combines the lenses selected by the optional mask, sets up jackknife fields and computes DS along with the errors.
* Sample config files are provided, see [compute config](config/sample_compute.yaml).
* Currently only plots the results but we probably want this to write to file.

## Data

You need the following data files.
* Lenses: https://www.dropbox.com/s/edf1tbyfg1h1m3b/s16a_fastlane_massive_short.npy?dl=0

If you are running on different data, make sure that the format of your data files (.npy, column headers) match these ones.
