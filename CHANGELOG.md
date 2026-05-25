# Changelog
Notable changes to dsigma will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for DECADE and KiDS-Legacy.
- DES Y3, HSC Y3, KiDS-Legacy, and DECADE data reduction scripts under `dsigma.scripts`. These scripts now encode all survey-specific functionality.
- Helper function `dsigma.helpers.interpolate_over_redshift` as part of a code refactoring.

### Changed

- `dsigma` now consistently supports `astropy` units. Input parameters and output results, including $\Delta\Sigma$, now have units. For inputs, default units are $\mathrm{Mpc}/h$ (if no unit is specified) and $\Delta\Sigma$ is given in $h M_\odot / \mathrm{pc}^2$. Note that previously, $\Delta\Sigma$ was returned in units of $M_\odot / \mathrm{pc}^2$, i.e., without little $h$.
- The default cosmology was changed from `astropy.cosmology.FlatLambdaCDM(H0=100, Om0=0.3)` to `astropy.cosmology.Planck15`. While this changes $h=1$ to $h = 0.6774$, this only leads to small differences in the numerical results since `dsigma` now consistently uses little $h$ units. Also, the default cosmology can be set by changing `dsigma.default_cosmology`.
- In the results table of `dsigma.stacking.tangential_shear`, the `et` and `et_raw` columns were renamed to `gt` and `gt_raw`.
- Instead of taking an instance of `camb.results.CAMBdata` as input, `dsigma.physics.lens_magnification_shear_bias` and `dsigma.physics.lens_magnification_bias` now convert between `astropy` and `camb`, internally. The user only needs to provide `sigma_8` and `n_s`.
- Replaced `dsigma.jackknife.smooth_correlation_matrix` with `dsigma.jackknife.smooth_covariance_matrix`.
- The C engine has been refactored and received a slight performance tweak.

### Removed

- `dsigma.surveys` in favor of `dsigma.scripts`.
- The `lens_source_cut` keyword argument. Please specify the maximum lens redshift via the `z_l_max` column.

## [1.1.0] - 2025-06-25

### Added

- The function `survey.hsc.multiplicative_selection_bias` to compute `m_sel` for HSC.
- `Y3` was added as a supported version to the `survey.hsc` functions.

### Changed

- The way multiplicative selection biases are handled was changed in order for the code to be more general. Specifically, the `hsc_selection_bias_correction` keyword argument in `dsigma.stacking.excess_surface_density` was replaced by the `selection_bias_correction` keyword argument. For this to work, a `m_sel` column needs to be added to the source table before the precomputation.
- 'Y3' is now the default version for HSC.

### Fixed

- The HSC Y1 selection bias was applied incorrectly. Instead of dividing by `1+m_sel`, the result was multiplied by `1+m_sel`. This bug goes back to the overhaul of `dsigma` in 2020. Fortunately, this only caused an error at the level of around 2%, well below statistical significance of actual measurements.

### Removed

- The function `survey.hsc.apply_photo_z_quality_cut` was removed since it was rarely used and is only valid for Y1.

## [1.0.1] - 2025-06-09

### Fixed

- Fixed an incorrect data type that caused a crash. Thanks to @suqik for spotting this!

## [1.0.0] - 2024-05-26

### Added
- `dsigma.stacking.mean_critical_surface_density`.

### Changed
- The photometric redshift correction now always applied when computing the mean source redshift.
- `dsigma.stacking.lens_magnification_bias` now uses `dsigma.stacking.mean_critical_surface_density` to estimate the critical surface density instead of estimating it from the mean lens and source redshift.
- `dsigma.physics.lens_magnification_shear_bias` can now use angles expressed with `astropy` units.

## [0.7.2] - 2023-06-02

### Added

- `dsigma.stacking.lens_magnification_bias` can now be used to compute the bias in the tangential shear.

### Changed

- The mean source redshift now takes into account $n(z)$'s passed to `dsigma.precompute.precompute`.

### Fixed

- Fixed an incompatibility with numpy 1.24.
- Fixed a bug in `dsigma.stacking.tangential_shear` when `random_subtraction=True`.
- Fixed an error in tomographic redshift bin assignment for KiDS. Sources with photo-z's at the bin edges were assigned to the wrong tomographic bin. This biased KiDS lensing measurements by order 2%.

## [0.7.1] - 2023-01-18

### Changed

- `dsigma.jackknife.compress_jackknife_fields` now suppresses `numpy` warnings if columns contain NaNs.

### Fixed

- Fixed a bug in the calculation of the photo-z dilution correction factor. This led to percent-level biases in the total galaxy-galaxy lensing amplitude. It did not, however, affect DES and KiDS calculations since those are based on $n(z)$'s. The bug was introduced in version 0.6.

## [0.7.0] - 2023-01-06

### Changed

- `dsigma.precompute.add_precompute_results` has been renamed to `dsigma.precompute.precompute`.
- `dsigma.precompute.add_maximum_lens_redshift` has been removed and the functionality integrated into `dsigma.precompute.precompute` using the `lens_source_cut` argument.
- `dsigma.jackknife.add_continous_fields`, `dsigma.jackknife.transfer_continous_fields`, `dsigma.jackknife.jackknife_field_centers`, and `dsigma.jackknife.add_jackknife_fields` have been merged into a single function, `dsigma.jackknife.compute_jackknife_fields`.
- The computation of continuous fields to construct jackknife patches now uses DBSCAN instead of agglomerative clustering. Additionally, points are also not downsampled, anymore.

### Removed

- `dsigma.stacking.shape_noise_error`. Please use jackknife resampling to estimate errors.

## [0.6.1] - 2023-01-01

### Changed

- Implemented significant performance improvements for `dsigma.precompute.add_precompute_results`.

### Fixed

- Crashes in `dsigma.stacking.shape_noise_error`.
