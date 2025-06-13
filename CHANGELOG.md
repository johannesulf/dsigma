# Changelog
Notable changes to dsigma will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The way multiplicative selection biases are handled was changed in order for the code to be more general. Specifically, the `hsc_selection_bias_correction` keyword argument in `dsigma.stacking.excess_surface_density` was replaced by the `selection_bias_correction` keyword argument. For this to work, a `m_sel` column needs to be added to the source table before the precomputation.
- 'Y3' is now the default version for HSC.

### Added

- The function `survey.hsc.multiplicative_selection_bias` to compute `m_sel` for HSC was added.
- `Y3` was added as a supported version to the `survey.hsc` functions.

### Removed

- The function `survey.hsc.apply_photo_z_quality_cut` was removed since it was rarely used and is only valid for Y1.

## [1.0.1] - 2025-06-09

### Fixed

- Fixed an incorrect data type that caused a crash. Thanks to @suqik for spotting this!

## [1.0.0] - 2024-05-26

### Added
- Added `dsigma.stacking.mean_critical_surface_density`.

### Changed
- The photometric redshift correction now always applied when computing the mean source redshift.
- `dsigma.stacking.lens_magnification_bias` now uses `dsigma.stacking.mean_critical_surface_density` to estimate the critical surface density. It does not calculate it based on the mean lens and source redshift, anymore.
- `dsigma.physics.lens_magnification_shear_bias` can now use angles expressed with `astropy` units.

## [0.7.2] - 2023-06-02

### Added

- `dsigma.stacking.lens_magnification_bias` can now be used to compute the bias in the tangential shear.

### Changed

- The mean source redshift now takes into account n(z)'s passed to `dsigma.precompute.precompute`.

### Fixed

- Fixed an incompatibility with numpy 1.24.
- Fixed a bug in `dsigma.stacking.tangential_shear` when `random_subtraction=True`.
- Fixed an error in tomographic redshift bin assignment for KiDS. Sources with photo-z's at the bin edges were assigned to the wrong tomographic bin. This biased KiDS lensing measurements by order 2%.

## [0.7.1] - 2023-01-18

### Changed

- `dsigma.jackknife.compress_jackknife_fields` now suppresses `numpy` warnings if columns contain NaNs.

### Fixed

- Fixed a bug in the calculation of the photo-z dilution correction factor. This led to percent-level biases in the total galaxy-galaxy lensing amplitude. It did not, however, affect DES and KiDS calculations since those are based on n(z)'s. The bug was introduced in version 0.6.

## [0.7.0] - 2023-01-06

### Changed

- `dsigma.precompute.add_precompute_results` has been renamed to `dsigma.precompute.precompute`.
- `dsigma.precompute.add_maximum_lens_redshift` has been removed and the functionality integrated into `dsigma.precompute.precompute` using the `lens_source_cut` argument.
- `dsigma.jackknife.add_continous_fields`, `dsigma.jackknife.transfer_continous_fields`, `dsigma.jackknife.jackknife_field_centers`, and `dsigma.jackknife.add_jackknife_fields` have been merged into a single function, `dsigma.jackknife.compute_jackknife_fields`.
- The computation of continuous fields to construct jackknife patches now uses DBSCAN instead of agglomerative clustering. Additionally, points are also not downsampled, anymore.

### Removed

- Removed `dsigma.stacking.shape_noise_error`. Please use jackknife resampling to estimate errors.

## [0.6.1] - 2023-01-01

### Changed

- Implemented significant performance improvements for `dsigma.precompute.add_precompute_results`.

### Fixed

- Fixed crashes in `dsigma.stacking.shape_noise_error`.
