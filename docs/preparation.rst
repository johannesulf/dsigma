Data Preparation
================

To calculate the galaxy-galaxy lensing signal, we need a lens and a source
catalog. Fortunately, many data sets are publicly available, such as data from
HSC or CFHTLenS. Once we have obtained the data, we need to convert it into
a format that's understandable by ``dsigma``.

Generally, ``dsigma`` expects all lens, source and calibration catalogs to be
``astropy`` tables with data stored in specific, pre-defined columns.

Lens Catalog
------------

The following columns are required for lens catalogs.

* ``ra``: right ascension
* ``dec``: declination
* ``z``: best-fit redshift
* ``w_sys``: systematic weight :math:`w_{\mathrm{sys}}`

The weight :math:`w_{\mathrm{sys}}` is often used to mitigate systematics in
the lens selection.

Source Catalog
--------------

The following columns are required for source catalogs.

* ``ra``: right ascension
* ``dec``: declination
* ``z``: best-fit photometric redshift
* ``w``: inverse variance weight for galaxy shape
* ``e_1``: + component of ellipticity
* ``e_2``: x component of ellipticity

Additionally, the following columns may be used in the analysis.

* ``z_low``: lower limit on the photometric redshift
* ``z_err``: uncertainty on the photometric redshift
* ``m``: multiplicative shear bias
* ``sigma_rms``: root mean square ellipticity
* ``R_2``: HSC resolution factor (0=unresolved, 1=resolved)

Calibration Catalog
-------------------

The following columns are required in (optional) calibration catalogs.

* ``z``: best-fit photometric redshift
* ``z_true``: "true" redshift
* ``w``: inverse variance weight for galaxy shape
* ``w_sys``: systematic weight :math:`w_{\mathrm{sys}}`

The weight :math:`w_{\mathrm{sys}}` is used to offset, for example, color
differences between the source and the calibration catalog. Additionally, the
columns ``z_low`` and ``z_err`` may also be present in the calibration catalog
with the same meaning as in the source catalog.
