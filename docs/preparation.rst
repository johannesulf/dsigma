Data Preparation
================

We need a lens and a source catalog to calculate the galaxy-galaxy lensing signal. Fortunately, many data sets are publicly available, such as data from HSC or CFHTLenS. Once we have obtained the data, we need to convert it into a format understandable by :code:`dsigma`.

Generally, :code:`dsigma` expects all lens, source, and calibration catalogs to be :code:`astropy` tables with data stored in specific, pre-defined columns.

Lens Catalog
------------

The following columns are required for lens catalogs.

* ``ra``: right ascension in degrees
* ``dec``: declination in degrees
* ``z``: best-fit redshift
* ``w_sys``: systematic weight :math:`w_{\mathrm{sys}}`

The weight :math:`w_{\mathrm{sys}}` is often used to mitigate systematics in the lens selection.

Source Catalog
--------------

The following columns are required for source catalogs.

* ``ra``: right ascension in degrees
* ``dec``: declination in degrees
* ``z``: best-fit photometric redshift
* ``w``: inverse variance weight for galaxy shape
* ``e_1``: + component of ellipticity
* ``e_2``: x component of ellipticity

Additionally, the following columns may be used in the analysis.

* ``m``: multiplicative shear bias
* ``e_rms``: root mean square ellipticity
* ``R_2``: HSC resolution factor (0=unresolved, 1=resolved)
* ``R_11``, ``R_22``, ``R_12``, ``R_21``: METACALIBRATION shear response
* ``z_bin``: tomographic redshift bin, non-negative and starts at 0

Calibration Catalog
-------------------

The following columns are required in (optional) calibration catalogs.

* ``z``: best-fit photometric redshift
* ``z_true``: "true" redshift
* ``w``: inverse variance weight for galaxy shape
* ``w_sys``: systematic weight :math:`w_{\mathrm{sys}}`

The weight :math:`w_{\mathrm{sys}}` is used to offset, for example, color differences between the source and the calibration catalog. Additionally, the columns ``z_low`` and ``z_err`` may also be present in the calibration catalog with the same meaning as in the source catalog.
