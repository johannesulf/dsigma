Data Preparation
================

We need a lens and a source catalog to calculate the galaxy-galaxy lensing signal. Several publicly available datasets are supported, as described in the Applications section. Once we have obtained the data, we need to convert it into a format understandable by ``dsigma``.

Generally, ``dsigma`` expects all lens, source, and calibration catalogs to be ``astropy`` tables with data stored in specific, pre-defined columns.

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
* ``m_sel``: selection bias
* ``e_rms``: root mean square ellipticity
* ``R_2``: HSC resolution factor (0=unresolved, 1=resolved)
* ``R_11``, ``R_22``, ``R_12``, ``R_21``: METACALIBRATION shear response
* ``z_bin``: tomographic redshift bin, non-negative and starts at 0
* ``z_l_max``: maximum lens redshift used for lens-source pairs

Redshift Distributions
----------------------

For most modern weak lensing surveys, sources are divided into tomographic bins. Each tomographic bins has a redshift distribution :math:`n(z)`. These can be specified as follows:

* ``z``: redshift grid
* ``n``: :math:`n(z)`, must be two-dimensional, i.e., the value for each bin at redshift :math:`z`

Calibration Catalog
-------------------

As an alternative to redshift distributions, photometric redshifts can be corrected using the :math:`f_{\rm bias}` correction, which requires a calibration catalog.

* ``z``: best-fit photometric redshift
* ``z_true``: "true" redshift
* ``w``: inverse variance weight for galaxy shape
* ``w_sys``: systematic weight :math:`w_{\mathrm{sys}}`

The weight :math:`w_{\mathrm{sys}}` is used to offset, for example, color differences between the source and the calibration catalog.
