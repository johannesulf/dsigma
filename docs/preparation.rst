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
* ``hsc_res``: HSC resolution factor (0=unresolved, 1=resolved)

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

Example: BOSS x HSC
-------------------

``dsigma`` comes with a convenience function, :code:`dsigma_table`, that
converts tables into formats understood by ``dsigma``. Let's assume we wanted
to calculate the lensing signal around BOSS LOWZ galaxies with HSC PDR2 source
galaxies. The following code is enough to prepare the data.

.. code-block:: python

    from astropy.table import Table
    from dsigma import helpers

    # https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_LOWZ_North.fits.gz
    table_l = Table.read('galaxy_DR12v5_LOWZ_North.fits')

    # https://hsc-release.mtk.nao.ac.jp/datasearch/
    table_s = Table.read('hsc_source_pdr2.fits')

    table_l = helpers.dsigma_table(table_l, 'lens', ra='RA', dec='DEC', z='Z',
                                   w_sys='WEIGHT_SYSTOT')
    table_s = helpers.dsigma_table(table_s, 'source', survey='HSC')

For the lens table, we have manually specified how are columns are named in
the input table. On the other hand, for the source table we have let
``dsigma`` use a set of default columns used in the HSC PDR2 shape catalogs.
The following output is generated to show which column assignments are used.

.. code-block:: none

    Assignment for lens table...
        z          -> Z
        w_sys      -> WEIGHT_SYSTOT
        ra         -> RA
        dec        -> DEC
    Assignment for source table...
        z          -> frankenz_photoz_best
        w          -> ishape_hsm_regauss_derived_shape_weight
        ra         -> ira
        dec        -> idec
        e_1        -> ishape_hsm_regauss_e1
        e_2        -> ishape_hsm_regauss_e2
        z_low      -> frankenz_photoz_err68_min
        m          -> ishape_hsm_regauss_derived_shear_bias_m
        sigma_rms  -> ishape_hsm_regauss_derived_rms_e
        hsc_res    -> ishape_hsm_regauss_resolution
