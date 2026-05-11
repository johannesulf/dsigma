Kilo-Degree Survey (KiDS)
=========================

.. note::
    This is an unofficial guide to using KiDS data with ``dsigma``. It has not been reviewed by the KiDS collaboration. For questions about the data products themselves, refer to the `official KiDS documentation <https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php>`_.

This page explains how to measure galaxy-galaxy lensing using the KiDS-Legacy shape catalog.

Downloading the Data
--------------------

KiDS-Legacy catalogs are publicly available `here <https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php>`_. Download the required files with:

.. code-block:: bash

    BASE_URL=https://kids.strw.leidenuniv.nl/DR5/data_files
    wget $BASE_URL/{KiDS_Legacy_NS_unblind_final.fits.gz,KiDZ_Legacy_unblind_final.fits}

Then run :program:`dsigma-process-kids-legacy` (see :func:`~dsigma.scripts.process_kids_legacy.process_kids_legacy`) to process the raw files into a single ``kids_legacy.hdf5`` file used in the steps below.

Precomputing the Signal
-----------------------

We apply a lens-source separation cut of :math:`z_l < z_{t, \rm low} - 0.1`, where :math:`z_{t, \rm low}` is the lower edge of the tomographic bin each source belongs to `(Wright et al., 2026) <https://doi.org/10.1051/0004-6361/202554909>`_.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import Planck15
    from astropy.table import Table

    from dsigma.precompute import precompute

    table_s = Table.read('kids_legacy.hdf5', path='catalog')
    table_s['z_l_max'] = np.array([0.1, 0.42, 0.58, 0.71, 0.90, 1.14])[
        table_s['z_bin']] - 0.1
    table_n = Table.read('kids_legacy.hdf5', path='calibration')

    rp_bins = np.logspace(-1, 1.6, 14)
    kwargs = dict(cosmology=Planck15, comoving=True, table_n=table_n,
                  progress_bar=True)
    precompute(table_l, table_s, rp_bins, **kwargs)
    precompute(table_r, table_s, rp_bins, **kwargs)

Stacking the Signal
-------------------

We stack the signal in four BOSS redshift bins. Lenses and randoms with no nearby source galaxies are removed first. Jackknife resampling with 100 fields is used to estimate uncertainties.

We apply a scalar shear response correction and subtract the signal around randoms. Random subtraction removes additive systematics, reduces noise, and is strongly recommended. We do not apply a boost correction, as our estimator may be biased for KiDS.

.. code-block:: python

    import numpy as np

    from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
    from dsigma.stacking import excess_surface_density

    # Drop all lenses and randoms that did not have any nearby source.
    table_l = table_l[np.sum(table_l['sum 1'], axis=1) > 0]
    table_r = table_r[np.sum(table_r['sum 1'], axis=1) > 0]

    centers = compute_jackknife_fields(
        table_l, 100, weights=np.sum(table_l['sum 1'], axis=1))
    compute_jackknife_fields(table_r, centers)

    kwargs = dict(scalar_shear_response_correction=True,
                  random_subtraction=True)
    z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

    for lens_bin, (z_min, z_max) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        table_l_bin = table_l[(z_min <= table_l['z']) & (table_l['z'] < z_max)]
        table_r_bin = table_r[(z_min <= table_r['z']) & (table_r['z'] < z_max)]

        result = excess_surface_density(
            table_l_bin, table_r=table_r_bin, return_table=True, **kwargs)
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l_bin, table_r=table_r_bin, **kwargs)))

        result.write(f'kids_{lens_bin}.csv', overwrite=True)

Acknowledgments
---------------

If you use KiDS-Legacy data in your research, please follow the `KiDS-Legacy acknowledgment guidelines <https://kids.strw.leidenuniv.nl/DR5/acknowledgments.php>`_.
