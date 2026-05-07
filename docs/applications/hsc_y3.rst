Hyper Suprime-Cam (HSC)
=======================

.. note::
    This is an unofficial guide to using HSC data with ``dsigma``. It has not been reviewed by the HSC collaboration. For questions about the data products themselves, refer to the `official HSC documentation <https://hsc-release.mtk.nao.ac.jp/doc/index.php/s19a-shape-catalog-pdr3/>`_.

This page explains how to measure galaxy-galaxy lensing using the HSC Y3 shape catalog, also known as PDR3 or S19A.

Downloading the Data
--------------------

HSC Y3 catalog data is available from the HSC website after registering. Download the `flat files <https://hsc-release.mtk.nao.ac.jp/archive/filetree/shape_catalog_y3/catalog_obs_reGaus_public/>`_ and `n(z) file <https://hsc-release.mtk.nao.ac.jp/archive/filetree/shape_catalog_y3/li23/nz/nz.fits>`_. Then run :program:`dsigma-process-hsc-y3` (see :func:`~dsigma.scripts.process_hsc_y3.process_hsc_y3`) to process the raw files into a single ``hsc_y3.hdf5`` file used in the steps below.

Precomputing the Signal
-----------------------

We apply a lens-source separation cut of :math:`z_l < \langle z_t \rangle - 0.3`, where :math:`\langle z_t \rangle` is the mean redshift of the tomographic bin each source belongs to.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import Planck15
    from astropy.table import Table

    from dsigma.precompute import precompute

    table_s = Table.read('hsc_y3.hdf5', path='catalog')
    table_n = Table.read('hsc_y3.hdf5', path='calibration')
    table_s['z_l_max'] = np.sum(table_n['z'][:, np.newaxis] *
                          table_n['n'], axis=0)[table_s['z_bin']] - 0.3

    rp_bins = np.logspace(-1, 1.6, 14)
    kwargs = dict(cosmology=Planck15, comoving=True, table_n=table_n,
                  progress_bar=True)
    precompute(table_l, table_s, rp_bins, **kwargs)
    precompute(table_r, table_s, rp_bins, **kwargs)

Stacking the Signal
-------------------

We stack the signal in four BOSS redshift bins. Lenses and randoms with no nearby source galaxies are removed first. Jackknife resampling with 100 fields is used to estimate uncertainties.

We apply the responsivity correction, a scalar shear response correction, a selection bias correction, and subtract the signal around randoms. Random subtraction removes additive systematics, reduces noise, and is strongly recommended. We do not apply a boost correction, as our estimator may be biased for HSC.

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
                  shear_responsivity_correction=True,
                  selection_bias_correction=True,
                  random_subtraction=True)
    z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

    for lens_bin, (z_min, z_max) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        table_l_bin = table_l[(z_min <= table_l['z']) & (table_l['z'] < z_max)]
        table_r_bin = table_r[(z_min <= table_r['z']) & (table_r['z'] < z_max)]

        result = excess_surface_density(
            table_l_bin, table_r=table_r_bin, return_table=True, **kwargs)
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l_bin, table_r=table_r_bin, **kwargs)))

        result.write(f'hsc_{lens_bin}.csv', overwrite=True)

Acknowledgments
---------------

If you use HSC Y3 data in your research, please cite `Li et al. (2022) <https://doi.org/10.1093/pasj/psac006>`_.
