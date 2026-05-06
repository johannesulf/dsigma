DECADE Shape Catalog
====================

.. note::
    This is an unofficial guide to using DECADE data with ``dsigma``. It has not been reviewed by the members of the DECADE campaign. For questions about the data products themselves, refer to the `DECADE documentation <https://dhayaaanbajagane.github.io/data_release/decade/>`_.

This page explains how to measure galaxy-galaxy lensing using the Dark Energy Camera All Data Everywhere (DECADE) shape catalog.

Downloading the Data
--------------------

A curated DECADE catalog, ``shear_catalog_sparse.hdf5`` is publicly available via `Globus <https://app.globus.org/file-manager?origin_id=0b5ae9e1-9026-4b94-bc29-ff5f51f94671>`_. You can also download the :math:`n(z)` files ``NGC_n_of_z.npy``, ``SGC_n_of_z.npy``, and ``z_grid.npy`` from the same location.

Then run :program:`dsigma-process-decade` (see :func:`~dsigma.scripts.process_decade.process_decade`) to process the raw files into two files: ``decade_ngc.hdf5`` and ``decade_sgc.hdf5``, corresponding to DECADE data in the different galactic hemispheres, each having slightly different selection responses and :math:`n(z)`'s. In the following, we will only use the NGC data.

Precomputing the Signal
-----------------------

We apply a lens-source separation cut of :math:`z_l + 0.1 < z_{t, \rm low}`, where :math:`z_{t, \rm low}` is the lower edge of the tomographic bin each source belongs to `(Anbajagane et al., 2025) <https://doi.org/10.33232/001c.146159>`_.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import Planck15
    from astropy.table import Table

    from dsigma.precompute import precompute
    
    table_s = Table.read('decade_ngc.hdf5', path='catalog')
    table_s['z'] = np.array([0.0, 0.381, 0.619, 0.803])[table_s['z_bin']]
    table_n = Table.read('decade_ngc.hdf5', path='calibration')

    rp_bins = np.logspace(-1, 1.6, 14)
    kwargs = dict(cosmology=Planck15, comoving=True, table_n=table_n,
                  lens_source_cut=0.1, progress_bar=True)
    precompute(table_l, table_s, rp_bins, **kwargs)
    precompute(table_r, table_s, rp_bins, **kwargs)

Stacking the Signal
-------------------

We stack the signal in four BOSS redshift bins. Lenses and randoms with no nearby source galaxies are removed first. Jackknife resampling with 100 fields is used to estimate uncertainties.

We apply the METACALIBRATION matrix shear response correction, a scalar shear response correction which accounts for blending, and subtract the signal around randoms. Random subtraction removes additive systematics, reduces noise, and is strongly recommended. We do not apply a boost correction, as our estimator may be biased for DECADE.

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
                  matrix_shear_response_correction=True,
                  random_subtraction=True)
    z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

    for lens_bin, (z_min, z_max) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        table_l_bin = table_l[(z_min <= table_l['z']) & (table_l['z'] < z_max)]
        table_r_bin = table_r[(z_min <= table_r['z']) & (table_r['z'] < z_max)]

        result = excess_surface_density(
            table_l_bin, table_r=table_r_bin, return_table=True, **kwargs)
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l_bin, table_r=table_r_bin, **kwargs)))

        result.write(f'decade_{lens_bin}.csv', overwrite=True)

Acknowledgments
---------------

If you use DECADE data in your research, please acknowledge the appropriate publications listed on `DECADE Cosmic Shear Project website <https://dhayaaanbajagane.github.io/projects/DECADE/>`_.
