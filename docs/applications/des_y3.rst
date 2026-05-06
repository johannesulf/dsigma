Dark Energy Survey (DES)
========================

.. note::
    This is an unofficial guide to using DES data with ``dsigma``. It has not been reviewed by the DES collaboration. For questions about the data products themselves, refer to the `official DES documentation <https://des.ncsa.illinois.edu/releases/y3a2>`_.

Here, we will show how to work with the DES Y3 data release.

Downloading the Data
--------------------

DES Y3 catalog data is publicly available `here <https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files>`_. Download the required files with:

.. code-block:: none

    wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_sompz_v0.50.h5
    wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_metacal_v03-004.h5
    wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_indexcat.h5
    wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits

Then run :program:`dsigma-reduce-decade` (see :func:`dsigma.scripts.process_des_y3.main`) to process the raw files into a single ``des_y3.hdf5`` file used in the steps below.

Precomputing the Signal
-----------------------

We apply a lens-source separation cut of :math:`z_l + 0.1 < z_{t, \rm low}`, where :math:`z_{t, \rm low}` is the lower edge of the tomographic bin each source belongs to `(Myles et al., 2021) <https://doi.org/10.1093/mnras/stab1515>`_.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import Planck15
    from astropy.table import Table

    from dsigma.precompute import precompute
    
    table_s = Table.read('des_y3.hdf5', path='catalog')
    table_s['z'] = np.array([0.0, 0.358, 0.631, 0.872])[table_s['z_bin']]
    table_n = Table.read('des_y3.hdf5', path='calibration')

    rp_bins = np.logspace(-1, 1.6, 14)
    precompute(table_l, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_n=table_n, lens_source_cut=0.1, progress_bar=True)
    precompute(table_r, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_n=table_n, lens_source_cut=0.1, progress_bar=True)

Stacking the Signal
-------------------

We stack the signal in four BOSS redshift bins. Lenses and randoms with no nearby source galaxies are removed first. Jackknife resampling with 100 fields is used to estimate uncertainties.

We apply the METACALIBRATION matrix shear response correction, a scalar shear response correction (for blending) and subtract the signal around randoms. Random subtraction removes additive systematics, reduces noise, and is strongly recommended. We do not apply a boost correction, as our estimator may be biased for DES.

.. code-block:: python

    from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
    from dsigma.stacking import excess_surface_density

    # Drop all lenses and randoms that did not have any nearby source.
    table_l = table_l[np.sum(table_l['sum 1'], axis=1) > 0]
    table_r = table_r[np.sum(table_r['sum 1'], axis=1) > 0]

    centers = compute_jackknife_fields(
        table_l, 100, weights=np.sum(table_l['sum 1'], axis=1))
    compute_jackknife_fields(table_r, centers)

    z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

    for lens_bin in range(len(z_bins) - 1):
        use_l = ((z_bins[lens_bin] <= table_l['z']) &
                 (table_l['z'] < z_bins[lens_bin + 1]))
        use_r = ((z_bins[lens_bin] <= table_r['z']) &
                 (table_r['z'] < z_bins[lens_bin + 1]))

        kwargs = dict(return_table=True, scalar_shear_response_correction=True,
                      matrix_shear_response_correction=True,
                      random_subtraction=True, table_r=table_r[use_r])

        result = excess_surface_density(table_l[use_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l[use_l], **kwargs)))

        result.write(f'des_{lens_bin}.csv', overwrite=True)

Acknowledgments
---------------

If you use DES Y3 data in your research, please follow the acknowledgment guidelines on the `DES Y3 data release page <https://des.ncsa.illinois.edu/releases/y3a2>`_.
