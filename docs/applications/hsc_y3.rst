Hyper Suprime-Cam (HSC)
=======================

.. note::
    This guide has not been inspected or endorsed by the HSC collaboration.

This tutorial will teach us how to cross-correlate BOSS lens galaxies with lensing catalogs from the HSC survey. We will work with the Y3 data release, also known as PDR3 or S19A. The guide for the older Y1 release is :doc:`here <hsc_y1>`.

Downloading the Data
--------------------

First, we need to download the necessary HSC data files. Head to the `HSC data release site <https://hsc-release.mtk.nao.ac.jp/doc/>`_ and register for an account if you haven't done so already.

We follow the HSC recommendation and use the following SQL command to download all necessary data from the `CAS search <https://hsc-release.mtk.nao.ac.jp/datasearch/>`_. Make sure to select PDR3 as the release; otherwise, the SQL command will fail. Also, choose FITS as the output format.

.. code-block:: sql

    select
    b.*, c.i_ra, c.i_dec, a.i_hsmshaperegauss_e1, a.i_hsmshaperegauss_e2
    from
    s19a_wide.meas2 a
    inner join s19a_wide.weaklensing_hsm_regauss b using (object_id)
    inner join s19a_wide.meas c using (object_id)

In the following, this tutorial assumes that the above source catalog is saved as :code:`hsc_y3.fits` in the working directory.

In addition to the source catalog, we need the redshift distribution, :math:`n(z)`, for all samples which can be downloaded `here <https://hsc-release.mtk.nao.ac.jp/archive/filetree/shape_catalog_y3/li23/nz/nz.fits>`_.

Preparing the Data
------------------

First, we must put the data into a format easily understandable by :code:`dsigma`. There are several helper functions to make this easy.

.. code-block:: python

    from astropy.table import Table
    from dsigma.helpers import dsigma_table
    from dsigma.surveys import hsc

    table_s = Table.read('hsc_y3.fits')
    # Remove regions with large B-modes.
    table_s = table_s[table_s['b_mode_mask'] == 1]
    table_s = dsigma_table(table_s, 'source', survey='HSC')
    table_s['m_sel'] = hsc.multiplicative_selection_bias(table_s)
    # Remove galaxies with bimodal P(z)'s.
    table_s = table_s[table_s['z_bin'] > 0]
    # dsigma expects the first redshift bin to be 0, not 1.
    table_s['z_bin'] = table_s['z_bin'] - 1

    table_n = Table.read('nz.fits')
    # Create the columns expected by dsigma.
    table_n.rename_column('Z_MID', 'z')
    table_n['n'] = np.column_stack([table_n[f'BIN{i+1}'] for i in range(4)])
    table_n.keep_columns(['z', 'n'])

Precomputing the Signal
-----------------------

We will now run the computationally expensive precomputation phase. For the lens-source separation, we require :math:`z_l + 0.3 < z_s` where :math:`z_s` is the mean redshift of the tomographic bin each source belongs to. This ensures that we don't stack the lensing signal for souces that are mostly in front of the lens.

.. code-block:: python

    import numpy as np

    from astropy.cosmology import Planck15
    from dsigma.precompute import precompute

    # Assign each galaxy in the source catalog the mean redshift of the bin. This
    # is only used to determine which lens-source pairs to  use.
    table_s['z'] = np.sum(table_n['z'][:, np.newaxis] *
                          table_n['n'], axis=0)[table_s['z_bin']]

    rp_bins = np.logspace(-1, 1.6, 14)
    precompute(table_l, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_n=table_n, lens_source_cut=0.3, progress_bar=True)
    precompute(table_r, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_n=table_n, lens_source_cut=0.3, progress_bar=True)

Stacking the Signal
-------------------

The total galaxy-galaxy lensing signal can be obtained with the following code. It first filters out all BOSS galaxies for which we couldn't find any source galaxy nearby. Then we divide it into jackknife samples that we will later use to estimate uncertainties. Finally, we stack the lensing signal in 4 different BOSS redshift bins and save the data.

We choose to include all the necessary corrections factors. The shear responsivity correction and multiplicative shear correction are the most important and necessary. The selection bias corrections do not dramatically impact the signal but are also required for HSC data. Finally, random subtraction is also highly recommended, especially to mitigate additive shear biases. Note that we don't use a boost correction, but this would also be possible.

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
        mask_l = ((z_bins[lens_bin] <= table_l['z']) &
                  (table_l['z'] < z_bins[lens_bin + 1]))
        mask_r = ((z_bins[lens_bin] <= table_r['z']) &
                  (table_r['z'] < z_bins[lens_bin + 1]))

        kwargs = {'return_table': True,
                  'scalar_shear_response_correction': True,
                  'shear_responsivity_correction': True,
                  'selection_bias_correction': True,
                  'boost_correction': False, 'random_subtraction': True,
                  'table_r': table_r[mask_r]}

        result = excess_surface_density(table_l[mask_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l[mask_l], **kwargs)))

        result.write('hsc_{}.csv'.format(lens_bin))