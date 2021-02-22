Hyper Suprime-Cam (HSC)
=======================

In this tutorial, we will learn how to cross-correlate BOSS lens galaxies with
shape catalogs from the HSC survey.

Downloading the Data
--------------------

First, we need to download the necessary HSC data files. Head over to the
`HSC data release site <https://hsc-release.mtk.nao.ac.jp/doc/>`_ and register
for an account if you haven't done so already. As of September 2020, the only
publicly available data set is part of the public data release 2 (PDR2) and
goes back to the internal S16A release.

The following SQL script will return all the necessary data from the
`CAS search <https://hsc-release.mtk.nao.ac.jp/datasearch/>`_. ::

    SELECT
        meas.object_id, meas.ira, meas.idec,
        meas2.ishape_hsm_regauss_e1, meas2.ishape_hsm_regauss_e2,
        meas2.ishape_hsm_regauss_resolution,
        photoz_ephor_ab.photoz_best, photoz_ephor_ab.photoz_err68_min,
        photoz_ephor_ab.photoz_risk_best, photoz_ephor_ab.photoz_std_best,
        weaklensing_hsm_regauss.ishape_hsm_regauss_derived_shape_weight,
        weaklensing_hsm_regauss.ishape_hsm_regauss_derived_shear_bias_m,
        weaklensing_hsm_regauss.ishape_hsm_regauss_derived_rms_e

    FROM
        s16a_wide.weaklensing_hsm_regauss
        LEFT JOIN s16a_wide.meas USING (object_id)
        LEFT JOIN s16a_wide.meas2 USING (object_id)
    	LEFT JOIN s16a_wide.photoz_ephor_ab USING (object_id)

    ORDER BY meas.object_id

As you can see, we will use the Ephor Afterburner photometric redshifts in
this application. But you're free to use any other photometric redshift
estimate available in the HSC database. Also note that we neglect additive
shear biases which is fine because we will use random catalogs to correct for
those. In the following, this tutorial assumes that the above source catalog
is saved as :code:`hsc_sources.fits` in the working directory.

In addition to the source catalog, we need a calibration catalog to correct
for eventual biases stemming from using shallow photometric redshift point
estimates. The relevant files can be downloaded using the following links:
`1 <https://hsc-release.mtk.nao.ac.jp/archive/filetree/
cosmos_photoz_catalog_reweighted_to_s16a_shape_catalog/
Afterburner_reweighted_COSMOS_photoz_FDFC.fits>`_,
`2 <https://hsc-release.mtk.nao.ac.jp/archive/filetree/
cosmos_photoz_catalog_reweighted_to_s16a_shape_catalog/
ephor_ab/pdf-s17a_wide-9812.cat.fits>`_,
`3 <https://hsc-release.mtk.nao.ac.jp/archive/filetree/
cosmos_photoz_catalog_reweighted_to_s16a_shape_catalog/
ephor_ab/pdf-s17a_wide-9813.cat.fits>`_.

Preparing the Data
------------------

First, we must put the data into a format easily understandable by
:code:`dsigma`. There are a number of helper functions to make this easy. ::

    import numpy as np
    from astropy import units as u
    from astropy.table import Table, vstack, join
    from dsigma.helpers import dsigma_table

    table_s = Table.read('hsc_sources.fits')
    table_s = dsigma_table(table_s, 'source', survey='HSC')

    table_c_1 = vstack([Table.read('pdf-s17a_wide-9812.cat.fits'),
                        Table.read('pdf-s17a_wide-9813.cat.fits')])
    for key in table_c_1.colnames:
        table_c_1.rename_column(key, key.lower())
    table_c_2 = Table.read('Afterburner_reweighted_COSMOS_photoz_FDFC.fits')
    table_c_2.rename_column('S17a_objid', 'id')
    table_c = join(table_c_1, table_c_2, keys='id')
    table_c = dsigma_table(table_c, 'calibration', w_sys='SOM_weight',
                           w='weight_source', z_true='COSMOS_photoz', survey='HSC')

Pre-Computing the Signal
------------------------

We will now run the computationally expensive pre-computation phase. Here,
we first define the lens-source separation cuts. We require that
:math:`z_l < z_{s, \rm min}` and :math:`z_l + 0.1 < z_s`. Afterwards, we run
the actual pre-computation. ::

    from astropy.cosmology import Planck15
    from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog

    add_maximum_lens_redshift(table_s, dz_min=0.1, apply_z_low=True)
    add_maximum_lens_redshift(table_c, dz_min=0.1, apply_z_low=True)

    rp_bins = np.logspace(-1, 1.6, 14)
    table_l_pre = precompute_catalog(table_l, table_s, rp_bins, table_c=table_c,
                                     cosmology=Planck15, comoving=True)
    table_r_pre = precompute_catalog(table_r, table_s, rp_bins, table_c=table_c,
                                     cosmology=Planck15, comoving=True)

Stacking the Signal
-------------------

The total galaxy-galaxy lensing signal can be obtained with the following code.
It first filters out all BOSS galaxies for which we couldn't find any source
galaxy nearby. Then we divide it into different jackknife samples that we will
later use to estimate uncertainties. Finally, we stack the lensing signal in
4 different BOSS redshift bins and save the data.

We choose to include all the necessary corrections factors. The shear
responsivity correction and multiplicative shear correction are the most
important and absolutely necessary. The selection bias corrections do not have
a big impact on the signal but is also required for HSC data. The photo-z
dilution correction is not strictly necessary but highly recommended. Finally,
the random subtraction is also highly recommended but not always applied. Note
that we don't apply a boost correction, but this is something that would also
be possible.::

    from dsigma.jackknife import add_continous_fields, jackknife_field_centers
    from dsigma.jackknife import add_jackknife_fields, jackknife_resampling
    from dsigma.stacking import excess_surface_density

    # Drop all lenses that did not have any nearby source.
    table_l_pre['n_s_tot'] = np.sum(table_l_pre['sum 1'], axis=1)
    table_l_pre = table_l_pre[table_l_pre['n_s_tot'] > 0]

    table_r_pre['n_s_tot'] = np.sum(table_r_pre['sum 1'], axis=1)
    table_r_pre = table_r_pre[table_r_pre['n_s_tot'] > 0]

    add_continous_fields(table_l_pre, distance_threshold=2)
    centers = jackknife_field_centers(table_l_pre, 100, weight='n_s_tot')
    add_jackknife_fields(table_l_pre, centers)
    add_jackknife_fields(table_r_pre, centers)

    z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

    for lens_bin in range(len(z_bins) - 1):
        mask_l = ((z_bins[lens_bin] <= table_l_pre['z']) &
                  (table_l_pre['z'] < z_bins[lens_bin + 1]))
        mask_r = ((z_bins[lens_bin] <= table_r_pre['z']) &
                  (table_r_pre['z'] < z_bins[lens_bin + 1]))

        kwargs = {'return_table': True, 'shear_bias_correction': True,
                  'shear_responsivity_correction': True,
                  'selection_bias_correction': True,
                  'boost_correction': False, 'random_subtraction': True,
                  'photo_z_dilution_correction': True,
                  'rotation': False, 'table_r': table_r_pre[mask_r]}

        result = excess_surface_density(table_l_pre[mask_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l_pre[mask_l], **kwargs)))
    
        result.write('hsc_{}.csv'.format(lens_bin))

Acknowledgements
----------------

When using the above data and algorithms, please make sure to cite
`Mandelbaum et al. (2018a) <https://ui.adsabs.harvard.edu/abs/
2018PASJ...70S..25M/abstract>`_ and `Mandelbaum et al. (2018b)
<https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.3170M/abstract>`_.
