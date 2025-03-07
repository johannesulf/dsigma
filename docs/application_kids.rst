Kilo-Degree Survey (KiDS)
=========================

.. note::
    This guide has not been inspected or endorsed by the KiDS collaboration.

This tutorial will teach us how to cross-correlate BOSS lens galaxies with source catalogs from KiDS. We will work with the 4th data release (KiDS-1000).

Downloading the Data
--------------------

First, we need to download the necessary KiDS data files. The following commands should download all the required data.

.. code-block:: none

    wget http://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits
    wget http://kids.strw.leidenuniv.nl/DR4/data_files/KiDS1000_SOM_N_of_Z.tar.gz
    tar -xvf KiDS1000_SOM_N_of_Z.tar.gz --strip 1

Preparing the Data
------------------

First, we must put the data into a format easily understandable by :code:`dsigma`. There are several helper functions to make this easy. Additionally, we want to use the :math:`n(z)`'s provided by KiDS to correct for photometric redshift biases. Thus, we also bin the source galaxies by photometric redshift and read the source redshift distribution in each photometric redshift bin.

.. code-block:: python

    import numpy as np
    from astropy import units as u
    from astropy.table import Table
    from dsigma.helpers import dsigma_table
    from dsigma.surveys import kids

    table_s = Table.read('KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits')
    select = ((table_s['MASK'] == 0) & (table_s['weight'] > 0) &
              (table_s['model_SNratio'] > 0))
    table_s = table_s[select]
    table_s = dsigma_table(table_s, 'source', survey='KiDS')

    table_s['z_bin'] = kids.tomographic_redshift_bin(table_s['z'], version='DR4')
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z_bin'], version='DR4')
    table_s = table_s[table_s['z_bin'] >= 0]

    fname = ('K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2' +
             '_SOMcols_Fid_blindC_TOMO{}_Nz.asc')
    table_n = Table()
    table_n['z'] = np.genfromtxt(fname.format(1))[:, 0] + 0.025
    table_n['n'] = np.vstack(
        [np.genfromtxt(fname.format(i + 1))[:, 1] for i in range(5)]).T

Precomputing the Signal
-----------------------

We will now run the computationally expensive precomputation phase. Here, we first define the lens-source separation cuts. We require that :math:`z_l + 0.1 < z_{t, \rm min}` where :math:`z_{t, \rm min}` is the minimum redshift of the tomographic bin each source galaxy belongs to. Afterward, we run the actual precomputation.

.. code-block:: python

    from astropy.cosmology import Planck15
    from dsigma.precompute import precompute

    table_s['z'] = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[table_s['z_bin']]

    rp_bins = np.logspace(-1, 1.6, 14)
    precompute(table_l, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_n=table_n, lens_source_cut=0.1, progress_bar=True)
    precompute(table_r, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_n=table_n, lens_source_cut=0.1, progress_bar=True)

Stacking the Signal
-------------------

The total galaxy-galaxy lensing signal can be obtained with the following code. It first filters out all BOSS galaxies for which we couldn't find any source galaxy nearby. Then we divide it into jackknife samples that we will later use to estimate uncertainties. Finally, we stack the lensing signal in 4 different BOSS redshift bins and save the data.

We choose to include all the necessary corrections factors. The multiplicative shear correction is the most important and necessary. Random subtraction is recommended but not strictly necessary. Note that we don't apply a boost correction, but this would also be possible.

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
                  'random_subtraction': True, 'table_r': table_r[mask_r]}

        result = excess_surface_density(table_l[mask_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l[mask_l], **kwargs)))

        result.write('kids_{}.csv'.format(lens_bin), overwrite=True)

Acknowledgments
---------------

When using the above data and algorithms, please read and follow the acknowledgment section on the `KiDS DR4 release site <http://kids.strw.leidenuniv.nl/DR4/KiDS-1000_shearcatalogue.php#ack>`_.
