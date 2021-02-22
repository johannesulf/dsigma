Kilo-Degree Survey (KiDS)
=========================

In this tutorial, we will learn how to cross-correlate BOSS lens galaxies with
shape catalogs from KiDS. We will work with the KV450 data release but it
should be straightforward to adjust the code to work with DR3 (KiDS-450) or
DR4 (KiDS-1000) which are available as of January 2021.

Downloading the Data
--------------------

First, we need to download the necessary KiDS data files. The following 
commands should download all the necessary data. ::

    wget http://ds.astro.rug.astro-wise.org:8000/KV450_G9_reweight_3x4x4_v2_good.cat
    wget http://ds.astro.rug.astro-wise.org:8000/KV450_G12_reweight_3x4x4_v2_good.cat
    wget http://ds.astro.rug.astro-wise.org:8000/KV450_G15_reweight_3x4x4_v2_good.cat
    wget http://ds.astro.rug.astro-wise.org:8000/KV450_G23_reweight_3x4x4_v2_good.cat
    wget http://ds.astro.rug.astro-wise.org:8000/KV450_GS_reweight_3x4x4_v2_good.cat

    wget http://kids.strw.leidenuniv.nl/cs2018/KV450_COSMIC_SHEAR_DATA_RELEASE.tar.gz
    tar -xvf KV450_COSMIC_SHEAR_DATA_RELEASE.tar.gz KV450_COSMIC_SHEAR_DATA_RELEASE/REDSHIFT_DISTRIBUTIONS/Nz_DIR/Nz_DIR_Mean ./


Preparing the Data
------------------

First, we must put the data into a format easily understandable by
:code:`dsigma`. There are a number of helper functions to make this easy.
Additionally, we want to use the :math:`n(z)`'s provided by KiDS to correct
for photometric redshift biases. Thus, we also bin the source galaxies
by photometric redshift and read in the source redshift distribution in each
photometric redshift bin. ::

    import numpy as np
    from astropy import units as u
    from astropy.table import Table, vstack, join
    from dsigma.helpers import dsigma_table
    from dsigma.surveys import kids

    table_s = Table()

    for region in [9, 12, 15, 23, 'S']:
        table_s = vstack([table_s, Table.read(
            'KV450_G{}_reweight_3x4x4_v2_good.cat'.format(region), hdu=1)],
            metadata_conflicts='silent')

    table_s = table_s[table_s['MASK'] == 0]
    table_s = dsigma_table(table_s, 'source', survey='KiDS', version='KV450')

    z_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
    table_s['z_bin'] = np.digitize(table_s['z'], z_bins) - 1

    nz = np.array([np.genfromtxt('Nz_DIR_z{}t{}.asc'.format(z_min, z_max)).T for
                   z_min, z_max in zip(z_bins[:-1], z_bins[1:])])

    table_s = table_s[(table_s['z_bin'] >= 0) &
                      (table_s['z_bin'] < len(z_bins) - 1)]
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z'],
                                                  version='KV450')

Pre-Computing the Signal
------------------------

We will now run the computationally expensive pre-computation phase. Here,
we first define the lens-source separation cuts. We require that
:math:`z_l + 0.1 < z_s`. Afterwards, we run the actual pre-computation. ::

    from astropy.cosmology import Planck15
    from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog

    add_maximum_lens_redshift(table_s, dz_min=0.1)
    add_maximum_lens_redshift(table_c, dz_min=0.1)

    rp_bins = np.logspace(-1, 1.6, 14)
    table_l_pre = precompute_catalog(table_l, table_s, rp_bins, cosmology=Planck15,
                                     comoving=True, nz=nz)
    table_r_pre = precompute_catalog(table_r, table_s, rp_bins, cosmology=Planck15,
                                     comoving=True, nz=nz)

Stacking the Signal
-------------------

The total galaxy-galaxy lensing signal can be obtained with the following code.
It first filters out all BOSS galaxies for which we couldn't find any source
galaxy nearby. Then we divide it into different jackknife samples that we will
later use to estimate uncertainties. Finally, we stack the lensing signal in
4 different BOSS redshift bins and save the data.

We choose to include all the necessary corrections factors. The multiplicative
shear correction is the most important and absolutely necessary. The random
subtraction is recommended but not strictlynecessary. Note that we don't apply
a boost correction, but this is something that would also be possible.::

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

    for lens_bin in range(3, len(z_bins) - 1):
        mask_l = ((z_bins[lens_bin] <= table_l_pre['z']) &
                  (table_l_pre['z'] < z_bins[lens_bin + 1]))
        mask_r = ((z_bins[lens_bin] <= table_r_pre['z']) &
                  (table_r_pre['z'] < z_bins[lens_bin + 1]))
    
        kwargs = {'return_table': True, 'shear_bias_correction': True,
                  'random_subtraction': True, 'table_r': table_r_pre[mask_r]}

        result = excess_surface_density(table_l_pre[mask_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l_pre[mask_l], **kwargs)))

    result.write('kids_{}.csv'.format(lens_bin), overwrite=True)

Acknowledgements
----------------

When using the above data and algorithms, please to read and follow the
acknowledgement section on the
`KiDS KV450 release site <http://kids.strw.leidenuniv.nl/DR3/kv450data.php>`_.
