Dark Energy Survey (DES)
========================

.. note::
    This guide has not been inspected or endorsed by the DES collaboration.

Here, we we will show how to cross-correlate BOSS lens galaxies with shape catalogs from DES. We will work with the Y3 data release. Check out the documentation of :code:`dsigma` v0.6 if you want to see how to reduce DES Y1 data.

Downloading the Data
--------------------

DES Y3 data can be downloaded `here <https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/>`_. The following command should download all the necessary data.

.. code-block:: none

    wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_sompz_v0.40.h5
    wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_metacal_v03-004.h5
    wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_indexcat.h5

Unfortunately, the total amount of data is very large, i.e. hundreds of GBytes. We can use the following script first to reduce the data and only save what we need for the galaxy-galaxy lensing calculation.

.. code-block:: python

    import h5py
    import numpy as np
    from astropy.table import Table
    
    table_s = Table()
    
    fstream = h5py.File('DESY3_sompz_v0.40.h5')
    table_s['bhat'] = fstream['catalog/sompz/unsheared/bhat'][()]
    fstream.close()
    
    fstream = h5py.File('DESY3_metacal_v03-004.h5')
    
    for key in ['ra', 'dec', 'e_1', 'e_2', 'R11', 'R12', 'R21', 'R22', 'weight']:
        table_s[key] = fstream['catalog/unsheared/' + key][()]
    
    for sheared in ['1m', '1p', '2m', '2p']:
        table_s['weight_{}'.format(sheared)] = fstream[
            'catalog/sheared_{}/weight'.format(sheared)][()]
    
    fstream.close()
    
    fstream = h5py.File('DESY3_indexcat.h5')
    
    for flag in ['select', 'select_1p', 'select_1m', 'select_2p', 'select_2m']:
        table_s['flags_' + flag] = np.zeros(len(table_s), dtype=bool)
        table_s['flags_' + flag][fstream['index/' + flag][()]] = True
    
    select = (table_s['flags_select'] | table_s['flags_select_1p'] |
              table_s['flags_select_1m'] | table_s['flags_select_2p'] |
              table_s['flags_select_2m']) & (table_s['bhat'] >= 0)
    table_s = table_s[select]
    fstream.close()
    
    select = (table_s['ra'] < 60) | (table_s['ra'] > 300)
    select &= table_s['dec'] > -22.5
    table_s = table_s[select]
    table_s.write('des_y3.hdf5', path='catalog', overwrite=True)
    
    table_n = Table.read(
        '2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits', hdu=6)
    table_n.rename_column('Z_MID', 'z')
    table_n['n'] = np.vstack([table_n['BIN{}'.format(i + 1)] for i in range(4)]).T
    table_n.keep_columns(['z', 'n'])
    table_n.write('des_y3.hdf5', path='redshift', overwrite=True, append=True)

The final file, :code:`des_y3.hdf5`, is also available from the :code:`dsigma` authors upon request.

Preparing the Data
------------------

First, we must put the data into a format easily understandable by :code:`dsigma`. There are several helper functions to make this easy. Additionally, we want to use the :math:`n(z)`'s provided by DES Y3 to correct for photometric redshift biases.

In an intermediate step, we also calculate the so-called METACALIBRATION selection response. This factor takes into account how the selection flags of the METACALIBRATION shape measurements used by DES might be biased by shear itself. We need to correct such a bias in order to get unbiased shear and :math:`\Delta\Sigma` measurements. See `Sheldon & Huff (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJ...841...24S>`_ and `McClintock et al. (2018) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1352M>`_ for details. We will added this response to the total METACALIBRATION response that also takes into account how the measured shear is biased with respect to the intrinsic shear. Ideally, one would calculate the selection response for each radial bin and each specific lens sample (because this affects the source weighting). Additionally, one could also fold in how artificial shear affects the METACALIBRATION redshifts. However, as we can see above, the selection response bias is likely very small and not a strong function of redshift. Thus, we will ignore this complication here (cf. McClintock et al. 2018).

After running this selection response calculation, we are ready to drop all galaxies that are flagged for the unsheared images (and also those galaxies that fall outside the redshift bins).

.. code-block:: python

    table_s = Table.read('des_y3.hdf5', path='catalog')
    table_s = dsigma_table(table_s, 'source', survey='DES')

    for z_bin in range(4):
        select = table_s['z_bin'] == z_bin
        R_sel = des.selection_response(table_s[select])
        print("Bin {}: R_sel = {:.1f}%".format(
            z_bin + 1, 100 * 0.5 * np.sum(np.diag(R_sel))))
        table_s['R_11'][select] += 0.5 * np.sum(np.diag(R_sel))
        table_s['R_22'][select] += 0.5 * np.sum(np.diag(R_sel))

    table_s = table_s[table_s['z_bin'] >= 0]
    table_s = table_s[table_s['flags_select']]
    table_s['m'] = des.multiplicative_shear_bias(
        table_s['z_bin'], version='Y3')

    table_n = Table.read('des_y3.hdf5', path='redshift')

Precomputing the Signal
-----------------------

We will now run the computationally expensive precomputation phase. Here, we first define the lens-source separation cuts. We require that :math:`z_l + 0.1 < z_{t, \rm low}` where :math:`z_{t, \rm low}` is the lower redshift bin edge of the tomographic bin `(Myles et al., 2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.4249M>`_ each source galaxy belongs to. Afterward, we run the actual precomputation.


.. code-block:: python

    from astropy.cosmology import Planck15
    from dsigma.precompute import precompute
    
    table_s['z'] = np.array([0.0, 0.358, 0.631, 0.872])[table_s['z_bin']]

    rp_bins = np.logspace(-1, 1.6, 14)
    precompute(table_l, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_c=table_c, lens_source_cut=0.1, progress_bar=True)
    precompute(table_r, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_c=table_c, lens_source_cut=0.1, progress_bar=True)

Stacking the Signal
-------------------

The total galaxy-galaxy lensing signal can be obtained with the following code. It first filters out all BOSS galaxies for which we couldn't find any source galaxy nearby. Then we divide it into jackknife samples that we will later use to estimate uncertainties. Finally, we stack the lensing signal in 4 different BOSS redshift bins and save the data.

We choose to include all the necessary corrections factors. In addition to the matrix shear response correction (METACALIBRATION), we perform a random subtraction which is highly recommended but not strictly necessary. Note that we don't apply a boost correction since this might be biased for DES given our boost estimator.

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

    for lens_bin in range(3, len(z_bins) - 1):
        mask_l = ((z_bins[lens_bin] <= table_l['z']) &
                  (table_l['z'] < z_bins[lens_bin + 1]))
        mask_r = ((z_bins[lens_bin] <= table_r['z']) &
                  (table_r['z'] < z_bins[lens_bin + 1]))

        kwargs = {'return_table': True, 'scalar_shear_response_correction': True,
                  'matrix_shear_response_correction': True,
                  'random_subtraction': True, 'table_r': table_r[mask_r]}

        result = excess_surface_density(table_l[mask_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l[mask_l], **kwargs)))

        result.write('des_{}.csv'.format(lens_bin), overwrite=True)

Acknowledgements
----------------

When using the above data and algorithms, please to read and follow the acknowledgement section on the `DES Y3 data release site <https://des.ncsa.illinois.edu/releases/y3a2>`_.
