:orphan:

Dark Energy Survey (DES)
========================

.. note::
    This guide has not been inspected or endorsed by the DES collaboration.

Here, we will show how to cross-correlate BOSS lens galaxies with shape
catalogs from DES. We will work with the Y1 data release.

Downloading the Data
--------------------

DES data can be downloaded from the `DES data release website
<https://des.ncsa.illinois.edu/releases>`_. The following commands should
download all the necessary data. The first file downloaded is the main shape
catalog, the second one includes the METACALIBRATION photo-z's used to
bin galaxies and the last one includes the MOF Monte-Carlo redshifts we will
use to approximate the true redshift distribution :math:`n(z)`.

.. code-block:: none

    wget http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/shear_catalogs/mcal-y1a1-combined-riz-unblind-v4-matched.fits
    wget http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/photoz_catalogs/y1a1-gold-mof-badregion_BPZ.fits
    wget http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/photoz_catalogs/mcal-y1a1-combined-griz-blind-v3-matched_BPZbase.fits

Preparing the Data
------------------

The first step is to put the data into a format easily understandable by
:code:`dsigma`. There are a number of helper functions to make this easy.
Additionally, we are going to bin source galaxies by tomographic redshift.
(We are using :code:`fitsio` instead of :code:`astropy.io.fits` because it
allows us to read only certain columns of the very large input table. If
memory is not an issue, feel free to use :code:`astropy`, instead.)

.. code-block:: python

    import fitsio
    from astropy.table import Table, hstack
    from dsigma.helpers import dsigma_table
    from dsigma.surveys import des

    table_s = []

    fname_list = ['mcal-y1a1-combined-riz-unblind-v4-matched.fits',
                  'y1a1-gold-mof-badregion_BPZ.fits',
                  'mcal-y1a1-combined-griz-blind-v3-matched_BPZbase.fits']
    columns_list = [['e1', 'e2', 'R11', 'R12', 'R21', 'R22', 'ra', 'dec',
                     'flags_select', 'flags_select_1p', 'flags_select_1m',
                     'flags_select_2p', 'flags_select_2m'], ['Z_MC'], ['MEAN_Z']]

    for fname, columns in zip(fname_list, columns_list):
        table_s.append(Table(fitsio.read(fname, columns=columns), names=columns))

    table_s = hstack(table_s)
    table_s = dsigma_table(table_s, 'source', survey='DES')
    table_s['z_bin'] = des.tomographic_redshift_bin(table_s['z'])

Selection Response
------------------

In an intermediate step, we calculate the so-called METACALIBRATION selection
response. This factor takes into account how the selection flags of the
METACALIBRATION shape measurements used by DES might be biased by shear itself.
We need to correct such a bias in order to get unbiased shear and
:math:`\Delta\Sigma` measurements.
See `Sheldon & Huff (2017)
<https://doi.org/10.3847/1538-4357/aa704b>`_ and
`McClintock et al. (2018)
<https://doi.org/10.1093/mnras/sty2711>`_ for details.

.. code-block:: python

    import numpy as np

    for z_bin in range(4):
        use = table_s['z_bin'] == z_bin
        R_sel = des.selection_response(table_s[use])
        print("Bin {}: R_sel = {:.1f}%".format(
            z_bin + 1, 100 * 0.5 * np.sum(np.diag(R_sel))))
        table_s['R_11'][use] += 0.5 * np.sum(np.diag(R_sel))
        table_s['R_22'][use] += 0.5 * np.sum(np.diag(R_sel))

This will give the following output.

.. code-block:: python

    Bin 1: R_sel = 1.2%
    Bin 2: R_sel = 1.5%
    Bin 3: R_sel = 1.1%
    Bin 4: R_sel = 0.9%

You can see that we added this response to the total METACALIBRATION response
that also takes into account how the measured shear is affected by intrinsic
shear. Ideally, one would calculate the selection response for each radial bin
and each specific lens sample (because this affects the source weighting).
Additionally, one could also fold in how artificial shear affects the
METACALIBRATION redshifts. However, as we can see above, the selection response
bias is likely very small and not a strong function of redshift. Thus, we will
ignore this complication here (cf. McClintock et al. 2018).

After running this selection response calculation, we are ready to drop all
galaxies that are flagged for the unsheared images (and also those galaxies
that fall outside the redshift bins).

.. code-block:: python

    table_s = table_s[(table_s['flags_select'] == 0) & (table_s['z_bin'] != -1)]

Note on the Estimator
---------------------

The Dark Energy Survey uses the following estimator for :math:`\Delta\Sigma`
(excluding random subtraction):

.. math::

    \Delta\Sigma = \frac{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}} w_s
        \Sigma_{\rm crit}^{-1} (z_l, z_{s, \rm META}) e_t}{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}}
        \Sigma_{\rm crit}^{-1} (z_l, z_{s, \rm MOF, MC})
        \Sigma_{\rm crit}^{-1} (z_l, z_{s, \rm META}) w_s R_{t,ls}} \, ,

where :math:`z_{s, \rm META}` is the mean of the :math:`p(z)` of the
METACALIBRATION photometric redshift and :math:`z_{s, \rm MOF, MC}` a
Monte-Carlo draw of the :math:`p(z)` from the multi-object fitting (MOF)
photometric redshift. Finally, :math:`R_{t,ls}` denotes the tangential
component of the shear response of each individual lens-source pair. Initially,
this might look very different from the estimators shown in the earlier section
on :doc:`stacking </workflow/stacking>`. However, we can re-arrange the terms to
have the following form:

.. math::

    \Delta\Sigma =& \frac{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}} w_{ls}}{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}} w_{ls} R_{t,ls}} \frac{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}} w_{ls} R_{t,ls}}{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}} w_{ls} R_{t,ls}
        \frac{\Sigma_{\rm crit} (z_l, z_{s, \rm META})}{
              \Sigma_{\rm crit} (z_l, z_{s, \rm MOF, MC})}}\\&\frac{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}} w_{ls} e_t
        \Sigma_{\rm crit} (z_l, z_{s, \rm META})}{
        \sum_{\mathrm{ls}} w_{\mathrm{sys, l}} w_{ls}} \, ,

where

.. math::

    w_{ls} = \frac{w_s}{\Sigma_{\rm crit}^2 (z_l, z_{\rm META})}

is the usual weight using the METACALIBRATION redshift. The above equation is
much more similar to what :code:`dsigma` works with. The first of the three
fractions corresponds to the correction for the mean shear response. The second
term is a response-weighted :math:`f_{\rm bias}` factor where the Monte-Carlo
draw from the MOF :math:`p(z)` is used as the "true" redshift. This implies
that we implicitly assume that the Monte-Carlo draws from the MOF :math:`p(z)`
sample the true redshift distributions :math:`n(z)` of source galaxies.
Finally, the third time is the usual estimate of the raw excess surface density
where we used the METACALIBRATION redshift to calculate weights and critical
surface densities.

:code:`dsigma` implements almost the same estimator with the only difference
being how the :math:`f_{\rm bias}` term is calculated and applied. First, the
DES estimator calculates :math:`f_{\rm bias}` using only the actual lens-source
pairs in each radial bin and using the actual projected response
:math:`R_{t,ls}` for the response weighting. On the other hand, :code:`dsigma`
uses all lenses and all sources, regardless of angular separation. Also,
because of this, :code:`dsigma` does not use the projected shear response
and we will use the spherically-averaged response :math:`0.5 (R_{11} + R_{22})`
for each source, instead.

.. code-block:: python

    table_c = table_s['z', 'z_true', 'w']
    table_c['w_sys'] = 0.5 * (table_s['R_11'] + table_s['R_22'])

This difference should not induce changes as long as neither the tangential
response nor the source photometric redshifts are a strong function of
lens-source separation. The assumption regarding the tangential response can
easily be verified using :code:`dsigma` in the stacking analysis below. On the
other hand, source redshifts, both photometric and intrinsic, will likely
change close to lenses. So additional correction may need to be applied,
anyway. :code:`dsigma` uses the lens-source spatial clustering to (optionally)
estimate this effect but due to the complex selection function in DES, this
estimator is likely biased, too. Overall, :math:`\Delta\Sigma` estimates on
small scales (:math:`r_p \lesssim 1 \mathrm{Mpc} / h`) may currently be biased.
The second difference is that the :math:`f_{\rm bias}` in DES is applied to the
averaged raw :math:`\Delta\Sigma` estimate whereas :code:`dsigma` applies it to
each individual raw :math:`\Delta\Sigma` estimate as a function of lens
redshift, i.e. :math:`f_{\rm bias} = f_{\rm bias} (z_l)`. This difference
should not matter since both estimators can be shown to be unbiased.


Precomputing the Signal
-----------------------

We will now run the computationally expensive pre-computation phase. Here,
we first define the lens-source separation cuts. We require that
:math:`z_l + 0.1 < z_s`. Afterwards, we run the actual pre-computation.

.. code-block:: python

    from astropy.cosmology import Planck15
    from dsigma.precompute import add_maximum_lens_redshift, add_precompute_results

    add_maximum_lens_redshift(table_s, dz_min=0.1)
    add_maximum_lens_redshift(table_c, dz_min=0.1)

    rp_bins = np.logspace(-1, 1.6, 14)
    add_precompute_results(table_l, table_s, rp_bins, cosmology=Planck15,
                           comoving=True, table_c=table_c)
    add_precompute_results(table_r, table_s, rp_bins, cosmology=Planck15,
                           comoving=True, table_c=table_c)

Stacking the Signal
-------------------

The total galaxy-galaxy lensing signal can be obtained with the following code.
It first filters out all BOSS galaxies for which we couldn't find any source
galaxy nearby. Then we divide it into different jackknife samples that we will
later use to estimate uncertainties. Finally, we stack the lensing signal in
4 different BOSS redshift bins and save the data.

We choose to include all the necessary corrections factors. The tensor shear
response correction and the photo-z dilution correction are the ones discussed
above. Additionally, we perform a random subtraction which is highly
recommended but not strictly necessary. Note that we don't apply
a boost correction since this might be biased for DES given our boost
estimator.

.. code-block:: python

    from dsigma.jackknife import add_continous_fields, jackknife_field_centers
    from dsigma.jackknife import add_jackknife_fields, jackknife_resampling
    from dsigma.stacking import excess_surface_density

    # Drop all lenses that did not have any nearby source.
    table_l['n_s_tot'] = np.sum(table_l['sum 1'], axis=1)
    table_l = table_l[table_l['n_s_tot'] > 0]

    table_r['n_s_tot'] = np.sum(table_r['sum 1'], axis=1)
    table_r = table_r[table_r['n_s_tot'] > 0]

    centers = compute_jackknife_fields(
        table_l, 100, weights=np.sum(table_l['sum 1'], axis=1))
    compute_jackknife_fields(table_r, centers)

    z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

    for lens_bin in range(3, len(z_bins) - 1):
        mask_l = ((z_bins[lens_bin] <= table_l['z']) &
                  (table_l['z'] < z_bins[lens_bin + 1]))
        mask_r = ((z_bins[lens_bin] <= table_r['z']) &
                  (table_r['z'] < z_bins[lens_bin + 1]))

        kwargs = dict(return_table=True, photo_z_dilution_correction=True,
                      matrix_shear_response_correction=True,
                      random_subtraction=True, table_r=table_r[mask_r])

        result = excess_surface_density(table_l[mask_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l[mask_l], **kwargs)))

        result.write(f'des_{lens_bin}.csv', overwrite=True)

Acknowledgements
----------------

When using the above data and algorithms, please read and follow the
acknowledgement section on the `DES data release site
<https://des.ncsa.illinois.edu/thanks>`_.

