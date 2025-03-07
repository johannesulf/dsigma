Precomputation
==============

We need to sum over all lens-source pairs when calculating the total galaxy-galaxy lensing signal and potential correction factors. That means we must look at each lens and sum over all sources that can form a pair with that lens. The precomputation part performs this summation over all sources for all objects in the lens catalog. It is computationally the most demanding task of :code:`dsigma`.

To run the precomputation, we must first make decisions regarding the radial binning, cosmology as well as lens-source cuts. We cannot change these choices after running the precomputation. Here is some example code, assuming one has already read in the lens and source catalogs. In this case, we assume Planck15 cosmology, radial bins going from 0.1 to 10 Mpc and we are using all lens-source pairs where :math:`z_l < z_s - 0.1`. :code:`dsigma` supports running this expensive calculation on multiple CPU cores. In this case, we are using 4.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import FlatLambdaCDM
    from dsigma.precompute import add_precompute_results

    rp_bins = np.logspace(-1, 1, 11)
    cosmology = FlatLambdaCDM(H0=100, Om0=0.27)
    lens_source_cut = 0.1

    # Perform the precomputation.
    add_precompute_results(table_l, table_s, rp_bins, cosmology=cosmology,
                           comoving=True, lens_source_cut=lens_source_cut,
                           progress_bar=True, n_jobs=4)

After the precomputation, we can stack the signal around each lens to obtain the total lensing signal. At the precomputation stage, applying cuts on the lens sample is unnecessary. We can do this immediately before the stacking. Consequently, after performing the precomputation, we can easily derive lensing signals for arbitrary sub-samples of the lens catalog.
