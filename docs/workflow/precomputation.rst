Precomputation
==============

We need to sum over all lens-source pairs when calculating the total galaxy-galaxy lensing signal and potential correction factors. That means we must look at each lens and sum over all sources that can form a pair with that lens. The precomputation part performs this summation over all sources for all objects in the lens catalog. It is computationally the most demanding task of ``dsigma``.

To run the precomputation, we must first make decisions regarding the radial binning, cosmology, and lens-source cuts. These choices are baked into the precomputed results and cannot be changed without rerunning the precomputation. The following example assumes Planck15 cosmology, radial bins going from 0.1 to 10 Mpc and we are using all lens-source pairs where :math:`z_l < z_s - 0.1`. ``dsigma`` supports running this expensive calculation on multiple CPU cores. Here we use four CPU cores.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import Planck15

    from dsigma.precompute import precompute

    rp_bins = np.logspace(-1, 1, 11)
    cosmology = Planck15
    table_s['z_l_max'] = table_s['z'] - 0.1

    # Perform the precomputation.
    precompute(table_l, table_s, rp_bins, cosmology=cosmology, comoving=True,
               progress_bar=True, n_jobs=4)

After the precomputation, we can stack the signal around each lens to obtain the total lensing signal. Cuts on the lens sample can be applied later, immediately before stacking. Consequently, after performing the precomputation, we can easily derive lensing signals for arbitrary sub-samples of the lens catalog.
