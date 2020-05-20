Precomputation
==============

When calculating the total galaxy-galaxy lensing signal, we need to perform
a summation over all lens-source pairs. That means we need to look at each
lens and sum over all sources that can form form a pair with that lens. The
precomputation part performs this summuation over all sources for all objects
in the lens catalog. It is computationally the most demanding task of
``dsigma``.

After the precomputation, we can stack the signal around each lens to obtain
the total lensing signal. At the precomputation stage, it is not necessary to
apply cuts on the lens sample. This can instead be done immediately before the
stacking. Consequently, after performing the precomputation, we can easily
derive lensing signals for arbitrary sub-samples of the lens catalog.

Lens-Source Cuts
----------------

In principle, we can employ all sources with redshift :math:`z_{\mathrm{s}}`
that places them behind the lens redshift :math:`z_{\mathrm{l}}`, i.e.
:math:`z_{\mathrm{l}} < z_{\mathrm{s}}`. However, our estimate of the source
redshift often comes with considerable uncertainties. It is thus reasonable
to employ additional redshift cuts to ensure that the vast majority of sources
are behind the lens. This can be done with :code:`add_maximum_lens_redshift`.
In the following example, we require from the source and calibration catalog
that :math:`z_{\mathrm{l}} < z_{\mathrm{s}} - \mathrm{max} (0.1, \sigma_z)`.

.. code-block:: python

    for table in [table_s, table_c]:
        table = add_maximum_lens_redshift(table, dz_min=0.1, z_err_factor=1)

Precomputation
--------------

After defining lens-source cuts, we can run the precomputation. However, we
must decide on cosmological parameters and the binning in projected radii.
Note that these choices cannot be changed after running the pre-computation.
Here is some example code, assuming one has already read in the lens and
source catalogs.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import FlatLambdaCDM
    from dsigma.precompute import precompute_catalog

    rp_bins = np.logspace(-1, 1, 11)
    cosmology = FlatLambdaCDM(H0=100, Om0=0.27)
    
    # Perform the precomputation.
    table_l = precompute_catalog(table_l, table_s, rp_bins,
                                 cosmology=cosmology, comoving=True)
