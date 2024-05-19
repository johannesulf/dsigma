Jackknife Resampling
====================

In addition to having a measurement of the total lensing signal, we would also like to know the associated uncertainty. We can do this with the jackknife resampling technique. To do so, we need to divide our lens catalog into :math:`n_{\mathrm{jk}}` subsamples of roughly equal sizes. The uncertainty of any observable :math:`d` can then be expressed as

.. math::

    \sigma_d^2 = \frac{n_{\mathrm{jk}} - 1}{n_{\mathrm{jk}}}
    \sum\limits_{i=1}^{n_{\mathrm{jk}}} (d_i - d)^2,

where :math:`d` is the observable computed from the entire sample and :math:`d_i` is the observable computed from the entire sample except subsample :math:`i`. :code:`dsigma` allows you to construct the jackknife samples and to perform the resampling to obtain uncertainties.

Constructing Jackknife Fields
-----------------------------

:code:`dsigma` uses a combination of DBSCAN and K-means clustering for determining jackknife fields. The jackknife fields are determined via :math:`n_{\mathrm{jk}}` centers and lenses are associated to each jackknife field based on the nearest center. This allows use to easily use the same jackknife fields for both lenses and randoms. The following code will construct 100 jackknife fields.

.. code-block:: python

    from dsigma.jackknife import compute_jackknife_fields

    centers = compute_jackknife_fields(table_l, 100)
    compute_jackknife_fields(table_r, centers)
    
In the above code, ``table_l`` and ``table_r`` are the precompute results for a set of lenses and randoms, respectively. Information about jackknife regions is stored in the ``field_jk`` column of the lens and random tables.

Jackknife Resampling
--------------------

We are now ready to derive uncertainties on any summary statistic derived from the entire lens and (optionally) random samples. The following code calculates the covariance matrix on the galaxy-galaxy lensing signal.

.. code-block:: python

    from dsigma.stacking import excess_surface_density
    from dsigma.jackknife import jackknife_resampling

    delta_sigma_cov = jackknife_resampling(excess_surface_density, table_l,
                                           table_r)