Jackknife Resampling
====================

In addition to having a measurement of the total lensing signal, we would also like to know the associated uncertainty. We can do this with the jackknife resampling technique. To do so, we need to divide our lens catalog into :math:`n_{\mathrm{jk}}` subsamples of roughly equal sizes. The uncertainty of any observable :math:`d` can then be expressed as

.. math::

    \sigma_d^2 = \frac{n_{\mathrm{jk}} - 1}{n_{\mathrm{jk}}}
    \sum\limits_{i=1}^{n_{\mathrm{jk}}} (d_i - d)^2,

where :math:`d` is the observable computed from the entire sample and :math:`d_i` is the observable computed from the entire sample except subsample :math:`i`. :code:`dsigma` allows you to construct the jackknife samples and to perform the resampling to obtain uncertainties.

Note that the jackknife procedure used by :code:`dsigma` corresponds to the method studied in `Shirasaki et al. (2017) <https://doi.org/10.1093/mnras/stx1477>`_. Specifically, for each jackknife sample, only lenses in the jackknife patch are removed while :code:`dsigma` always uses all sources. This matters for lens-source pairs where one galaxy is inside the jackknife region and the other one is outside. For :code:`dsigma`, such a pair is removed if and only if the lens falls into the jackknife patch. This cross-patch behavior can differ from how this is handled in, e.g., :code:`treecorr`, where jackknifing is typically based on both lenses and sources.

Constructing Jackknife Fields
-----------------------------

:code:`dsigma` uses a combination of DBSCAN and K-means clustering for determining jackknife fields. The jackknife fields are determined via :math:`n_{\mathrm{jk}}` centers and lenses are associated to each jackknife field based on the nearest center. This makes it straightforward to use the same jackknife fields for both lenses and randoms. The following code will construct 100 jackknife fields.

.. code-block:: python

    from dsigma.jackknife import compute_jackknife_fields

    centers = compute_jackknife_fields(table_l, 100)
    compute_jackknife_fields(table_r, centers)

In the above code, ``table_l`` and ``table_r`` are the precompute results for a set of lenses and randoms, respectively. Information about jackknife regions is stored in the ``field_jk`` column of the lens and random tables.

Jackknife Resampling
--------------------

We are now ready to derive uncertainties for any summary statistic derived from the entire lens and (optionally) random samples. The following code calculates the covariance matrix on the galaxy-galaxy lensing signal.

.. code-block:: python

    from dsigma.stacking import excess_surface_density
    from dsigma.jackknife import jackknife_resampling

    delta_sigma_cov = jackknife_resampling(excess_surface_density, table_l,
                                           table_r)

Choosing the Number of Fields
-----------------------------

As discussed in `Shirasaki et al. (2017) <https://doi.org/10.1093/mnras/stx1477>`_, there is a trade-off when choosing the number of jackknife fields :math:`N_\mathrm{jk}`. If the number is too small, the jackknife covariance estimate :math:`C_\mathrm{jk}` is very noisy. Furthermore, the estimate of the inverse of the covariance matrix, the precision matrix :math:`\Psi`, may be biased, as discussed in `Hartlap et al. (2007) <https://doi.org/10.1051/0004-6361:20066170>`_. In fact, the covariance matrix is non-invertible if the number of jackknife samples is not larger than the number of observations :math:`N_\mathrm{obs}`, i.e., the number of radial bins. However, choosing a larger number makes the jackknife regions smaller. As discussed in `Shirasaki et al. (2017) <https://doi.org/10.1093/mnras/stx1477>`_, this may bias the covariance estimate if the scale of the jackknife regions is similar or smaller than the separations for which :math:`\Delta\Sigma` is measured.

Small biases in the precision matrix, as discussed in `Hartlap et al. (2007) <https://doi.org/10.1051/0004-6361:20066170>`_, may be mitigated using the explicit "Hartlap correction", :math:`\hat\Psi = (N_\mathrm{jk} - N_\mathrm{obs} - 2) / (N_\mathrm{jk} - 1) C_\mathrm{jk}^{-1}`. Alternatively, one may use a smoothing procedure as outlined in `Mandelbaum et al. (2013) <https://doi.org/10.1093/mnras/stt572>`_. The smoothing procedure is implemented in :code:`dsigma` via the :func:`~dsigma.jackknife.smooth_covariance_matrix` method. In this case, the off-diagonal terms of the correlation matrix are smoothed with a Gaussian kernel of variable length.

.. code-block:: python

    from dsigma.jackknife import smooth_covariance_matrix

    delta_sigma_cov = smooth_covariance_matrix(delta_sigma_cov, sigma=1)

Note that :func:`~dsigma.jackknife.smooth_covariance_matrix` only smoothes the off-diagonal elements of the covariance matrix and leaves the diagonal elements unchanged.
