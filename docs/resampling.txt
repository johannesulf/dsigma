Jackknife Resampling
====================

In addition to having a measurement of the total lensing signal, we would also
like to know an associated uncertainty. This can be done with the jackknife
resampling technique. To do so, we need to divide our lens catalog into
:math:`n_{\mathrm{jk}}` subsamples of roughly equal sizes. The uncertainty of
any observable :math:`d` can then be expressed as

.. math::

    \sigma_d^2 = \frac{n_{\mathrm{jk}} - 1}{n_{\mathrm{jk}}}
    \sum\limits_{i=1}^{n_{\mathrm{jk}}} (d_i - d)^2,

where :math:`d_i` is the observable computed from the entire sample except
subsample :math:`i`. ``dsigma`` can be used to calculate jackknife subsamples
and calculate the derived uncertainties.

Continous Fields
----------------

Many lensing surveys do not probe a single continous field of the sky.
Instead, many distinct patches of the sky are surveyed. That adds slight
complications to calculating jackknife subsamples. We must first determine
all continous fields.

.. code-block:: python

    from dsigma.jackknife import add_continous_fields

    table_l = add_continous_fields(table_l, distance_threshold=1)

That's all that's needed to calculate continous fields. Internally, ``dsigma``
links all lenses closer than :math:`1^\circ` and defines continous fields
as groups of lenses that can be linked together. Information about field
association is stored in the ``field`` column. Of course, if such an
assignment is already given in the original catalog, feel free to use it.

Jackknife Fields
----------------

After determining continous fields, we are ready to calculate the jackknife
subsamples. ``dsigma`` uses simple k-means clustering for that.

.. code-block:: python

    from dsigma.jackknife import jackknife_field_centers, add_jackknife_fields

    centers = jackknife_field_centers(table_l, n_jk=100)
    
    # lenses
    table_l = add_jackknife_fields(table_l, centers)

    # random lenses
    table_lr = add_jackknife_fields(table_lr, centers)
    

In this case, we calculated 100 jackknife subsamples. Information about
jackknife regions is stored in the ``field_jk`` column.

Resampling
----------

We are now ready to derive uncertainties on any summary statistic derived from
the entire lens sample. The following code calculates the covariance matrix on
the galaxy-galaxy lensing signal.

.. code-block:: python

    from dsigma.stacking import excess_surface_density
    from dsigma.jackknife import jackknife_resampling

    delta_sigma_cov = jackknife_resampling(excess_surface_density, table_l,
                                           table_lr)

In the above code, ``table_l`` and ``table_lr`` are the precompute results for
a set of lenses and random lenses, respectively.
