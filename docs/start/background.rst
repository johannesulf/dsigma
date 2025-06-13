Background
==========

Here, we will give a brief overview of the various equations and estimators provided by :code:`dsigma`. However, this document is not a full theoretical discussion of the theory of gravitational lensing and galaxy-galaxy lensing. For this, we refer the reader to suitable standard literature like `Bartelmann (2001) <https://ui.adsabs.harvard.edu/abs/2001PhR...340..291B/abstract>`_.

Excess Surface Density
----------------------

The ultimate goal of :code:`dsigma` is to provide estimates of the mean so-called excess surface density (ESD), also written as :math:`\Delta\Sigma`, around a set of "lens" galaxies. The ESD is defined as the following quantity:

.. math::
    
    \Delta\Sigma (r_p) = \langle \Sigma (<r_p) \rangle - \Sigma (r_p), .

The first term is the (area-weighted) average surface density inside a circle of (projected) radius :math:`r_p` around the lens galaxy, and the second term is the mean surface density at the edge of that circle.

For most applications, the ESD will be positive since the surface density generally decreases with increasing distance :math:`r_p` from the lens. It is worthwhile to look at a few limiting cases. For example, for a point mass, one can show that the first term in the above equation is :math:`M / \pi r_p^2` whereas the second term vanishes. Thus, :math:`\Delta\Sigma = M / \pi r_p^2`. For non-trivial mass distributions, the above expression is more complicated. However, if we have two mass distributions whose only difference is a re-scaling, i.e. :math:`\Sigma_1 (r_p) / \Sigma_2 (r_p) = c` or :math:`\rho (r) / \rho_2 (r) = c` (:math:`\rho` refers to the three-dimensional density), then the two ESDs obey :math:`\Delta\Sigma_1 / \Delta\Sigma_2 = c`. Ultimately, :math:`\Delta\Sigma` is a measure of the distribution of mass (both baryonic and dark) around lens galaxies.

Tangential Shear
----------------

In the framework of general relativity, mass and energy affect the space-time metric, which ultimately leads to the deflection of light by the gravitational field of an object. On cosmological distances, the potential light deflection is a function of the so-called critical surface density, :math:`\Sigma_\mathrm{crit}`.

.. math::

    \Sigma_\mathrm{crit} (z_l, z_s) = \frac{c^2}{4\pi G} \frac{D_A (z_s)}{
        D_A (z_l) D_A (z_l, z_s)} \, .

In the above equation, :math:`D_A` denotes the angular diameter distance as a function of redshift :math:`z`. Particularly, :math:`z_s` denotes the redshift of the "source" which emitted the light. In the weak lensing regime, the surface densities of the lenses are small compared to the critical surface density, i.e., :math:`\Delta\Sigma \lll \Sigma_\mathrm{crit}`. In this regime, gravitational lensing will cause small, subtle changes in the apparent shapes of background source  galaxies. Particularly, the shapes of background source galaxies will be preferentially aligned perpendicular (if :math:`\Delta\Sigma > 0`) to the foreground lens galaxies. This alignment can be expressed in the mean tangential shear :math:`\gamma_t`. In the weak lensing regime, the relationship between the different quantities is straightforward:

.. math::

    \gamma_t = \frac{\Delta \Sigma}{\Sigma_\mathrm{crit}} \, .

Thus, by measuring the shapes of source galaxies (in addition to the redshifts of lens and source galaxies), we can estimate :math:`\Delta\Sigma`.

Naive Estimator
---------------

Unfortunately, galaxies are not perfectly round objects and are often intrinsically elliptical with random orientations. Thus, the shape of an individual source galaxy with respect to a lens galaxy will be dominated by its intrinsic arbitrary shape and orientation instead of the gravitational lensing signal we are after. This effect is also known as "shape noise." To get to the gravitational tangential shear signal, we need to average the tangential ellipticities :math:`e_t` of a large number of lens-source pairs to overcome the shape noise. On average, it is true that :math:`\langle e_t \rangle = \gamma_t`. A naive, minimum variance estimator for :math:`\Delta \Sigma` is

.. math::

    \Delta\Sigma =
        \frac{\sum_{ls} w_{ls} \Sigma_{\mathrm{crit}} (z_l, z_s) e_t}{
            \sum_{ls} w_{ls}} \, .

where

.. math::

    w_{ls} = \frac{w_s}{\Sigma_{\mathrm{crit}}^2 (z_l, z_s)}

and :math:`w_s` is a source weight designed to minimize errors due to noise in the shape measurements. :math:`\sum_{ls}` denotes that the summation goes over all suitable lens-source pairs separated by a certain projected distance. Evaluation of the above equation is one of the main tasks of :code:`dsigma`. Although the above equation looks simple enough, the number of suitable lens-source pairs can easily be in billions or trillions for many applications. If one is not careful, simple python implementations of the above sum can be computationally prohibitive. :code:`dsigma` has various intelligent ways of calculating the sum efficiently.

Correction Terms
----------------

Another reason for using :code:`dsigma` is that the above estimator will be biased in almost all practical applications. One must additionally apply various corrections to get an unbiased estimate of :math:`\Delta\Sigma`. In the following, we will describe and motivate the multiple corrections implemented in the code.

Lens Selection Bias
~~~~~~~~~~~~~~~~~~~

Often, our lens sample is not complete. For example, we might only have redshifts for a subset of all lens galaxies. Since redshifts are required to estimate :math:`\Delta\Sigma`, lens galaxies without redshifts must be excluded from the analysis. However, depending on the properties of the lens incompleteness, the intrinsic :math:`\Delta\Sigma` is often correlated with whether a lens galaxy makes it into our sample. Thus, the naive :math:`\Delta\Sigma` estimate using the incomplete lens sample could be biased concerning the :math:`\Delta\Sigma` of the complete sample. Assigning suitable systematic weights :math:`w_{\mathrm{sys}, l}` to the lens galaxies can counteract this effect. In this case, we replace the above estimator with

.. math::
    
    \Delta\Sigma =
        \frac{\sum_{ls} w_{\mathrm{sys}, l} w_{ls} \Sigma_{\mathrm{crit}}
              (z_l, z_s) e_t}{\sum_{ls} w_{\mathrm{sys}, l} w_{ls}} \, .

Photometric Redshifts
~~~~~~~~~~~~~~~~~~~~~

When calculating the critical surface density :math:`\Sigma_{\mathrm{crit}}`, we need to know both lens and source redshift. However, in many lensing surveys, we only have very inaccurate and possibly biased photometric redshifts for sources. Even in the case of just inaccurate redshifts, this can cause systematic biases in calculating :math:`\Sigma_{\mathrm{crit}}`. However, we can statistically correct the bias if we have a "calibration" catalog of sources with their inaccurate photometric redshifts and their true redshifts. The correction factor is called :math:`f_\mathrm{bias}` and can be calculated via

.. math::

    f_{\rm bias}^{-1} (z_l) =
        \frac{\sum_s w_{\mathrm{sys}, s} w_s \Sigma_\mathrm{crit, true}^{-1}
              \Sigma_\mathrm{crit, photo-z}^{-1}}{\sum_s w_{\mathrm{sys}, s}
              w_s \Sigma_{ls, \mathrm{photo-z}}^{-2}} =
        \langle \Sigma_\mathrm{crit, photo-z} / \Sigma_\mathrm{crit, true}
        \rangle\, ,

where :math:`w_{\mathrm{sys}, s}` is an additional weight assigned to each source in the calibration catalog to counter-act biases (similar to biases in the lens sample) or to account for the different response to shear (see below).

To apply this correction, the estimator becomes

.. math::

    \Delta\Sigma =
        \frac{\sum_{ls} w_{\mathrm{sys}, l} w_{ls} \Sigma_{\mathrm{crit}}
              (z_l, z_s) e_t f_\mathrm{bias} (z_l)}{\sum_{ls}
              w_{\mathrm{sys}, l} w_{ls}} \, .

The above formalism can also be extended to the case that we have no reliable redshift estimate for any individual source galaxy and only know an (effective) redshift distribution :math:`n(z)` for the entire population. In this case, the above estimator becomes

.. math::

    \Delta\Sigma =
        \frac{\sum_{ls} w_{\mathrm{sys}, l} w_s \langle
              \Sigma_{\mathrm{crit}}^{-1} (z_l) \rangle e_t}{\sum_{ls}
              w_{\mathrm{sys}, l} w_s \langle \Sigma_{\mathrm{crit}}^{-1} (z_l)
              \rangle^2} \, ,

where

.. math::

    \langle \Sigma_{\mathrm{crit}}^{-1} (z_l) \rangle = \int \Sigma_{\rm crit}^{-1}
        (z_l, z_s) n(z_s) \mathrm{d}z_s \, .

Boost Factor
~~~~~~~~~~~~

The photometric redshift correction above only accounts for the average effect of photometric redshift errors. However, it doesn't consider that there is an overabundance of physically associated sources close to lenses. As a result, close to actual lens galaxies, a more significant fraction of sources have actual redshifts placing them close to the lens, i.e. :math:`z_{\rm l} \approx z_{\rm s}` than what was going into the calculation of :math:`f_{\rm bias}`. For those physically associated sources :math:`\Sigma_{\mathrm{crit}} \approx \infty`, i.e., they do not induce any shear. Not considering this effect can lead to underestimating the proper lensing amplitude. One way to correct for this effect is to calculate the so-called boost factor :math:`b` via

.. math::

    b =
        \frac{\sum_{ls} w_{\mathrm{sys}, l} w_{ls}}{\sum_{rs}
              w_{\mathrm{sys}, l} w_{ls}} \, .

Note that the sum in the denominator goes over a set of random lenses with the same overall redshift and spatial extent as the actual lenses. Effectively, we are trying to detect any over-abundance of sources close to lenses instead of random points. If sources cluster around lenses, i.e., there is an over-abundance of sources physically associated with the lenses, then :math:`b \geq 1`. To apply the correction, we multiply the raw lensing signal by the radially dependent boost factor :math:`b`.

Note, however, that this boost factor estimate might be biased for various shape detection algorithms. For example, it is not unreasonable to assume that a more significant fraction of potential source galaxies is rejected close to massive cluster lenses due to increased blending between different sources. Such an effect would lead to biases in estimating :math:`b`.

Shear Response
~~~~~~~~~~~~~~

Although the algorithms used by different weak lensing groups are very sophisticated, the measured shapes of galaxies can still be biased. We need to correct this in the galaxy-galaxy lensing estimator to get unbiased estimates of :math:`\Delta\Sigma`. Typically, lensing surveys provide a (scalar) bias estimate :math:`m` for every object or group of objects. This bias quantifies the response of the measured shapes to changes in intrinsic shapes such that, on average, the measured ellipticities are biased by :math:`1 + m`. To correct for this bias, one first calculates the average shear bias via

.. math::

    \bar{m} =
        \frac{\sum_{ls} w_{\mathrm{sys, l}} w_{ls} m_s}{\sum_{ls}
              w_{\mathrm{sys, l}} w_{ls}} \, .

To apply the correction, we divide the raw lensing signal by :math:`1 + \bar{m}` in each radial bin. For some applications, this correction must be applied to a set of lens-source pairs and not every object individually. The reason is that individual shear bias estimates might be noisy. Thus, using this correction to the shapes of individual objects can lead to biases because generally :math:`\langle 1 + m \rangle^{-1} \neq \langle (1 + m)^{-1} \rangle`.

The Dark Energy Survey uses the shear measurement code METACALIBRATION. This code generalizes the shear response formalism by providing a shear response tensor :math:`\mathbf{R}` instead of a scalar.

.. math::

    \mathbf{R} = \frac{\partial \mathbf{e}}{\partial \mathbf{\gamma}} =
                 \begin{bmatrix} R_{11} & R_{12}\\R_{21} & R_{22} \end{bmatrix}

This formalism allows it to quantify, for example, how the sensitivity to shape distortions depends on the direction. However, this does not change the response formalism substantially. Instead of averaging the (scalar) bias, we average the projection of the response tensor onto the tangential direction of the lens-source pair,

.. math::

    R_t = R_{11} \cos^2 (2 \phi) + R_{22} \sin^2 (2 \phi)+ (R_{12} + R_{21})
          \sin (2 \phi) \cos (2 \phi)

where :math:`\phi` is the polar angle of the source in the lens coordinate system. We then calculate the mean tangential shear response and divide :math:`\Delta\Sigma` by :math:`\overline{R_t}`, similar to the scalar case.

Shear Responsivity
~~~~~~~~~~~~~~~~~~

Depending on the shape estimator, the ellipticity vector of some shape detection algorithm will be biased :math:`2 (1 - e_\mathrm{rms}^2)`, where :math:`e_\mathrm{rms}` is the intrinsic shape dispersion per component. In contrast to the shear response bias, this bias is not due to, for example, imperfections like blending and instead is a natural mathematical property of the estimator. Nonetheless, we need to correct it and do so virtually the same way as for the shear response bias.

.. math::

    \mathcal{R} = 1 -
        \frac{\sum_{ls} w_{\mathrm{sys}, l} w_{ls} e_{{\rm rms}, s}^2}{
              \sum_{ls} w_{\mathrm{sys}, l} w_{ls}} \, .

To apply the shear responsivity correction, we need to divide the final lensing signal by :math:`2\mathcal{R}`. This correction is significant, so it is critical to apply it when necessary. One example is the public shape catalog of the Hyper Suprime-Cam survey.

Source Selection Bias
~~~~~~~~~~~~~~~~~~~~~

While the shear response and responsivity describe how the measured shapes of individual galaxies are affected by the measurement pipeline, the selection bias :math:`m_\mathrm{sel}` describes the phenomenon that the selection of sources, i.e., quality cuts, can depend on the intrinsic shape itself. If not corrected, this effect could lead to biases in the mean tangential shear.

Correcting for selection bias is handled differently between different surveys. :code:`dsigma` implements this by calculating per-object estimates of :math:`m_\mathrm{sel}`. One can then calculate a mean :math:`m_\mathrm{sel}` and correct the measurements for the selection bias in the same way as for the shear response term :math:`m` describe above. Luckily, the selection bias affects the shear only at the level of :math:`\sim 1\%`.

Please have a look at the tutorials for DES and HSC to see how this is implemented in practice. For HSC in particular, the multiplicative selection bias takes the form

.. math::

    m_\mathrm{sel} = A P(R_2 = 0.3) + B P(\mathrm{mag}_A = 25.5) \, .

Here, :math:`A` and :math:`B` are constants that differ between HSC Y1 and HSC Y3 and :math:`R_2` and :math:`\mathrm{mag}_A` are per-object properties that were used in the selection. Finally, :math:`P` indicates the source probability density at the edge of the selection cut. The densities can be estimated by choosing some width in :math:`R_2` and :math:`\mathrm{mag}_A` around the selection cut, i.e.,

.. math::

    m_\mathrm{sel} \approx \frac{A P(0.3 - \delta R_2 \leq R_2 \leq 0.3)}{\delta R_2} + \frac{B P(25.5 - \delta\mathrm{mag}_A \leq \mathrm{mag}_A \leq 25.5)}{\delta\mathrm{mag}_A} \, .

This can be generalized to a per-object estimate of :math:`m_\mathrm{sel}` by setting :math:`P` to :math:`1` if :math:`R_2` or :math:`\mathrm{mag}_A` fall in the range and :math:`0`, otherwise.

Random Subtraction
~~~~~~~~~~~~~~~~~~

Often, we can calculate the lensing signal around a set of random lenses with the same overall redshift and spatial distribution as the actual lenses. If these random lenses are genuinely random, they do not correlate with the large-scale matter field. As a result, they should give a lensing amplitude consistent with 0. However, lensing systematics like systematic shears can lead to non-zero signals, even for random points. Subtracting those points can alleviate such systematic errors. Furthermore, even without lensing systematics, subtracting randoms leads to a reduced variance of the lensing signal on large scales, as shown in `Singh et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.3827S/abstract>`_.
