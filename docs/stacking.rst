Stacking the Signal
===================

After the precomputation phase, we are ready to calculate the total
galaxy-galaxy lensing signal. The main quantity that is calculated is the
excess surface density, :math:`\Delta\Sigma`. In the absence of any
systematics, it can be calculated as follows.

.. math::
    
    \Delta\Sigma =
        \frac{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}} w_{\mathrm{sys, l}}
              \sum_{\mathrm{ls}} w_{\mathrm{ls}} \Sigma_{\mathrm{crit}}
              e_{\mathrm{t}}}{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}}
              w_{\mathrm{sys, l}} \sum_{\mathrm{ls}} w_{\mathrm{ls}}}

In the above expression, the first sum goes over all lenses and the second
sum over all sources that can form a pair with the lens. Additionally,
:math:`\Sigma_{\mathrm{crit}}` is the so-called critical surface density that
depends on the lens and source redshift, :math:`e_t` the shear component
tangential to the lens and :math:`w_{\mathrm{sys}}` and
:math:`w_{\mathrm{ls}}` systematic weights for lenses and lens-source pairs,
respectively. The former is often used to correct for biases in the lens
selection. The latter is used to maximize the signal-to-noise ratio. It is
calculated as

.. math::

    w_{\mathrm{ls}} = \frac{w_{\mathrm{s}}}{\Sigma_{\mathrm{crit}}^2}

where :math:`w_{\mathrm{s}}` is some weight associated to the noise of galaxy
shapes.

Correction Factors
------------------

Unfortunately, for virtually all lensing surveys, the above estimate is biased.
Thus, we need to apply a variety of correction terms to correction terms to get
an unbiased estimate of :math:`\Delta\Sigma`.

Photo-z Dilution
^^^^^^^^^^^^^^^^

When calculating the critical excess surface density
:math:`\Sigma_{\mathrm{crit}}`, we need to know both lens and source redshift.
However, in many lensing surveys, we only have very inaccurate and possibly
biased photometric redshifts for sources. Even in case of just inaccurate
redshifts, this can cause systematic biases in the calculation of
:math:`\Sigma_{\mathrm{crit}}`. We can correct for this via the photo-z
dilution correction, commonly described via :math:`f_{\rm bias}`. We refer the
reader to Leauthaud et al. (in prep) for a derivation of this correction
factor. It's defined via

.. math::

    f_{\rm bias}^{-1} =
        \frac{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}} w_{\mathrm{sys, l}}
              \sum_{\mathrm{ls}} w_{\mathrm{ls}} \frac{
                  \Sigma_{\rm c, ls, true}}{\Sigma_{\rm c, ls, photo-z}}
              }{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}}
              w_{\mathrm{sys, l}} \sum_{\mathrm{ls}} w_{\mathrm{ls}}} \, .

To apply the correction, we multiply the raw lensing signal by
:math:`f_{\rm bias}`. In order to run the photo-z dilution correction, you must
have passed a calibration catalog during the pre-compuation phase.

Boost Factor
^^^^^^^^^^^^

The photo-z dilution correction above only accounts for the global effect
of photo-z errors. However, it doesn't take into account that close to real
lenses there is an over-abundance of physically associated sources. As a
result, close to real lenses, a larger fraction of sources have true redshifts
:math:`z_{\rm l} \approx z_{\rm s}` than what was going into the calculation
of :math:`f_{\rm bias}`. For those physically associated sources
:math:`\Sigma_{\mathrm{crit}} \approx \infty`, i.e. they do not induce any
shear. Not taking this effect into account can lead to an underestimate of the
true lensing amplitue. To correct for this effect, we calculate the so-called
boost factor :math:`b` via

.. math::

    b(R) =
        \frac{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}} w_{\mathrm{sys, l}}
              \sum_{\mathrm{ls}} w_{\mathrm{ls}}
              }{\sum_{\mathrm{r} = 1}^{N_{\mathrm{r}}}
              w_{\mathrm{sys, r}} \sum_{\mathrm{rs}} w_{\mathrm{rs}}} \, .

Note that the sum in the denominator goes over a set of random lenses with the
same overall redshift and spatial distribution as the true lenses. Effectively,
we try to detect any over-abundance of sources close to lenses as opposed to
random points. Generally :math:`b \geq 1`. To apply the correction, we multiply
the raw lensing signal by the radially dependent boost factor :math:`b(R)`. In
order to run this correction, you need a random catalog.

Multiplicative Shear Bias
^^^^^^^^^^^^^^^^^^^^^^^^^

Individual ellipticies of galaxies can be biased. Thus, we need to correct for
this in the galaxy-galaxy lensing estimator. Typically, lensing surveys provide
a bias estimate :math:`m` for every object or group of objects. In order to
correct for this bias, one first calculates the average shear bias via

.. math::

    \bar{m}(R) =
        \frac{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}} w_{\mathrm{sys, l}}
              \sum_{\mathrm{ls}} w_{\mathrm{ls}} m_s
              }{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}}
              w_{\mathrm{sys, l}} \sum_{\mathrm{ls}} w_{\mathrm{ls}}} \, .

To apply the correction, we simply divide the raw lensing signal by
:math:`1 + \bar{m}(R)` in each radial bin.

Shear Responsivity
^^^^^^^^^^^^^^^^^^

The shear responsivity correction is necessary for some lensing surveys. For
this we need to calculate the shear responsivity :math:`\mathcal{R}` which
represents the response of the distortion to a small shear. :code:`dsigma`
calculates it via

.. math::

    \mathcal{R} = 1 -
        \frac{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}} w_{\mathrm{sys, l}}
              \sum_{\mathrm{ls}} w_{\mathrm{ls}} e_{{\rm rms}, s}^2
              }{\sum_{\mathrm{l} = 1}^{N_{\mathrm{l}}}
              w_{\mathrm{sys, l}} \sum_{\mathrm{ls}} w_{\mathrm{ls}}} \, .

Here, :math:`e_{{\rm rms}, s}^2`  is the intrinsic shape dispersion per
component for a source galaxy. To apply the shear responsivity correction, we
need to divide the final lensing signal by :math:`2\mathcal{R}`. This
correction is somewhat large but not applied to all surveys. For example, it
is applied to HSC and SDSS data, but not CFHTLenS or KiDS.

HSC Selection Bias
^^^^^^^^^^^^^^^^^^

As the name suggests, this correction is only applied to HSC lensing data.
Details can be found in section 5.6.2 of `Mandelbaum et al. (2018)
<https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.3170M/abstract>`_.

Random Subtraction
^^^^^^^^^^^^^^^^^^

Often, we can calculate the lensing signal around a set of random lenses with
the same overall redshift and spatial distribution as the true lenses. If these
random lenses are truly random, they have no correlation with the large-scale
matter field. As a result, they should give a lensing amplitude consistent with
0. However, lensing systematics like systematic shears can lead to non-zero
signals even for random points. Subtracting those points can alleviate such
systematic errors. Furthermore, even in the absence of lensing systematics,
subtracting randoms leads to a reduced variance of the lensing signal on large
scales, as shown in `Singh et al. (2017) <https://arxiv.org/abs/1611.00752>`_.

Note: When applying the random subtraction, we first calculate the lensing
amplitude around lenses including their own shear bias, shear responsivity and
HSC selection bias correction, if such corrections are also applied to the lens
sample, as well. Applying individual corrections for the random sample instead
just using the ones for the lens sample is important when the lens sample
itself is very small and the correction factors are noisy. After the random
signal is subtracted from the lens signal, we apply boost factor and photo-z
dilution corrections.

Total Signal
------------

The function :func:`dsigma.stacking.excess_surface_density` allows us to
calculate the total galaxy-galaxy lensing signal, including all correction
factors. In the example code below, we are analyzing the galaxy-galaxy
lensing signal with HSC data. We have performed the precomputation and the
results are stored in ``table_l``. For HSC, we need to apply shear bias, shear
responsivity and selection bias correction terms. The code below takes all of
that into account.

.. code-block:: python

    esd = excess_surface_density(table_l, return_table=True,
                                 shear_bias_correction='divide',
                                 shear_responsivity_correction=True,
                                 selection_bias_correction=True,
                                 boost_correction=False,
                                 random_subtraction=False,
                                 photo_z_dilution_correction=False)
    
    for key in esd.colnames:
        esd[key].format='.4f'
    
    esd.pprint_all()

The output looks as follows.

.. code-block:: none

    rp_min  rp_max    rp    delta sigma_raw delta sigma 1 + m    2R   1 + m_sel
    ------- ------- ------- --------------- ----------- ------ ------ ---------
     0.0500  0.0629  0.0561        152.1075     90.6249 0.8834 1.6784    1.0100
     0.0629  0.0792  0.0706        111.1277     66.2133 0.8844 1.6783    1.0070
     0.0792  0.0998  0.0889         95.0552     56.6578 0.8844 1.6777    1.0088
     0.0998  0.1256  0.1119         81.3570     48.5203 0.8875 1.6768    1.0087
     0.1256  0.1581  0.1409         79.6912     47.4898 0.8920 1.6781    1.0108
     0.1581  0.1991  0.1774         50.6677     30.1843 0.8924 1.6786    1.0112
     0.1991  0.2506  0.2233         53.3570     31.7837 0.8931 1.6788    1.0096
     0.2506  0.3155  0.2812         38.1371     22.7233 0.8922 1.6783    1.0094
     0.3155  0.3972  0.3540         35.8193     21.3457 0.8917 1.6781    1.0101
     0.3972  0.5000  0.4456         29.8587     17.7944 0.8913 1.6780    1.0099
     0.5000  0.6295  0.5610         20.8229     12.4090 0.8914 1.6780    1.0097
     0.6295  0.7924  0.7063         16.4423      9.7997 0.8910 1.6778    1.0098
     0.7924  0.9976  0.8891         15.7522      9.3880 0.8911 1.6779    1.0100
     0.9976  1.2559  1.1194         13.9731      8.3277 0.8912 1.6779    1.0099
     1.2559  1.5811  1.4092         10.6780      6.3637 0.8915 1.6780    1.0100
     1.5811  1.9905  1.7741          8.1250      4.8419 0.8917 1.6780    1.0100
     1.9905  2.5059  2.2334          6.3451      3.7813 0.8916 1.6780    1.0100
     2.5059  3.1548  2.8117          3.8842      2.3147 0.8918 1.6781    1.0100
     3.1548  3.9716  3.5397          3.6218      2.1583 0.8917 1.6781    1.0100
     3.9716  5.0000  4.4563          2.2735      1.3549 0.8917 1.6780    1.0100
     5.0000  6.2946  5.6101          2.0328      1.2114 0.8917 1.6780    1.0100
     6.2946  7.9245  7.0627          1.3313      0.7934 0.8919 1.6781    1.0100
     7.9245  9.9763  8.8914          1.1858      0.7066 0.8920 1.6781    1.0101
     9.9763 12.5594 11.1936          0.6759      0.4028 0.8922 1.6782    1.0101
    12.5594 15.8114 14.0919          0.9251      0.5512 0.8922 1.6782    1.0101
    15.8114 19.9054 17.7407          1.0275      0.6122 0.8922 1.6783    1.0101
    19.9054 25.0594 22.3342          1.3240      0.7888 0.8921 1.6784    1.0102
    25.0594 31.5479 28.1171          1.7110      1.0194 0.8920 1.6785    1.0103
    31.5479 39.7164 35.3973          1.6912      1.0075 0.8918 1.6786    1.0105
    39.7164 50.0000 44.5625          1.8857      1.1233 0.8915 1.6788    1.0106
