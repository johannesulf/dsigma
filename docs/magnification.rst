Lens Magnification
==================

Galaxy-galaxy lensing measures the mean tangential shear of source galaxies at redshift :math:`z_{\rm s}` induced by lens galaxies at redshift :math:`z_{\rm l} < z_{\rm s}`. However, in principle, all foreground structures at :math:`z_{\rm f} \neq z_{\rm l}` will contribute to the shear distortion of source galaxies. To first order, foreground structures are uncorrelated with the lens plane such that the expected tangential shear around lenses coming from foreground structures is zero, on average.

However, the magnification of lens galaxies by foreground structures at :math:`z_{\rm f} < z_{\rm l}` will induce spatial correlations such that the tangential shear induced by foreground structures is expected to be non-zero. This effect is called lens magnification and can be an important contributor to the mean tangential shear around lens galaxies. The strength of this effect depends on the response of lenses to magnification, i.e., how much more likely a galaxy is to make it into the lens sample if it is gravitationally lensed. This is quantified by the parameter :math:`\alpha`. We refer the reader to `Unruh et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020A%26A...638A..96U/abstract>`_ for a detailed investigation of this effect.

`dsigma` implements the estimate for lens magnification described in this publication. For example, the code below reproduces the pink line in Fig. 5 of Unruh et al. (2019). To estimate the power spectrum, it relies on `CAMB <https://camb.readthedocs.io>`_.

.. code-block:: python

    import camb
    import matplotlib.pyplot as plt
    import numpy as np

    from astropy import units as u
    from dsigma.physics import lens_magnification_shear_bias

    z_d = 0.41
    z_s = 0.99
    alpha_d = 2.71

    h = 0.73
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=0.045 * h**2, omch2=0.205 * h**2)
    pars.InitPower.set_params(ns=1.0, As=2.83e-9)
    pars.set_matter_power(redshifts=np.linspace(z_d, 0, 10), kmax=2000.0,
                          nonlinear=True)
    camb_results = camb.get_results(pars)

    theta = np.geomspace(0.5, 20, 30) * u.arcmin
    d_gamma = [lens_magnification_shear_bias(t, alpha_d, z_d, z_s, camb_results)
               for t in theta]
    plt.plot(theta, d_gamma)

    plt.title(rf'$z_d = {z_d:.2f}, z_s = {z_s:.2f}, \alpha_d = {alpha_d:.2f}$')
    plt.xscale('log')
    plt.xlabel(r'$\theta$ in arcmin')
    plt.ylabel(r'Bias $\Delta \gamma_t$')
    plt.xlim(0.5, 20)

.. image:: magnification.png
   :width: 80 %
   :align: center

In the same way, we can use :func:`dsigma.stacking.lens_magnification_bias` function to estimate the leans magnification bias. In this case, to calculate the additive shear bias, `dsigma` uses the mean lens and source redshift. Furthermore, to convert this into an estimate of the bias in :math:`\Delta\Sigma`, it multiplies this with the mean critical surface density. Note that the lens magnification bias is purely additive, i.e., it can be corrected for by subtracting the bias estimate from the total lensing signal.