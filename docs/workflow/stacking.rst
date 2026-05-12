Stacking the Signal
===================

The function :func:`~dsigma.stacking.excess_surface_density` allows us to calculate the total galaxy-galaxy lensing signal, including all correction factors. The following example computes the galaxy-galaxy lensing signal with HSC data. We have performed the precomputation, and the results are stored in ``table_l``. For HSC Y3, we need to apply shear bias, shear responsivity, and selection bias correction terms.

.. code-block:: python

    from astropy import units as u
    from astropy.cosmology import units as cu

    from dsigma.stacking import excess_surface_density
    
    result = excess_surface_density(
        table_l, return_table=True, scalar_shear_response_correction=True,
        shear_responsivity_correction=True, selection_bias_correction=True)
    
    for key in result.colnames:
        # Drop little h from units.
        if key[:2] == 'rp':
            result[key] = result[key].to(u.Mpc, cu.with_H0(Planck15.H0))
        if key[:2] == 'ds':
            result[key] = result[key].to(u.Msun / u.pc**2, cu.with_H0(Planck15.H0))
        if key != 'n_pairs':
            result[key].format='.3f'
    
    result.pprint_all()

The output looks something like this. ``ds_raw`` is the original, uncorrected lensing amplitude whereas ``ds`` denotes the corrected one. ``1 + m``, ``2R``, and ``1 + m_sel`` denote the shear bias, shear responsivity, and selection bias correction terms, respectively.

.. code-block:: console

    rp_min rp_max  n_pairs       ds_raw          ds       z_l   z_s   1+m    2R  1+m_sel
     Mpc    Mpc              solMass / pc2 solMass / pc2                                
    ------ ------ ---------- ------------- ------------- ----- ----- ----- ----- -------
     0.100  0.158     113251        71.999        48.512 0.414 1.092 0.875 1.682   1.008
     0.158  0.251     289741        50.827        34.175 0.413 1.094 0.876 1.682   1.010
     0.251  0.398     736162        34.026        22.934 0.414 1.095 0.875 1.682   1.009
     0.398  0.631    1851563        21.676        14.602 0.414 1.095 0.875 1.682   1.009
     0.631  1.000    4649960        14.659         9.872 0.414 1.095 0.875 1.682   1.009
     1.000  1.585   11637038         9.394         6.325 0.414 1.095 0.875 1.682   1.009
     1.585  2.512   29179989         5.924         3.989 0.414 1.095 0.875 1.682   1.009
     2.512  3.981   73071540         3.046         2.051 0.414 1.095 0.875 1.682   1.009
     3.981  6.310  183114638         1.844         1.241 0.413 1.095 0.875 1.682   1.009
     6.310 10.000  459001166         1.209         0.814 0.413 1.095 0.875 1.682   1.009
    10.000 15.849 1150384474         0.865         0.582 0.413 1.095 0.875 1.682   1.009
    15.849 25.119 2878215211         0.592         0.398 0.414 1.095 0.875 1.682   1.009
    25.119 39.811 7184234609         0.442         0.298 0.414 1.095 0.875 1.682   1.009