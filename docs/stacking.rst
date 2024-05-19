Stacking the Signal
===================

The function :func:`dsigma.stacking.excess_surface_density` allows us to calculate the total galaxy-galaxy lensing signal, including all correction factors. In the example code below, we are analyzing the galaxy-galaxy lensing signal with HSC data. We have performed the pre-computation, and the results are stored in ``table_l``. For HSC, we need to apply shear bias, shear responsivity, and selection bias correction terms. The code below takes all of that into account.

.. code-block:: python

    esd = excess_surface_density(table_l, return_table=True,
                                 scalar_shear_response_correction=True,
                                 shear_responsivity_correction=True,
                                 boost_correction=False,
                                 random_subtraction=False,
                                 photo_z_dilution_correction=False)
    
    for key in esd.colnames:
        esd[key].format='.4f'
    
    esd.pprint_all()

The output looks something like this.

.. code-block:: console

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
