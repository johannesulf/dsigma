Introduction
============

We will discuss various examples of calculating galaxy-galaxy lensing signals in the following. Precisely, we will calculate galaxy-galaxy lensing signals for the three extensive imaging surveys: the Dark Energy Survey (DES), the Hyper Suprime-Cam (HSC) survey, and the Kilo-Degree Survey (KiDS). We will use galaxies from the Baryon Oscillation Spectroscopic Survey (BOSS) as lenses. Specifically, we will qualitatively reproduce the results of the Lensing Without Borders project. To this end, we will cross-correlate the same sets of BOSS lens galaxies with different imaging surveys. If everything works correctly, the lensing amplitude :math:`\Delta\Sigma` for the same lens samples should be consistent between the various imaging surveys.

BOSS Lens Catalog
-----------------

The BOSS target catalogs are publicly available from the `SDSS data server <https://data.sdss.org/sas/dr12/boss/lss/>`_. In the following, we will assume that all relevant lens
(:code:`galaxy_DR12v5_CMASSLOWZTOT_*.fits.gz`) and random files (:code:`random0_DR12v5_CMASSLOWZTOT_*.fits.gz`) are in the working directory. The following code reads the data and puts it in a format easily understandable by :code:`dsigma`.

.. code-block:: python

    from astropy.table import Table, vstack
    from dsigma.helpers import dsigma_table

    table_l = vstack([Table.read('galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                      Table.read('galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz')])
    table_l = dsigma_table(table_l, 'lens', z='Z', ra='RA', dec='DEC',
                           w_sys=1)
    table_l = table_l[table_l['z'] >= 0.15]

    table_r = vstack([Table.read('random0_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                      Table.read('random0_DR12v5_CMASSLOWZTOT_North.fits.gz')])
    table_r = dsigma_table(table_r, 'lens', z='Z', ra='RA', dec='DEC',
                           w_sys=1)[::5]
    table_r = table_r[table_r['z'] >= 0.15]

Note that we only process every 5th random. We do this to reduce computation time and memory use. Even when only using every 5th random, we still have ten times more randoms than lenses which should suffice `(Singh et al., 2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.3827S/abstract>`_.
