Introduction
============

We will show how to calculate galaxy-galaxy lensing amplitudes using four widely-used lensing surveys: the Dark Energy Camera All Data Everywhere (DECADE) cosmic shear project, the Dark Energy Survey (DES), the Hyper Suprime-Cam (HSC) survey, and the Kilo-Degree Survey (KiDS). For this example, we will use galaxies from the Baryon Oscillation Spectroscopic Survey (BOSS) as lenses. If everything works correctly, the lensing amplitude :math:`\Delta\Sigma` for the same lens samples should be roughly consistent between the various lensing surveys.

BOSS Lens Catalog
-----------------

The BOSS target catalogs are publicly available from the `SDSS data server <https://data.sdss.org/sas/dr12/boss/lss/>`_. In the following, we will assume that all relevant lens
(:code:`galaxy_DR12v5_CMASSLOWZTOT_*.fits.gz`) and random files (:code:`random0_DR12v5_CMASSLOWZTOT_*.fits.gz`) are in the working directory. The following code reads the data and puts it in a format easily understandable by :code:`dsigma`.

.. code-block:: python

    from astropy.table import Table, vstack

    table_l = vstack([Table.read('galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                      Table.read('galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz')])
    table_r = vstack([Table.read('random0_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                      Table.read('random0_DR12v5_CMASSLOWZTOT_North.fits.gz')])
    keys = dict(z='Z', ra='RA', dec='DEC')
    for table in [table_l, table_r]:
        for new_key, old_key in keys.items():
            table.rename_column(old_key, new_key)
        table.keep_columns(keys.keys())
        table['w_sys'] = 1

    table_l = table_l[table_l['z'] >= 0.15]
    table_r = table_r[table_r['z'] >= 0.15][::5]

Note that we only process every 5th random to reduce computation time and memory use. Even so, we still have ten times more randoms than lenses, which should suffice `(Singh et al., 2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.3827S/abstract>`_. Note also that we set the systematic weights to unity here. For science analyses, these should be chosen to correct for observational biases in BOSS.
