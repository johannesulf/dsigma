BOSS Lens Catalog
=================

In the following, we will walk through various examples of calculating
galaxy-galaxy lensing signals. Specifically, we will qualitatively reproduce
results of the Lensing Without Borders project. To this end, we will
cross-correlate BOSS lens galaxies with different imaging surveys. If
everything works correctly, the lensing amplitude for the same lens samples
should be comparable between the different imaging surveys.

The BOSS target catalogs are publicly available from the
`SDSS data server <https://data.sdss.org/sas/dr12/boss/lss/>`_. In the
following, we will assume that all relevant lens
(:code:`galaxy_DR12v5_CMASSLOWZ_*.fits.gz`) and random files
(:code:`random0_DR12v5_CMASSLOWZ_*.fits.gz`) are in the working directory. The
following code reads in the data and puts it in the a format easily
understandable by :code:`dsigma`. ::

    from astropy.table import Table, vstack
    from dsigma.helpers import dsigma_table

    table_l = vstack([Table.read('galaxy_DR12v5_CMASSLOWZ_South.fits.gz'),
                      Table.read('galaxy_DR12v5_CMASSLOWZ_North.fits.gz')])
    table_l = dsigma_table(table_l, 'lens', z='Z', ra='RA', dec='DEC',
                           w_sys=1)
    table_l = table_l[table_l['z'] >= 0.15]

    table_r = vstack([Table.read('random0_DR12v5_CMASSLOWZ_South.fits.gz'),
                      Table.read('random0_DR12v5_CMASSLOWZ_North.fits.gz')])
    table_r = dsigma_table(table_r, 'lens', z='Z', ra='RA', dec='DEC',
                           w_sys=1)
    table_r = table_r[table_r['z'] >= 0.15]
