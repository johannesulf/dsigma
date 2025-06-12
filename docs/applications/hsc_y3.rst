Hyper Suprime-Cam (HSC)
=======================

.. note::
    This guide has not been inspected or endorsed by the HSC collaboration.

This tutorial will teach us how to cross-correlate BOSS lens galaxies with lensing catalogs from the HSC survey. We will work with the Y3 data release, also known as PDR3 or S19A. The guide for the older Y1 release is :doc:`here <hsc_y1>`.

Downloading the Data
--------------------

First, we need to download the necessary HSC data files. Head to the `HSC data release site <https://hsc-release.mtk.nao.ac.jp/doc/>`_ and register for an account if you haven't done so already.

We follow the HSC recommendation and use the following SQL command to download all necessary data from the `CAS search <https://hsc-release.mtk.nao.ac.jp/datasearch/>`_. Make sure to select PDR3 as the release; otherwise, the SQL command will fail. Also, choose FITS as the output format.

.. code-block:: sql

    select
    b.*, c.i_ra, c.i_dec, a.i_hsmshaperegauss_e1, a.i_hsmshaperegauss_e2
    from
    s19a_wide.meas2 a
    inner join s19a_wide.weaklensing_hsm_regauss b using (object_id)
    inner join s19a_wide.meas c using (object_id)

In the following, this tutorial assumes that the above source catalog is saved as :code:`hsc_y3.fits` in the working directory.

In addition to the source catalog, we need the redshift distribution, :math:`n(z)`, for all samples which can be downloaded `here <https://hsc-release.mtk.nao.ac.jp/archive/filetree/shape_catalog_y3/li23/nz/nz.fits>`_.
