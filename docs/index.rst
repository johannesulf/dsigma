.. image:: dsigma.png
   :width: 80 %
   :align: center

Overview
========

:code:`dsigma` is an easy-to-use python package for measuring gravitational galaxy-galaxy lensing. Using a lensing catalog, it estimates excess surface density around a population of lenses, such as galaxies in the Sloan Digital Sky Survey or the Baryon Oscillation Spectroscopic Survey. It has a broadly applicable API and can utilize data from the Dark Energy Survey (DES), the Kilo-Degree Survey (KiDS), and the Hyper-Suprime Cam (HSC) lensing surveys, among others. With core computations written in C, :code:`dsigma` is very fast. Additionally, :code:`dsigma` provides out-of-the-box support for estimating covariances with jackknife resampling and calculating various summary statistics. Below is a plot showing the excess surface density around galaxies in the CMASS sample calculated with :code:`dsigma`.

.. image:: plot.png
   :width: 80 %
   :align: center

.. toctree::
    :hidden:
    :caption: Getting Started
    :maxdepth: 1

    start/installation
    start/background

.. toctree::
    :hidden:
    :caption: Workflow
    :maxdepth: 1

    workflow/preparation
    workflow/precomputation
    workflow/stacking
    workflow/resampling
    workflow/magnification

.. toctree::
    :caption: Applications
    :maxdepth: 1
    :hidden:

    applications/intro
    applications/des_y3
    applications/hsc_y3
    applications/kids-1000

.. toctree::
    :caption: API Documentation
    :maxdepth: 1
    :hidden:

    api/helpers
    api/jackknife
    api/physics
    api/precompute
    api/stacking
    api/surveys

Authors
-------

* Johannes Lange
* Song Huang

Attribution
-----------

:code:`dsigma` is listed in the `Astronomy Source Code Library <https://ascl.net/2204.006>`_. If you find the code useful in your research, please cite `Lange & Huang (2022) <https://ui.adsabs.harvard.edu/abs/2022ascl.soft04006L/abstract>`_.

License
-------

:code:`dsigma` is licensed under the MIT License.
