A Simple Python Galaxy-Galaxy Lensing Pipeline
==============================================

:code:`dsigma` is an easy-to-use pipeline for analyzing galaxy-galaxy lensing.
The package is written in python, has a broadly applicable API and is
optimized for computational efficiency. While originally intended to be used
with the shape catalog of the Hyper-Suprime Cam (HSC) survey, it is intended
to also work with surveys like the Canada-France-Hawaii Telescope Lensing
Survey (CFHTLenS) or the Kilo-Degree Survey (KiDS).

Authors
-------

* Johannes Lange
* Song Huang

.. toctree::
   :caption: Getting Started

   installation

.. toctree::
   :caption: Workflow

   preparation
   precomputation
   stacking
   resampling

.. toctree::
   :caption: API Documentation

   api/helpers
   api/jackknife
   api/physics
   api/precompute
   api/stacking
   api/surveys