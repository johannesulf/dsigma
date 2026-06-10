<div align="center">

![logo](https://raw.githubusercontent.com/johannesulf/dsigma/main/docs/dsigma.png)

[![Unit Testing Status](https://img.shields.io/github/actions/workflow/status/johannesulf/dsigma/tests.yml?branch=main&label=tests)](https://github.com/johannesulf/dsigma/actions)
[![Documentation Status](https://img.shields.io/readthedocs/dsigma)](https://dsigma.readthedocs.io/en/latest/)
[![Code Coverage](https://img.shields.io/coverallsCoverage/github/johannesulf/dsigma)](https://coveralls.io/github/johannesulf/dsigma?branch=main)
![Downloads](https://img.shields.io/pypi/dm/dsigma)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjohannesulf%2Fdsigma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![PyPI - Version](https://img.shields.io/pypi/v/dsigma)
[![License: MIT](https://img.shields.io/github/license/johannesulf/dsigma?color=blue)](https://raw.githubusercontent.com/johannesulf/dsigma/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20378643.svg)](https://doi.org/10.5281/zenodo.20378643)

</div>

``dsigma`` is an easy-to-use Python package for measuring gravitational galaxy-galaxy lensing. Using a lensing catalog, it estimates excess surface density around a population of lenses, such as galaxies in the Sloan Digital Sky Survey or the Baryon Oscillation Spectroscopic Survey. It has a flexible API and can utilize data from, DECADE, the Dark Energy Survey (DES), the Kilo-Degree Survey (KiDS), and the Hyper Suprime-Cam (HSC) lensing surveys, among others. With core computations written in C, ``dsigma`` is very fast. Additionally, ``dsigma`` provides out-of-the-box support for estimating covariances with jackknife resampling and calculating various summary statistics.

![plot](https://raw.githubusercontent.com/johannesulf/dsigma/main/docs/plot.png)

## Authors

* Johannes Lange
* Song Huang

## Installation

The easiest way to install ``dsigma`` is to use ``pip`` to install the latest stable version from the Python Package Index (PyPI).

    pip install dsigma

Alternatively, you can install the latest development version from GitHub.

    pip install git+https://github.com/johannesulf/dsigma

## Documentation

Documentation for ``dsigma`` with concept introductions, examples, and API documentation is available on [readthedocs](https://dsigma.readthedocs.io/).

## Attribution

``dsigma`` is listed in the [Astronomy Source Code Library](https://ascl.net/2204.006). If you find the code useful in your research, please cite [Lange & Huang (2022)](https://ui.adsabs.harvard.edu/abs/2022ascl.soft04006L/abstract).

## License

``dsigma`` is licensed under the MIT License.

## Generative AI

Generative AI was used to search for potential bugs in the code and to improve the documentation by finding typos and suggesting minor rewrites. No part of the ``dsigma`` code itself was written entirely or in parts by AI.
