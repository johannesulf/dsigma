![logo](https://raw.githubusercontent.com/johannesulf/dsigma/main/docs/dsigma.png)

[![Unit Testing Status](https://img.shields.io/github/actions/workflow/status/johannesulf/dsigma/tests.yml?branch=main&label=tests)](https://github.com/johannesulf/dsigma/actions)
[![Documentation Status](https://img.shields.io/readthedocs/dsigma)](https://dsigma.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/dsigma?color=blue)](https://pypi.org/project/dsigma/)
[![License: MIT](https://img.shields.io/github/license/johannesulf/dsigma?color=blue)](https://raw.githubusercontent.com/johannesulf/dsigma/main/LICENSE)
![Language: Python](https://img.shields.io/github/languages/top/johannesulf/dsigma)
[![ASCL](https://img.shields.io/badge/ascl-2204.006-orange.svg?style=flat)](https://ascl.net/2204.006)
![Downloads](https://img.shields.io/pypi/dm/dsigma)

`dsigma` is an easy-to-use python package for measuring gravitational galaxy-galaxy lensing. Using a lensing catalog, it estimates excess surface density around a population of lenses, such as galaxies in the Sloan Digital Sky Survey or the Baryon Oscillation Spectroscopic Survey. It has a broadly applicable API and can utilize data from the Dark Energy Survey (DES), the Kilo-Degree Survey (KiDS), and the Hyper Suprime-Cam (HSC) lensing surveys, among others. With core computations written in C, `dsigma` is very fast. Additionally, `dsigma` provides out-of-the-box support for estimating covariances with jackknife resampling and calculating various summary statistics. Below is a plot showing the excess surface density around galaxies in the CMASS sample calculated with `dsigma`.

![plot](https://raw.githubusercontent.com/johannesulf/dsigma/main/docs/plot.png)

## Authors

* Johannes Lange
* Song Huang

## Documentation

Documentation for `dsigma` with concept introductions, examples, and API documentation is available on [readthedocs](https://dsigma.readthedocs.io/).

## Attribution

`dsigma` is listed in the [Astronomy Source Code Library](https://ascl.net/2204.006). If you find the code useful in your research, please cite [Lange & Huang (2022)](https://ui.adsabs.harvard.edu/abs/2022ascl.soft04006L/abstract).

## License

`dsigma` is licensed under the MIT License.
