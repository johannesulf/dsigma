---
title: '`dsigma`: a user-friendly galaxy--galaxy lensing package'
tags:
  - Python
  - astronomy
  - cosmology
  - galaxies
  - gravitational lensing
  - dark matter
authors:
  - name: Johannes U. Lange
    orcid: 0000-0002-2450-1366
    surname: Lange
    affiliation: 1
    corresponding: true
  - name: Song Huang
    orcid: 0000-0003-1385-7591
    affiliation: 2
  - name: Alexie Leauthaud
    orcid: 0000-0002-3677-3617
    affiliation: 3
affiliations:
  - name: Department of Physics, American University, Washington, DC 20016, USA
    index: 1
  - name: Department of Astronomy, Tsinghua University, Beijing 100084, China
    index: 2
  - name: Department of Astronomy and Astrophysics, University of California, Santa Cruz, Santa Cruz, CA 95065, USA
    index: 3
date: 22 May 2026
bibliography: paper.bib
---

# Summary

Galaxy--galaxy lensing, a prediction of Einstein's theory of gravity, is the distortion of galaxy images by other foreground galaxies and a primary probe of cosmology and astrophysics. In recent years, data sets containing tens to hundreds of millions of galaxy images have been made available, allowing researchers to measure galaxy--galaxy lensing with unprecedented precision. However, in practice, estimating the lensing amplitude is computationally challenging due to the large number of galaxies involved. Additionally, significant expertise in cosmology, geometry and gravitational lensing surveys systematics is required to produce accurate measurements. `dsigma` alleviates this significant burden, providing a user-friendly, efficient, and modular Python framework to measure galaxy--galaxy lensing with state-of-the-art cosmological data sets. With core functionality written in C and tight integration with `astropy`, `dsigma` allows researchers at any career stages to quickly produce publication-ready galaxy--galaxy lensing measurements.

# Statement of need

Large spectroscopic galaxy surveys such as the Sloan Digital Sky Survey (SDSS) and the Dark Energy Spectroscopic Instrument (DESI) survey are indispensable data sets for studying galaxy formation and cosmology. Gravitational lensing can be used to infer the distribution of mass around these galaxies. In galaxy--galaxy lensing (GGL), the mass around a set of foreground "lens" galaxies causes lensing distortions in the apparent shapes of background "source" galaxies. Mathematically, the average tangential shear, $\gamma_\mathrm{t} \ll 1$, of source shapes is related to the so-called excess surface density $\Delta\Sigma$ around lens galaxies via
\begin{equation}
\gamma_\mathrm{t} = \frac{\Delta\Sigma}{\Sigma_\mathrm{crit}},
\end{equation}
where $\Sigma_\mathrm{crit}$ is the critical surface density that depends on cosmological parameters as well as the redshifts of lenses and sources. In addition to spectroscopic galaxy surveys which provide "lenses", there are a number of publicly available "shape catalogs" that can be used as source galaxies, including from the the Dark Energy Survey (DES), the Hyper Suprime-Cam (HSC) Subaru Strategic Program, the Kilo-Degree Survey (KiDS), and the Dark Energy Camera All Data Everywhere (DECADE) project.

In principle, calculating the GGL amplitude $\Delta\Sigma$ primarily involves taking a simple weighted average of $e_\mathrm{t} \Sigma_\mathrm{crit}$, where $e_\mathrm{t}$ is the measured tangential ellipticity, over all suitable lens-source galaxy pairs. In practice, care needs to be taken to correctly calculate the relevant angular and cosmological quantities from the original catalogs. Additionally, one needs to take into account a variety of survey-specific correction factors to arrive at an unbiased estimate of the GGL amplitude. Finally, given that modern galaxy surveys can contain hundreds of millions of galaxies, even a simple average can quickly become computationally challenging. Thus, writing an accurate and efficient GGL pipeline from scratch is a significant effort and may prevent some researchers from making use of lensing data in their studies. `dsigma` solves this problem by providing an easy-to-use pipeline that can be interfaced with publicly available data sets to estimate $\Delta\Sigma$.

# State of the field

A number of packages exist for measuring gravitational lensing observables from existing shape catalogs, with `TreeCorr`^[<https://github.com/rmjarvis/treecorr>] [@2004MNRAS.352.338J] being one of the most widely-used. However, there are a number of critical differences between `TreeCorr` (and similar packages) and `dsigma`. Most importantly, `TreeCorr` calculates angular correlation functions and does not explicitly include cosmological calculations. As such, `TreeCorr` natively calculates $\gamma_\mathrm{t}$ as a function of the angle $\theta$ whereas `dsigma` is meant to compute $\Delta\Sigma$ as a function of the physical separation $r_\mathrm{p}$. Additionally, `TreeCorr` is a more general framework for calculating angular correlation functions, not just GGL. For example, unlike `dsigma`, `TreeCorr` does not directly compute relevant correction factors for the GGL amplitude out-of-the-box and, instead, leaves those to the user. Ultimately, while one could use `TreeCorr` to construct estimators for $\Delta\Sigma(r_\mathrm{p})$ with relevant correction factors, this would be a significant effort requiring deep expertise in weak lensing calculations, something `dsigma` is designed to alleviate.  Finally, on a technical side, `TreeCorr` uses a tree algorithm to approximate the GGL amplitude whereas `dsigma` is an exact pair counter.

Beyond codes calculating angular correlation functions, there are only a few open-source packages to explicitly compute $\Delta\Sigma(r_\mathrm{p})$, including `swot`^[<https://github.com/jcoupon/swot>] and `xshear`^[<https://github.com/esheldon/xshear>]. However, neither is actively developed nor designed to compute all relevant correction factors for leading weak lensing surveys. To our knowledge, `dsigma` is the only one-stop solution to compute $\Delta\Sigma(r_\mathrm{p})$ with all state-of-the-art shape catalogs currently publicly available.

# Software design

The following core design principles drive the development of `dsigma`:

- **Ease of use**: The user-facing API is written in pure Python and makes extensive use of `astropy`. For example, galaxy catalogs and cosmologies are passed via `astropy` tables and cosmologies, respectively. Additionally, `dsigma` makes use of `astropy` units which helps newcomers avoid common pitfalls such as the use of "little h" [@2013PASA.30.52C]. `dsigma` also comes with scripts that process publicly available lensing catalogs into a common format understood by `dsigma`. Ultimately, the goal is to allow users to create GGL amplitudes quickly without having to know all the details of the shape catalogs first.
- **Modularity**: `dsigma` can be used with several weak lensing catalogs, including data DES, HSC, KiDS, and DECADE. While the estimator for the uncorrected lensing amplitude is universal, different lensing surveys use distinct correction factors to account for, e.g., shear or selection biases. As much as possible, `dsigma` unifies these correction factors. This reduces code maintenance and also allows users to use `dsigma` with lensing catalogs it was not originally tested with.
- **Speed**: For modern galaxy surveys, the number of suitable lens-source pairs for GGL can reach trillions. To make this computation fast, the core computational loop of `dsigma` is written in C via `cython` and can make use of multiprocessing. Additionally, expensive calculations such as trigonometric functions and cosmological distances are calculated for each lens and source galaxy instead of each lens-source pair since the latter often largely outnumber the former. Due to these optimizations, GGL amplitudes with state-of-the-art stage-III lensing surveys can be calculated in matter of minutes, including random subtractions [@2017MNRAS.471.3827S].

# Research impact statement

Originally designed to support the core developer's research [@2022MNRAS.510.6150L; @2023MNRAS.520.5373L; @2025PhRvD.111l3524K], `dsigma` has now been used by multiple other research groups [@2024A&A.686A.196S; @2024A&A.690A.221P; @2025arXiv250910455S; @2025arXiv250920434T; @2025arXiv250920458T; @2025ApJ.992.171C; @2025MNRAS.543.1393M; @2025arXiv251020896S; @2025arXiv251214636A]. Additionally, it is the default GGL pipeline used in the DESI collaboration [@2024OJAp.7E.57L; @2025arXiv250621677H; @2025OJAp.8E.149R; @2026OJAp.961342P]. Finally, the `dsigma` code has also been forked around one dozen times, often with additional modifications. Ultimately, we expect `dsigma` to continue being used by multiple research groups, including in combination with new data from the Euclid satellite, the Nancy Grace Roman Space Telescope, and the Vera C. Rubin Observatory.

# AI usage disclosure

Claud Sonnet 4.6 was used to search for potential bugs in the code. Despite flagging multiple issues, only one genuine minor bug was found. Claud Sonnet was also used to improve the documentation by finding typos and suggesting minor rewrites. No part of the code itself was written entirely or in parts by AI.

# Acknowledgements

We thank Rachel Mandelbaum and Jean Coupon for helping with some of the early design choices. Additionally, we thank Chris Blake for comparing the results of `dsigma` against his own independent pipeline. Finally, we are grateful to the DES, KiDS, HSC, and DECADE collaborations for making their data publicly available and providing excellent documentation.

# References
