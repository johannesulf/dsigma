---
title: 'dsigma: an easy-to-use galaxy--galaxy lensing package'
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
affiliations:
 - name: Department of Physics, American University, Washington, DC 20016, USA
   index: 1
 - name: Department of Astronomy, Tsinghua University, Beijing 100084, China
   index: 2
date: 18 May 2025
bibliography: paper.bib
---

# Summary

TBD

# Statement of need

TBD

# State of the field

A number of packages exist for calculating weak-lensing correlation functions, including GGL. One of the most widely-used among them is `TreeCorr`^[<https://github.com/rmjarvis/treecorr>] [@2004MNRAS.352.338J]. There are a number of critical differences between `TreeCorr` (and similar packages) and `dsigma`. Most importantly, `TreeCorr` calculates angular correlation functions and does not explicitly include cosmological calculations. As such, `TreeCorr` natively calculates $\gamma_\mathrm{t}(\theta)$ whereas `dsigma` is meant to compute $\Delta\Sigma(r_\mathrm{p})$. Additionally, `TreeCorr` is a more general framework for calculating angular correlation functions. For example, unlike `dsigma`, `TreeCorr` does not directly compute relevant correction factors for the GGL amplitude out-of-the-box and, instead, leaves those to the user. Ultimately, while one could use `TreeCorr` to construct estimators for $\Delta\Sigma(r_\mathrm{p})$ with all the necessary correction factors, this would be a significant effort requiring deep expertise in weak lensing calculations, something `dsigma` is designed to alleviate.  Finally, on a technical side, `TreeCorr` uses a tree algorithm to approximate the GGL amplitude whereas `dsigma` is an exact pair counter.

Besides codes calculating angular correlation functions, there are only a few open-source packages to explicitly compute $\Delta\Sigma(r_\mathrm{p})$, including `swot`^[<https://github.com/jcoupon/swot>] and `xshear`^[<https://github.com/esheldon/xshear>]. However, neither is actively developed nor designed to compute all relevant correction factors for leading weak lensing surveys. To our knowledge, `dsigma` is the only one-stop solution to compute $\Delta\Sigma(r_\mathrm{p})$ with all state-of-the-art lensing catalog currently publicly available.

# Software design

The following core design principles drive the development of `dsigma`:

- **Ease of use**: The user-facing API is written in pure Python and makes extensive use of `astropy`. For example, galaxy catalogs and cosmologies are passed via `astropy` tables and cosmologies, respectively. Additionally, `dsigma` makes use of `astropy` units which helps newcomers avoid common pitfalls such as the use of "little h" [@2013PASA.30.52C]. `dsigma` also comes with scripts that process publicly available lensing catalogs into a common format understood by `dsigma`. Ultimately, the goal is to allow users to create GGL amplitudes quickly without having to know all the details of the shape catalogs first.
- **Modularity**: `dsigma` can be used with several weak lensing catalogs, including data from the Dark Energy Survey (DES), the Kilo-Degree Survey (KiDS), the Hyper Suprime-Cam (HSC) Subaru Strategic Program, and the Dark Energy Camera All Data Everywhere (DECADE) project. While the estimator for the uncorrected lensing amplitude is universal, different lensing surveys use distinct correction factors to account for, e.g., shear or selection biases. As much as possible, `dsigma` unifies these correction factors. This reduces code maintenance and also allows users to use `dsigma` with lensing catalogs it was not originally tested with.
- **Speed**: For modern galaxy surveys, the number of suitable lens-source pairs for GGL can reach trillions. To make this computation fast, the core computational loop of `dsigma` is written in C via `cython` and can make use of multiprocessing. Additionally, expensive calculations such as trigonometric functions and cosmological distances are calculated for each lens and source galaxy instead of each lens-source pair since the latter often largely outnumber the former. Due to these optimizations, GGL amplitudes with state-of-the-art stage-III lensing surveys can be calculated in matter of minutes, including random subtractions [@2017MNRAS.471.3827S].

# Research impact statement

Originally designed to support the core developer's research [@2022MNRAS.510.6150L; @2023MNRAS.520.5373L; @2025PhRvD.111l3524K], `dsigma` has now been used by multiple other research groups [@2024A&A.686A.196S; @2024A&A.690A.221P; @2025arXiv250910455S; @2025arXiv250920434T; @2025arXiv250920458T; @2025ApJ.992.171C; @2025MNRAS.543.1393M; @2025arXiv251020896S; @2025arXiv251214636A]. Additionally, it is the default GGL pipeline used in the Dark Energy Spectroscopic Instrument (DESI) collaboration [@2024OJAp.7E.57L; @2025arXiv250621677H; @2025OJAp.8E.149R; @2026OJAp.961342P]. Finally, the `dsigma` code has also around one dozen forks, many of which with additional commits. We expect `dsigma` to continue being used by multiple research groups, including in combination with new data from the Euclid satellite, the Nancy Grace Roman Space Telescope, and the Rubin Observatory.

# AI usage disclosure

Claud Sonnet 4.6 was used to search for potential bugs in the code. Despite flagging multiple issues, only one genuine minor bug was found. Claud Sonnet was also used to improve the documentation by finding typos and suggesting minor rewrites. No part of the code itself was written entirely or in parts by AI.

# Acknowledgements

We thank Alexie Leauthaud, Rachel Mandelbaum, and Jean Coupon for helping with some of the early design choices. Additionally, we thank Chris Blake for comparing the results of `dsigma` against his own independent pipeline. Finally, we are grateful to the DES, KiDS, HSC, and DECADE collaborations for making their data publicly available and providing excellent documentation.

# References
