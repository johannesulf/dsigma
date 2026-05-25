"""A Galaxy-Galaxy Lensing Python Package."""

from astropy.cosmology import Planck15

default_cosmology = Planck15

from . import helpers, jackknife, physics, precompute, stacking   # noqa: E402

__all__ = ['default_cosmology', 'helpers', 'jackknife', 'physics',
           'precompute', 'stacking']
__version__ = '1.2.0'
