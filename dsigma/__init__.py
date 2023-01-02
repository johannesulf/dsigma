"""Delta sigma pipeline for galaxy-galaxy lensing."""

from . import helpers
from . import jackknife
from . import physics
from . import precompute
from . import stacking
from . import surveys
import importlib.metadata


__all__ = ["helpers", "jackknife", "physics", "precompute", "stacking",
           "surveys"]
__version__ = importlib.metadata.version("dsigma")
