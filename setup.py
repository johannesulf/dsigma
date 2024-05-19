import platform

from Cython.Build import cythonize
from setuptools import setup, Extension

extra_compile_args = ['-O3']

if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
    extra_compile_args += ['-march=native']

setup(
    ext_modules=cythonize([Extension(
        'dsigma.precompute_engine', ['dsigma/precompute_engine.pyx'],
        extra_compile_args=extra_compile_args)])
)
