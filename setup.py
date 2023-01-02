from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([Extension(
        'dsigma.precompute_engine', ['dsigma/precompute_engine.pyx'],
        extra_compile_args=['-O3', '-march=native'])])
)
