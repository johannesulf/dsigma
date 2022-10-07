import os
import re
from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.sdist import sdist
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext = 'pyx' if USE_CYTHON else 'c'

extensions = [Extension(
    'dsigma.precompute_engine', ['dsigma/precompute_engine.{}'.format(ext)],
    extra_compile_args=['-O3', '-march=native'])]

if USE_CYTHON:
    extensions = cythonize(extensions)


class sdist_with_cythonize(sdist):
    def run(self):
        cythonize(['dsigma/precompute_engine.pyx'])
        sdist.run(self)


with open('README.md', 'r') as fstream:
    long_description = fstream.read()

init = open(os.path.join((os.path.dirname(os.path.realpath(__file__))),
                         'dsigma', '__init__.py')).read()
version = re.search('^__version__ = "([A-Za-z0-9.-]+)"', init, re.M).group(1)

setup(
    name='dsigma',
    version=version,
    description=('A Galaxy-Galaxy Lensing Pipeline'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    keywords='astronomy, weak-lensing',
    url='https://github.com/johannesulf/dsigma',
    author='Johannes U. Lange, Song Huang',
    author_email='julange.astro@proton.me',
    packages=find_packages(),
    install_requires=['numpy', 'astropy', 'scipy', 'scikit-learn',
                      'astropy-healpix', 'tqdm'],
    python_requires='>=3.4',
    ext_modules=extensions,
    cmdclass={'sdist': sdist_with_cythonize}
)
