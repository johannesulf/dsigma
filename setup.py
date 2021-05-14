from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.sdist import sdist
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


class sdist_with_cythonize(sdist):
    def run(self):
        cythonize(['dsigma/precompute_engine.pyx'])
        sdist.run(self)


ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension(
    'dsigma.precompute_engine', ['dsigma/precompute_engine' + ext])]

if USE_CYTHON:
    extensions = cythonize(extensions)


with open('README.md', 'r') as fstream:
    long_description = fstream.read()

setup(
    name='dsigma',
    version='0.5.0',
    description=('A Galaxy-Galaxy Lensing Pipeline'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='astronomy, weak-lensing',
    url='https://github.com/johannesulf/dsigma',
    author='Johannes Lange, Song Huang',
    author_email='jolange@ucsc.edu',
    packages=find_packages(),
    install_requires=['numpy', 'astropy', 'scipy', 'scikit-learn',
                      'healpy'],
    python_requires='>=3.4',
    ext_modules=extensions,
    cmdclass={'sdist': sdist_with_cythonize}
)
