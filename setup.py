from setuptools import setup, find_packages

with open('README.md', 'r') as fstream:
    long_description = fstream.read()

setup(
    name='dsigma',
    version="0.3.0rc",
    description=('A Galaxy-Galaxy Lensing Pipeline'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='astronomy, weak-lensing',
    url='https://github.com/dr-guangtou/dsigma',
    author='Johannes Lange, Song Huang',
    author_email='jolange@ucsc.edu',
    packages=find_packages(),
    install_requires=['numpy', 'astropy', 'scipy', 'scikit-learn',
                      'matplotlib'],
    python_requires='>=3.4',
)
