Installation
============

To install ``dsigma``, you can use pip or clone the source code from GitHub.

Automatic Installation
----------------------

The easiest way to install ``dsigma`` is to use ``pip`` to install the latest stable version from the Python Package Index (PyPI). This will install the package and all its dependencies.

.. code-block:: none

    pip install dsigma

The following packages will be installed alongside ``dsigma`` if not installed already.

* ``numpy``
* ``scipy``
* ``astropy``
* ``scikit-learn``
* ``astropy-healpix``
* ``tqdm``

To process DECADE, DES, HSC, or KiDS data using the provided scripts, you'll also need ``h5py``. Additionally, for calculating the lens magnification bias, the installation of the Cosmic Microwave Background (CMB) code ``camb`` is recommended. However, this is not a hard requirement and ``dsigma`` will run fine without it.

Manual Installation
-------------------

You can install ``dsigma`` directly from the source code. This allows you to, for example, use versions of ``dsigma`` not yet released on PyPI and to change compiler flags for the C code. The following code will install the most recent version of ``dsigma`` on GitHub.

.. code-block:: none

    git clone https://github.com/johannesulf/dsigma.git
    cd dsigma
    pip install -e .
