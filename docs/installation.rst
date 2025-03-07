Installation
============

To install :code:`dsigma`, you can use pip or clone the source code from GitHub.

Automatic Installation
----------------------

The easiest way to install :code:`dsigma` is to use :code:`pip` to install the latest stable version from the Python Package Index (PyPI). This will automatically install the package and all its dependencies.

.. code-block:: none

    pip install dsigma

The following packages will be installed alongside :code:`dsigma` if not installed already.

* :code:`numpy`
* :code:`scipy`
* :code:`astropy`
* :code:`scikit-learn`
* :code:`astropy-healpix`
* :code:`tqdm`

Additionally, for calculating the lens magnification bias, the installation of the Cosmic Microwave Background (CMB) code :code:`camb` is recommended. However,  this is no hard requirement and :code:`dsigma` will run fine without it.

Manual Installation
-------------------

You can install :code:`dsigma` directly from the source code. This allows you to, for example, use versions of :code:`dsigma` not yet released on PyPI and to change compiler flags for the C code. The following code will install the most recent version of :code:`dsigma` on GitHub.

.. code-block:: none

    git clone https://github.com/johannesulf/dsigma.git
    cd dsigma
    pip install .
