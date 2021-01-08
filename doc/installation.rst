Installation
============

Dependencies:
  - ``Python 3.6``, ``Python 3.7`` or ``Python 3.8``.
  - ``NumPy``, ``SciPy`` (automatically downloaded during installation).
  - ``Cython>=0.29.17``, ``Numpy`` (if building from source).

Environment
-----------

To install Sumu you will need Python installed. Although you can
install Python system wide it is in most cases preferable to have a
local installation in a virtual environment, e.g., using `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_.

`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ is the
minimal installation of Conda. After installing it and `creating and
activating an environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ for one of the supported Python versions you
can either install the latest release version or the development
version of Sumu.

Latest release version
----------------------

The easiest way to install the package is by running

::

    $ pip install sumu

which will install the latest release version from `PyPI
<https://pypi.org/project/sumu/>`_.

Development version
-------------------

To alternatively install from sources you should clone the repository
and run

::

    $ pip install .


in the repository root.

Finally, if you wish to work on the source code it is preferable to
install with

::

    $ pip install --verbose --no-build-isolation --editable .

as the ``editable`` flag allows you to modify the Python code without
reinstallation. Any changes to compiled code (i.e., C++ or Cython)
will still need to be recompiled by running the same command.
