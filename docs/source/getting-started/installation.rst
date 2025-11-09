.. _getting_started.installation:

============
Installation
============

qexpy can be installed from `PyPI <https://pypi.org/project/qexpy>`_:

.. code-block:: shell

   pip install qexpy

Installing from source
----------------------

It is recommended to install qexpy source code and its dependencies using the
`uv <https://docs.astral.sh/uv/>`_ package manager. After cloning the project:

.. code-block:: shell

   git clone https://github.com/Queens-Physics/qexpy.git
   cd qexpy

The following command sets up the environment automatically:

.. code-block:: shell

   uv sync --all-extras 

For development purposes, you can install all dev dependencies using:

.. code-block:: shell

   uv sync --all-groups --all-extras 

Python version support
----------------------

The package requires a minimum Python version of 3.11
