![PyPI - Version](https://img.shields.io/pypi/v/qexpy)
![GitHub License](https://img.shields.io/github/license/Queens-Physics/qexpy)

# qexpy: data analysis with error propagation

qexpy is a Python package designed to facilitate experimental data analysis in
physics laboratories. It provides an intuitive interface to record measurements
with uncertainties and perform calculations with automatic error propagation. 
qexpy also offers modules for plotting and curve fitting, providing researchers
with a comprehensive toolkit for data analysis. For more details on how to use
qexpy, check out the [Official Documentation](https://qexpy.readthedocs.io/en/latest/).

## Installation

The library is available to download from [PyPI](https://pypi.org/project/qexpy)

```sh
pip install qexpy
```

Optionally, ``matplotlib`` is required to enable the ``plotting`` module.

```sh
pip install matplotlib
```

## Development

For development purposes, it is recommended to install qexpy source code and
all of its dependencies using the [uv](https://docs.astral.sh/uv/) package manager:

```sh
uv sync --all-groups --all-extras 
```

This automatically creates a virtual environment in your current directory,
prepares all dependencies, and installs ``qexpy`` in editable mode.

## License

[GNU General Public License v3.0](LICENSE)
