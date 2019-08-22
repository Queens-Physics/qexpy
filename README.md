[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Queens-Physics/qexpy/master)

# QExPy

## Introduction

QExPy (Queen’s Experimental Physics) is a python 3 package designed to facilitate data analysis in undergraduate physics laboratories. The package contains a module to easily propagate errors in uncertainty calculations, and a module that provides an intuitive interface to plot and fit data. The package is designed to be efficient, correct, and to allow for a pedagogic introduction to error analysis. The package is extensively tested in the Jupyter Notebook environment to allow high quality reports to be generated directly from a browser.

## Getting Started

To install the package, type the following command in your terminal or [Anaconda](https://www.anaconda.com/distribution/#download-section) shell.

```sh
$ pip install qexpy
```

## Usage

```python
import qexpy as q
```

## Contributing

With a local clone of this repository, if you wish to do development work, run the following command in the project root directory.

```sh
$ pip install .[dev]
```

This will install pytest which we use for testing, and pylint which we use to control code quality. If you wish to participate in documenting the package, use the following command.

```sh
$ pip install .[dev,doc]
```

Before submitting any change, you should:

- Run `pytest -v --durations=0` in the test directory to execute all unit tests.
- Run `pylint qexpy` in the project root directory to make sure that your code matches the code stype requirements.
