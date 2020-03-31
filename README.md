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

It's recommanded to use this package in the Jupyter Notebook environment.

```python
import qexpy as q
```

## Contributing

With a local clone of this repository, if you wish to do development work, run the `make prep` in the project root directory, or run the following command explicitly:

```shell script
pip install -r requirements.txt
pip install -e .
```
 
This will install pytest which we use for testing, and pylint which we use to control code quality, as well as necessary packages for generating documentation.

Before submitting any change, you should run `make test` in the project root directory to make sure that your code matches all code style requirements and passes all unit tests.
 
 The following command checks your code against all code style requirements:

```shell script
pylint qexpy
```

Navigate to the tests directory, and execute the following command to run all unit tests:

```shell script
pytest -v --durations=0
```

Documentation for this package is located in the docs directory. Run `make docs` in the project root directory to build the full documentation. The html page will open after the build is complete.

Navigate to the docs directory, and run the following commands to build and see the full documentation page:

```shell script
make html
open docs/build/html/index.html
```
