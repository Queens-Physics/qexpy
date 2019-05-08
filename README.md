[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Queens-Physics/qexpy/master)

# QExPy

## Introduction

QExPy (Queen’s Experimental Physics) is a python 3 package designed to facilitate data analysis in undergraduate physics laboratories. The package contains a module to easily propagate errors in uncertainty calculations, and a module that provides an intuitive interface to plot and fit data. The package is designed to be efficient, correct, and to allow for a pedagogic introduction to error analysis. The package is extensively tested in the Jupyter Notebook environment to allow high quality reports to be generated directly from a browser.

**Highlights**:  
  * Easily propagate uncertainties in measured quantities  
  * Compare different uncertainty calculations (e.g. Min-Max, quadrature errors, Monte Carlo errors)  
  * Correctly include correlations between quantities when propagating uncertainties (e.g. the uncertainty on x-x should always be 0!)  
  * Calculate exact numerical values of derivatives  
  * Choose display format (standard, Latex, scientific notation)  
  * Control the number of significant figures  
  * Handle ensembles of measurements (e.g. combine multiple measurements, with uncertainties, of a single quantity)  
  * Produce interactive plots of data in the browser  
  * Fit data to common functions (polynomials, gaussians) or provide custom functions  
  * Examine residual plots after fits  
  * Track units in calculations (still in development)  
  * Plot confidence bands from the errors in fitted parameters (still in development)  
  * Integrates with Jupyter notebooks, numpy, bokeh  

## Examples
Up to date examples are maintained in the examples directory of the repository. These are likely the best way to get acquainted with the package.

## More information
Refer to the example notebooks in the examples/jupyter directory to learn how to use the package, and browse through the official documentstion.

Read the documentation at http://qexpy.readthedocs.io/en/latest/intro.html
