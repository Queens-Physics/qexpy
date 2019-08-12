============
Introduction
============

QExPy (Queen’s Experimental Physics) is a Python 3 package designed to facilitate data analysis in undergraduate physics laboratories. The package contains a module to easily propagate errors in uncertainty calculations, and a module that provides an intuitive interface to plot and fit data. The package is designed to be efficient, correct, and to allow for a pedagogic introduction to error analysis. The package is extensively tested in the Jupyter Notebook environment to allow high quality reports to be generated directly from a browser.

Highlights:
 * Easily propagate uncertainties in calculations involving measured quantities
 * Compare different methods of error propagation (e.g. Quadrature errors, Monte Carlo errors)
 * Correctly include correlations between quantities when propagating uncertainties
 * Calculate derivatives of calculated values with respect to the measured quantities from which the value is derived
 * Flexible display formats for values and their uncertainties (e.g. number of significant figures, different ways of displaying units, scientific notation)
 * Smart unit tracking in calculations (in development)
 * Fit data to common functions (polynomials, gaussian distribution) or any custom functions specified by the user
 * Intuitive interface for data plotting built on matplotlib
