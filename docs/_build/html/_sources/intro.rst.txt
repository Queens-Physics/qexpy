Introduction
============

QExPy (Queen's Experimental Physics) is a python 3 package designed to facilitate data analysis in undergraduate physics laboratories. The package contains a module to easily propagate errors in uncertainty calculations, and a module that provides an intuitive interface to plot and fit data. The package is designed to be efficient, correct, and to allow for a pedagogic introduction to error analysis. The package is extensively tested in the Jupyter Notebook environment to allow high quality reports to be generated directly from a browser. 

Highlights:
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

Examples
--------

Up to date Jupyter notebooks highlighting the features of QExPy can be found on `Github 
<https://github.com/Queens-Physics/qexpy/tree/master/examples/jupyter>`_. Some of the examples in this documentation may be out of date.

We can create "Measurement" objects to represent quantities with uncertainties, and propagate the error in those quantities.

.. nbinput:: ipython3
   :execution-count: 1
   
   #import the error propagation module
   import qexpy as q
   #declare 2 Measurements, x and y
   x = q.Measurement(10,1)
   y = q.Measurement(5,3)
   #define a quantitiy that depends on x and y:
   z = (x+y)/(x-y)
   #print z, with the correct error
   print(z)
   
.. nboutput:: ipython3

   3.0 +/- 0.6
   
   
The example below shows a case of plotting data and fitting them to a straight line:

.. nbinput:: ipython3
   :execution-count: 1

   import qexpy as q

   # This cell loads the module
	
.. nbinput:: ipython3
   :execution-count: 2

   # There are several ways to produce a Plot Object from a set of data.
   # Here, we pass the data directly to the plot object:
   
   fig1 = q.MakePlot(xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     ydata = [0.9, 1.4, 2.5, 4.2, 5.7, 6., 7.3, 7.1, 8.9, 10.8],
                     yerr = 0.5,
                     xname = 'length', xunits='m',
                     yname = 'force', yunits='N',
                     data_name = 'mydata')

 
	
.. nbinput:: ipython3
		      
   # We can now fit the data, and display a plot (optionally) showing the residuals

   fig1.fit("linear")
   fig1.add_residuals()
   fig1.show()
	 
.. nboutput:: ipython3

   -----------------Fit results-------------------
   Fit of  mydata  to  linear
   Fit parameters:
   mydata_linear_fit0_fitpars_intercept = -0.3 +/- 0.4,
   mydata_linear_fit0_fitpars_slope = 1.06 +/- 0.06

   Correlation matrix: 
   [[ 1.    -0.886]
   [-0.886  1.   ]]

   chi2/ndof = 0.71/7
   ---------------End fit results----------------
   
.. bokeh-plot::
   :source-position: none
   
   import qexpy as q
   fig1 = q.MakePlot(xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  ydata = [0.9, 1.4, 2.5, 4.2, 5.7, 6., 7.3, 7.1, 8.9, 10.8],
                  yerr = 0.5,
                  xname = 'length', xunits='m',
                  yname = 'force', yunits='N',
                  data_name = 'mydata')
   fig1.fit("linear")
   fig1.add_residuals()
   fig1.show()
