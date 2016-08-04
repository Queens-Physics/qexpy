Introduction
============

QExPy (Queen's Experimental Physics) is a python 3 package designed to facilitate data analysis in undergraduate physics laboratories. The package contains a module to easily propagate errors in uncertainty calculations, as we well a module that provides an intuitive interface to plot and fit data in a visually esthetic manner (using bokeh for the plots and scipy.optimize for the fitting). The package is designed to be efficient, correct, and to allow for a pedagogic introduction to error analysis. The package is extensively tested in the Jupyter Notebook environment to allow students to produce high quality reports directly from their browser. 

Highlights:
 - Easily propagate uncertainties in measured quantities
 - Compare different uncertainty calculations (e.g. Min-Max, Derivatives with quadrature errors, Monte Carlo error propagation)
 - Correctly include correlations between quantities
 - Track units in calculations
 - Handle ensembles of measurements (e.g. combining multiple measurements with errors of a single quantity)
 - Produce interactive plots of data in the browser
 - Fit data to common functions (polynomials, gaussians) or provide custom functions
 - Examine residual plots after fits
 - Plot confidence bands from the errors in fitted parameters



.. nbinput:: ipython3
   :execution-count: 1

   import qexpy.error as e
   import qexpy.plotting as p

   # This cell will load the package
	
.. nbinput:: ipython3
   :execution-count: 2

   # We can now enter the data gathered in the lab itself

   dl = e.Measurement([185e-6, 250e-6, 305e-6, 378e-6, 460e-6, 515e-6,
	573e-6, 659e-6, 733e-6, 799e-6, 1199e-6, 860e-6, 933e-6, 993e-6,
	1060e-6, 1125e-6], [5e-6], name='Lengthening', units='m')
   # This data is the amount that the cable stretched for each applied mass
   # As the error on each of these measurements is the same, we will use a 
   # single value of error instead of another list containing the error for
   # each point.

   m50 = 0.05008
   m1 = 0.10010
   m2x = 0.20019
   m2i = 0.20025
   m5 = 0.50054
   m5d = 0.50087

   mass = e.Measurement([0, m1, m2x, m2x+m1, m2x+m2i, m5, m1+m5, m2x+m5,
	m2x+m5+m1, m5+m2x+m2i,m5+m5d+m2x+m2i+m1, m5+m5d, m5+m5d+m1,
	m5+m5d+m2x, m5+m5d+m2x+m1,m5+m5d+m2x+m2i], [0.04],
	name='Suspended Mass', units='m')

   ''' This list is the combination of weights that were used in each
   trial.  As the error on each of these measurements is the same, we will
   use a single value of error instead of another list containing the error
   for each point.
   '''
	
.. nbinput:: ipython3
		      
   # Now that we have the data stored, we can plot the data, along with a
   # line of best fit

   plot = p.Plot(dl, mass) # This creates the plot and stores it as plot
   plot.fit('linear') # We can find a linear fit of the data
   plot.residuals() # This tells the plot that we also want a residual plot
   plot.show() # Now the plot can be shown
	 
.. bokeh-plot::
   :source-position: none

   import qexpy.plotting as p

   dl = [185e-6, 250e-6, 305e-6, 378e-6, 460e-6, 515e-6, 573e-6,
					 659e-6, 733e-6, 799e-6, 1199e-6, 860e-6, 933e-6,
					 993e-6, 1060e-6, 1125e-6]

   m50 = 0.05008
   m1 = 0.10010
   m2x = 0.20019
   m2i = 0.20025
   m5 = 0.50054
   m5d = 0.50087

   mass = [0, m1, m2x, m2x+m1, m2x+m2i, m5, m1+m5, m2x+m5,
					m2x+m5+m1, m5+m2x+m2i, m5+m5d+m2x+m2i+m1, m5+m5d,
					m5+m5d+m1, m5+m5d+m2x, m5+m5d+m2x+m1,m5+m5d+m2x+m2i]

   plot = p.Plot(dl, mass, xerr=5e-6, yerr=0.04) # This creates the plot
   plot.fit('linear') # We can find a linear fit of the data
   plot.residuals() # This tells the plot that we also want a residual plot
   plot.show() # Now the plot can be shown

