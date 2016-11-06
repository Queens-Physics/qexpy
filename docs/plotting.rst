Plotting
========

This module is what will allow for both fitting and the creation of
figures. Similar to the process used in creating Measurement objects,
the creation and use of figures is based on the Plot object.
Furthermore, the actual figure itself can be rendered using both the
Bokeh and Matplotlib Python packages. While each engine has different
benifits, this document will focus on the use of the Plot object, rather
than the plotting software itself.

The Plot Object
---------------

Plots created with QExPy are stored in a variable like any other value.
This variable can then be operated on to add or change different aspects
of the plot, such as lines of fit, user-defined functions or simply the
colour of data point.

.. bokeh-plot::
   :source-position: above

   import qexpy.plotting as p

   # This produces two sets of data which should be fit to a line with a
   # slope of 3 and an intercept 2

   figure = p.Plot([1, 2, 3, 4, 5], [5, 7, 11, 14, 17],
					xerr=0.5, yerr=1)
   figure.show()
	
Using methods such as *.fit* or *.residuals* will create a best fit of
the data and display the residual output.  The *.fit* attribute also has
arguments of what type of fit is required and, if the model is not
built-in to the module, an initial guess of the fitting parameters.