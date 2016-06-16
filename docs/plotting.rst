Plotting Module
===============

This module is where QExPy and Bokeh combine. The creation of plots and the
tools used to interact with said plots are created using Bokeh. The aim of
the plotting module is to allow for the creation of plots and fits without
the invloved methods required to create Bokeh plots.

The Plot Object
---------------

Unlike other modules which allow plotting, those created with QExPy are stored
in a variable like any other value. This variable can then be operated on to
add or change different aspects of the plot, such as lines of fit, user-defined
functions or simply the color of data point.

..  bokeh-plot::
	:source-position: above
	
	from inspect import getsourcefile
	import os.path as path, sys
	current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
	sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

	import qexpy.plotting as p
	import qexpy.error as e
	sys.path.pop(0)

	x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
	y = e.Measurement([5, 7, 11, 14, 17], [1], name='Appplied Mass', units='g')
	# This produces two sets of data which should be fit to a line with a slope
	# of 3 and an intercept 2

	figure = p.Plot(x, y)
	figure.show()
	
Using class methods such as *.fit* or *.residuals* will create a best fit of
the data and display the residual output. The *.fit* attribute also has 
arguments of what type of fit is required and, if the model is not previously
defined, an inital guess of the fitting parameters.

.. automethod:: qexpy.plotting.Plot.fit

Note, for the guess argument, the expected input is a list of numbers which
should be close to the true parameters. If said values are not known, a list
of ones, of the correct length will suffice, although the fitting algorithm
may take longer to complete. For example:

..  nbinput:: ipython3

	def model(x, pars):
		return pars[0] + pars[1]*x
		
	# As this model requires two parameters a guess should be:
	
	guess = [1, 1]
	
Using these methods, a plot with a best fit line and residuals can easily be
constructed.

.. nbinput:: ipython3

	import qexpy.error as e
	import qexpy.plotting as p

	x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
	y = e.Measurement([5, 7, 11, 14, 17], [1], name='Appplied Mass', units='g')

	figure = p.Plot(x, y)
	figure.fit('linear')
	figure.residuals()
	figure.show()

..  nboutput:: ipython3
	
..  bokeh-plot::
	:source-position: none
	
	from inspect import getsourcefile
	import os.path as path, sys
	current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
	sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

	import qexpy.plotting as p
	import qexpy.error as e
	sys.path.pop(0)

	x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
	y = e.Measurement([5, 7, 11, 14, 17], [1], name='Appplied Mass', units='g')

	figure = p.Plot(x, y)
	figure.fit('linear')
	figure.residuals()
	figure.show('file')

	
The included models for fitting include:

Linear: :math:`y=mx+b`

Gaussian: :math:`y=\frac{1}{\sqrt{2 \pi \sigma}}\exp{-\frac{(x-\mu)^2}{\sigma}}`

Polynomial: :math:`\sum_{i=0}^{N} a_i x^i` with parameters :math:`a_i`
