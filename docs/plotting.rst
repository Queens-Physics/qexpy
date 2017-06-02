Plotting
========

This module is what will allow for both fitting and the creation of
figures. Similar to the process used in creating Measurement objects,
the creation and use of figures is based on the Plot object.
Furthermore, the actual figure itself can be rendered using both the
Bokeh and MatPlotLib Python packages. While each engine has different
benifits, this document will focus on the use of the Plot object, rather
than the plotting software itself.

The Plot Object
---------------

Plots created with QExPy are stored in a variable like any other value.
This variable can then be operated on to add or change different aspects
of the plot, such as lines of fit, user-defined functions or simply the
colour of data point. To choose which plot engine is used by the
package, the *.plot_engine* setting can be set to either *'mpl'*, or
*'bokeh'*. The rendering performed by MatPlotLib is faster, though less
interactive than Bokeh, so it will be used for this demonstration.

.. nbinput:: ipython3

   import qexpy as q
   q.plot_engine = 'mpl'

   # This produces two sets of data which should be fit to a line with a
   # slope of 3 and an intercept 2

   figure = q.MakePlot([1, 2, 3, 4, 5], [5, 7, 11, 14, 17],
					xerr=0.5, yerr=1)
   figure.show()

For an example of the output of this code, please see the
`GitHub example notebooks`_.

.. _GitHub example notebooks: https://github.com/Queens-Physics/qexpy/tree/master/examples/jupyter
	
Using methods such as *.fit* or *.residuals* will create a best fit of
the data and display the residual output.  The *.fit* attribute also has
arguments of what type of fit is required and, if the model is not
built-in to the module, an initial guess of the fitting parameters.

.. automethod:: qexpy.plotting.Plot.fit
   :noindex:

Note, for the *parguess* argument, the expected input is a list of
numbers which should be close to the true parameters.  If said values
are not known, a list of ones, of the correct length will suffice,
although the fitting algorithm may take longer to complete.
For example:

.. nbinput:: ipython3

   def model(x, pars):
   return pars[0] + pars[1]*x
		
   # As this model requires two parameters a guess should be:
   guess = [1, 1]
	
Using these methods, a plot with a best fit line and residuals can
easily be constructed.

.. nbinput:: ipython3

   import qexpy as q
   q.plot_engine = 'mpl'

   x = q.MeasurementArray([1, 2, 3, 4, 5], [0.5],
							name='Length', units='cm')
   y = q.MeasurementArray([5, 7, 11, 14, 17], [1],
   \						name='Mass', units='g')

   figure = q.MakePlot(x, y)
   figure.fit('linear')
   figure.residuals()
   figure.show()

The included models for fitting include:

Linear: :math:`y=m x+b`

Gaussian: :math:`y=\frac{1}{\sqrt{2 \pi \sigma}}\exp{-\frac{(x-\mu)^2}{\sigma}}`

Polynomial: :math:`y=\sum_{i=0}^{N} a_i x^i` with parameters :math:`a_i`

Once fitted, the parameters of a fit can be stored by listing variables
as equal to the *.fit* method.

.. nbinput:: ipython3

   import qexpy as q
   q.plot_engine = 'mpl'

   x = q.MeasurementArray([1, 2, 3, 4, 5], [0.5],
							name='Length', units='cm')
   y = q.MeasurementArray([5, 7, 11, 14, 17], [1],
							name='Mass', units='g')

   figure = q.MakePlot(x, y)
   m, b = figure.fit('linear')
   figure.residuals()
   figure.show()

Parameters of a Fit
-------------------

A common non-linear fit used in physics is the normal, or Gaussian fit.
This function is build into the QexPy package and can be used as simply
as the linear fit function.

.. nbinput:: ipython3

   import qexpy as q
   q.plot_engine = 'mpl'

   x = q.MeasurementArray([1, 2, 3, 4, 5], [0.5],
							name='Length', units='cm')
   y = q.MeasurementArray([ 0.325,  0.882 ,  0.882 ,  0.325,  0.0439],
							[1], name='Mass', units='g')

   figure = q.MakePlot(x, y)
   m, b = figure.fit('Gausss')
   figure.residuals()
   figure.show()

User-Defined Functions
----------------------

A user defined function can be plotted using the *.function* method as
we have previously done for curve fits and residual outputs.
To add a theoretical curve, or any other curve:

.. nbinput:: ipython3

   import qexpy as q
   q.plot_engine = 'mpl'

   x = q.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y = q.Measurement([5, 7, 11, 14, 17], [1], name='Mass', units='g')

   figure = p.MakePlot(x, y)
   figure.fit('linear')

   def theoretical(x):
       return 3 + 2*x

   figure.function(x, theoretical)
   figure.show()
    
The final method relevant to Plot objects is the show method.
This, by default will output the Bokeh plot in a terminal, or output of a
Jupyter notebook, if that is where the code is executed.
This method does have an optional argument that determines where the plot
is shown, with options of 'inline' and 'file'.  The 'inline' option is
selected by default and refers to output in the console line itself,
while 'file' creates an HTML file that should open in your default
browser and save to whatever location your Python code file is currently
in.

.. nbinput:: ipython3

   import qexpy as q
   q.plot_engine = 'mpl'

   x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y = e.Measurement([5, 7, 11, 14, 17], [1], name='Applied Mass',
	units='g')

   figure = p.Plot(x, y)
   figure.show('file')

For this code, there is no output, as the plot will be saved in the
working directory and opened in a browser.  For example, if the above
code is located in *Diligent_Physics_Student/Documents/Python* then the
HTML file will also be in said */Python* folder.

Plotting Multiple Datasets
--------------------------

In many cases, multiple sets of data must be shown on a single plot,
possibly with multiple residuals. In this case, another Dataset object
must be created and 

.. autoclass:: qexpy.plotting.Plot
   :noindex:

This method is used by creating two separate plot objects and acting upon
each as you would with any other plot. When showing the plot, instead of
using the *.show* method, *.show_on(figure2)* is used, where *figure2* is
whatever you wish to add to the final plot.

.. nbinput:: ipython3

   import qexpy.error as e
   import qexpy.plotting as p

   x1 = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y1 = e.Measurement([5, 7, 11, 14, 17], [1], name='Applied Mass',
	units='g')

   figure1 = p.Plot(x1, y1)
   figure1.fit('linear')
   figure1.residual()

   x2 = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y2 = e.Measurement([4, 8, 13, 12, 19], [1], name='Applied Mass',
	units='g')
   
   figure2 = p.Plot(x2, y2)
   figure2.fit('linear')
   figure2.residual()

   figure1.show_on(figure2)

.. todo:::

   Adjust _plot_function so that lines are plotted along x-xerr to x+xerr
   Test for compatibility with on ReadTheDocs