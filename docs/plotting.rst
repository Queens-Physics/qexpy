Plotting Module
===============

This module is where QExPy and Bokeh combine.  The creation of plots and
the tools used to interact with said plots are created using Bokeh.
The aim of the plotting module is to allow for the creation of plots and
fits without the involved methods normally required to create an 
interactive plot.

The Plot Object
---------------

Plots created with QExPy are stored in a variable like any other value.
This variable can then be operated on to add or change different aspects of
the plot, such as lines of fit, user-defined functions or simply the colour
of data point.

.. bokeh-plot::
   :source-position: above

   import qexpy.plotting as p

   # This produces two sets of data which should be fit to a line with a
   # slope of 3 and an intercept 2

   figure = p.Plot([1, 2, 3, 4, 5], [5, 7, 11, 14, 17], xerr=0.5, yerr=1)
   figure.show()
	
Using methods such as *.fit* or *.residuals* will create a best fit of
the data and display the residual output.  The *.fit* attribute also has 
arguments of what type of fit is required and, if the model is not built-in
to the module, an initial guess of the fitting parameters.

.. automethod:: qexpy.plotting.Plot.fit

Note, for the guess argument, the expected input is a list of numbers which
should be close to the true parameters.  If said values are not known, a
list of ones, of the correct length will suffice, although the fitting
algorithm may take longer to complete.  For example:

.. nbinput:: ipython3

   def model(x, pars):
   return pars[0] + pars[1]*x
		
   # As this model requires two parameters a guess should be:
   guess = [1, 1]
	
Using these methods, a plot with a best fit line and residuals can easily
be constructed.

.. nbinput:: ipython3

   import qexpy.error as e
   import qexpy.plotting as p

   x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y = e.Measurement([5, 7, 11, 14, 17], [1], name='Mass', units='g')

   figure = p.Plot(x, y)
   figure.fit('linear')
   figure.residuals()
   figure.show()

.. nboutput:: ipython3
	
.. bokeh-plot::
   :source-position: none
	
   import qexpy.plotting as p

   figure = p.Plot([1, 2, 3, 4, 5], [5, 7, 11, 14, 17], xerr=0.5, yerr=1)
   figure.fit('linear')
   figure.residuals()
   figure.show('file')

The included models for fitting include:

Linear: :math:`y=m x+b`

Gaussian: :math:`y=\frac{1}{\sqrt{2 \pi \sigma}}\exp{-\frac{(x-\mu)^2}{\sigma}}`

Polynomial: :math:`\sum_{i=0}^{N} a_i x^i` with parameters :math:`a_i`

Once fitted, the parameters of a fit can be returned with the
*.fit_parameters* method.

Parameters of a Fit
-------------------

In the case of any polynomial fit, included as a model by default, each
parameter is labelled in accordance with the power of the *x* variable.
Thus in the case of a linear fit, the intercept would be *pars[0]* and the
slope would be *pars[1]*.  This pattern hold for any degree of polynomial
fitted to the data.

For the Gaussian fit, *pars[0]* refers to the mean and *pars[1]* to the
standard deviation of the Gaussian curve.  Any models given by the user are
required to have two arguments.  The first being the independent variable
and the second as the parameters of the model.  
	
Once calculated, the parameters are stored in the Plot object, and can be
printed using the *.print_fit* method, which will print the parameters and,
in the case of parameters with defined names, the name in a pretty format.

.. nbinput:: ipython3

   x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y = e.Measurement([5, 7, 11, 14, 17], [1], name='Mass', units='g')

   figure = p.Plot(x, y)
   figure.fit('linear')
   figure.print_fit()

.. nbinput:: ipython3

   intercept = 3 +/- 1
   slope = 2 +/- 1

User-Defined Functions
----------------------

A user defined function can be plotted using the *.function* method as we
have previously done for curve fits and residual outputs.
To add a theoretical curve, or any other curve:

.. nbinput:: ipython3

   import qexpy.error as e
   import qexpy.plotting as p

   x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y = e.Measurement([5, 7, 11, 14, 17], [1], name='Mass', units='g')

   figure = p.Plot(x, y)
   figure.fit('linear')

   def theoretical(x):
       return 3 + 2*x

   figure.function(x, theoretical)
   figure.show()
    
.. automethod:: qexpy.plotting.Plot.function

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

   import qexpy.error as e
   import qexpy.plotting as p

   x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
   y = e.Measurement([5, 7, 11, 14, 17], [1], name='Applied Mass',
	units='g')

   figure = p.Plot(x, y)
   figure.show('file')

For this code, there is no output, as the plot will be saved in the working
directory and opened in a browser.  For example, if the above code is
located in *Diligent_Physics_Student/Documents/Python* then the HTML file
will also be in said */Python* folder.

.. todo:::

   Add Bokeh object as attribute, allow return and entry of object
   Adjust _plot_function so that lines are plotted along x-xerr to x+xerr
   Test for compatibility with on ReadTheDocs
