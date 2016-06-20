Error Propagation
=================

The propagation of measurements, including the propagation of errors is at
the heart of this package. This section will describe how Measurement
objects are created and used in calculations. Furthermore, features such as
the calculation of the exact numerical derivative of expressions will be
outlined. While some aspects of this documentation will not neccesarily be
required to work with the package itself, many of the methods used in the
underlying code can be useful to understand.

Creating Measurement Objects
----------------------------

The method that will be used most commonly is the Measured class. This object
can store the mean, standard deviation, original data, name, units, and other
attributes which can be used by other elements of this package.

.. autoclass:: qexpy.error.Measurement

The arguments, or \*args, of this class can be entered in several forms:

A mean and standard deviation can be entred directly.

.. nbinput:: ipython3
   
   import qexpy.error as e
   x = e.Measurement(10, 1)
   # This would create an object with a mean of 10 and a standard
   # deviation of 1.

A list or numpy array of values can be provided, from which the mean and
standard deviation of the values is calculated. These values can be outputted
by calling for the mean and std attributes of whatever object is created.

.. nbinput:: ipython3
   
   x = e.Measurement([9, 10, 11])
   # This would also produce an object with a mean of 10 and a standard
   # deviation of 1. This can be shown by calling for x.mean and x.std:
	
   print(x.mean, x.std)
	
.. nboutput:: ipython3

   10, 1

If several measurements, each with an associated error needs to be entered,
the data can be entered either as pairs of a value and error, or as two lists
of data and error respectivly.

For example, given measurements 10 +/- 1, 9 +/- 0.5 and 11 +/- 0.25, the
data can be entered as either

.. nbinput:: ipython3
   
   x = e.Measurement([10, 1], [9, 0.5], [11, 0.25])
   y = e.Measurement([10, 9, 11], [1, 0.5, 0.25])
   # The mean and standard deviation of x and y are the same
   
If the error associated with each measured value is the same, a single
value can be entered into the second list in the *y* example shown above.
This is done simply for efficientcy and is treated as a list of repeated
values by the module.

.. nbinput:: ipython3

   x = e.Measurement([9, 10, 11], [1])
   # This is equivilent to:
   y = e.Measurement([9, 10, 11], [1, 1, 1])

In all cases, the optional arguments *name* and *units* can be used to
include strings for both of these parameters as shown below:

.. nbinput:: ipython3

   x = e.Measurement(10, 1, name='Length', units='cm')

Working with Measurement Objects
--------------------------------

Once created, these objects can be operated on just as any other value:

.. nbinput:: ipython3
   
   import qexpy.error as e

   x = e.Measurement(10, 1)
   y = e.Measurement(3, 0.1)
   z = x-y
   f = e.sin(z)

   print(z)

.. nboutput:: ipython3

   10 +/- 1

Elementary functions such as the trig functions, inverse trig functions,
natural logarithm and exponential function can also be used:

.. nbinput:: ipython3

   f = e.sin(z)
   print(f)
	
.. nboutput:: ipython3

   0.7 +/- 0.8

Furthermore, the use of Measured objects in equations also allows for the
calculation of the derivative of these expressions with respect to any of the
Measured objects used.

.. nbinput:: ipython3

   d1 = f.derivative(x)

   # This can be compared to the analytic expression of the derivative
   d2 = m.cos(10+3)*3
   print(d1 == d2)

.. nboutput:: ipython3

   True

This derivative method is what is used to propagate error by the error
propagation formula.

.. math::
    
   For\ some\ F(x,y):
   \sigma_F^2 = (\frac{\partial F}{\partial x} \sigma_x)^2 \
   + (\frac{\partial F}{\partial y} \sigma_y)^2
    
This formula is the default method of error propagation and will be accurate in
most cases.

Methods of Propagating Error
----------------------------

While the default method of propagating error is the derivative formula, there 
are a number of other methods by which error can be calculated. In addition to
the derivative method, this package is also capible of calculating error by the
Monte Carlo and min-max methods. While this documentation will not go into
detail about how these methods work, the output of each method is available
by default, and a specific method can be chosen as shown below.

.. nbinput:: ipython3

   import qexpy.error as e
	
   x = e.Measurement(13,2)
   y = e.Measurement(2,0.23)
   z = x**2 - x/y
	
   print([z.mean, z.std])
   print(z.MC)
   print(z.MinMax)
	
.. nboutput:: ipython3

   [162.5, 51.00547770828149]
   [162.88454043577516, 51.509516186100562]
   [162.5, 53.24850160192128]

While the Monte Carlo and min-max output of the default method are not as
elegent as the derivative method, it does provide an easy avenue to check
the error against another method to ensure accuracy.

Furthermore, the output can be limited to a single method if desired. In this
case, the output seen in the *print(x)* line would be from whatever method is
chosen.

.. nbinput:: ipython3
	
   x = e.Measurement(10,2)
   y = e.Measurement(5,1)

   e.Measurement.set_method("Derivative")
   # This option will limit the error calculation to using the derivative
   # formula

   z = x-y
   z.rename('Derivative Method')

   e.Measurement.set_method("Monte Carlo")
   # This option will limit the error calculation to using the derivative
   # formula

   z = x-y
   z.rename('Monte Carlo')

   e.Measurement.set_method("Min-Max")
   # This option will limit the error calculation to using the derivative
   # formula

   z = x-y
   z.rename("Min-Max")

Correlation
-----------

For many experiments, parameters may be correlated or may be expected to be
correlated. Thus, there exists methods to define and, in the case that th
arrays of data used to create two Measurements are equal in length, return the
covariance or correlation of some parameters. There are two methods which can
be used to set the correlation of two variables, or return the covariance of
two variables.

.. automethod:: qexpy.error.Measurement.set_correlation

.. automethod:: qexpy.error.Measurement.return_covariance

Furthermore, the covariance and correlation of the fitted parameters found
by the *.fit* method in QExPy.plotting 

.. todo::

   Build public method for finding name and ID of variable
   Min Max propagation should be altered to represent true min and max with
   a generalized function.

Derivatives
-----------

The method by which numerical solutions to the derivative of expressions are
evaluated is called automatic differentiation. This method relies on the chain
rule and the fact that the derivative of any expression can be reduced to some
combonation of elementary functions and operations. Consider the following
function.

.. math::

   f(x,y) = \cos(xy)
   \implies \partial_x f(x,y) = \cos(xy)y
   Let\ xy = z
   \partial_x f(x,y) \= \cos(z) \frac{\partial(z)}{\partial(x)} \\
   \= \cos(xy)y
   
What this example illustrates is how, by considering an expression as a series
of elementary operations and functions, the exact numerical derivative can be
calculated. All that is required is to be able to store the derivative of each
of these elementary operations with respect to whatever variables are involved.

.. todo::

   Outline operation wrapper
