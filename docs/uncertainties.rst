Uncertainties Module
====================

This module is the core of the QExPy package, this is where the Measurement
class of objects is created and the methods which allow for derivatives to be
calculated and the names and units of values to be stored and propagated.

Creating Measurement Objects
----------------------------

The method that will be used most commonly is the Measured class. This object
can store the mean, standard deviation, original data, name, units, and other
attributes which can be used by other elements of this package.

.. autoclass:: QExPy.uncertainties.Measured

The arguments, or \*args, of this class can be in several forms:

A mean and standard deviation can be entred directly.

..  nbinput:: ipython3
   
	import QExPy.uncertainties as u
	x = u.Measured(10, 1)
	# This would create an object with a mean of 10 and a standard
	# deviation of 1.

A list or numpy array of values can be provided, from which the mean and
standard deviation of the values can be taken.

..  nbinput:: ipython3
   
	x = u.Measured([9, 10, 11])
	# This would also produce an object with a mean of 10 and a standard
	# deviation of 1.

If the list of values has errors associated with each measurement, the
data can be entered either as pairs of a value and error, or as two lists
of data and error respectivly.

..  nbinput:: ipython3
   
	x = u.Measured([10, 1], [9, 0.5], [11, 0.25])
	y = u.Measured([10, 9, 11], [1, 0.5, 0.25])
	# The mean and standard deviation of x and y are the same

The optional arguments *name* and *units* can be used to include strings
for both of these parameters as shown below:

..  nbinput:: ipython3

    x = u.Measured(10, 1, name='Length', units='cm')

Working with Measurement Objects
--------------------------------

Once created, these objects can be operated on just as any other value:

..  nbinput:: ipython3
   
	import QExPy.uncertainties as u

	x = u.Measured(10, 1)
	y = u.Measured(3, 0.1)
	z = x-y
	f = u.sin(z)

	print(z)

..  nboutput:: ipython3

	10 +/- 1

Elementary functions such as the trig functions, inverse trig functions,
natural logarithm and exponential function can also be used:

..  nbinput:: ipython3

	f = u.sin(z)
	print(f)
	
..  nboutput:: ipython3

	0.7 +/- 0.8

Furthermore, the use of Measured objects in equations also allows for the
calculation of the derivative of these expressions with respect to any of the
Measured objects used.

..  nbinput:: ipython3

	d1 = f.derivative(x)

	# This can be compared to the analytic expression of the derivative
	d2 = m.cos(10+3)*3
	print(d1 == d2)

..  nboutput:: ipython3

	True

This derivative method is what is used to propagate error by the error
propagation formula.

..  math::
    
    For\ some\ F(x,y):
    \sigma_F^2 = (\frac{\partial F}{\partial x} \sigma_x)^2 \
	+ (\frac{\partial F}{\partial y} \sigma_y)^2
    
This formula is the default method of error propagation and will be accurate in
most cases.

Formatting
----------

In addition to containing a mean and standard deviation, Measurement objects
can also have a string name and unit associated with it. These can then be
used both in printing values and in labelling any plots created with these
values. By default, Measured objects are named unnamed_var0, with a unique
number assigned to each object. The name and units of a Measured object can
be declared either when the object is created or altered after.

..  nbinput:: ipython3

	import QExPy.uncertainties as u
	
	x = u.Measured(10, 1, name='Length', units='cm')
	# This value can be changed using the following method
	
	x.rename(name='Cable Length', units='m')
	# Note that units are only a marker and changing units does not change
	# any values with a Measurement
	
	print(x)
	
..  nboutput:: ipython3

	Cable Length = 10 +/- 1
	
Values which have more complicated units can also be entered using the
following syntax
	

As shown above, the default method of printing a value with an uncertainty is:
   
..  nbinput:: ipython3
   
	import QExPy.uncertainties as u
	x = u.Measured(10, 1)
	print(x)

..  nbinput:: ipython3

	10 +/- 1
	
However, there are three ways of outputting a Measurement object. Furthermore,
each method also allows for a specific number of significant digits to be 
shown.

One method is called scientific and will output the number in scientific
notation with the error being shown as a value with only a single whole digit.
In order to change between any printing method, the following function will
change how the package prints a Measurement object:

..  nbinput:: ipython3

	import QExPy.uncertainties as u
	
	x = u.Measured(122, 10)
	u.Measurement.print_style("Scientific")
	print(x)
	
..  nboutput:: ipython3

	(12 +/- 1)*10**1
	
The same process is used for a print style called Latex which, as the name
suggests, is formatted for use in Latex documents. This may be useful in the
creation of labs by allowing variables to be copied and pasted directly into a
Latex document.

..  nbinput:: ipython3

	import QExPy.uncertainties as u
	
	x = u.Measured(122, 10)
	u.Measurement.print_style("Latex")
	print(x)
	
..  nboutput:: ipython3

	(12 \pm 1)\e1
	
Methods of Propagating Error
----------------------------

While the default method of propagating error is the derivative formula, there 
are a number of other methods by which error can be calculated. In addition to
the derivative method, this package is also capible of calculating error by the
Monte Carlo and min-max methods. While this documentation will not go into
detail about how these methods work, the output of each method is available
by default, and a specific method can be chosen as shown below.

.. nbinput:: ipython3

	import QExPy.uncertainties as u
	
	x = u.Measured(13,2)
	y = u.Measured(2,0.23)
	z = x**2 - x/y
	
	print(z)
	print(z.MC())
	print(z.MinMax())
	
..  nboutput:: ipython3

	(0 \pm 2)*10**4
	[145.18464217708808, 27094.377985685125]
	[162.5, 27569.397460908742]

While the Monte Carlo and min-max output of the default method are not as
elegent as the derivative method, it does provide an easy avenue to check
the error against another method to ensure accuracy.

Furthermore, the output can be limited to a single method if desired. In this
case, the output seen in the *print(x)* line would be from whatever method is
chosen.

.. nbinput:: ipython3
	
	x = u.Measured(10,2)
	y = u.Measured(5,1)

	u.Measurement.set_method("Derivative")
	# This option will limit the error calculation to using the derivative
	# formula
	
	z = x-y
	z.rename('Derivative Method')
	
	u.Measurement.set_method("Monte Carlo")
	# This option will limit the error calculation to using the derivative
	# formula
	
	z = x-y
	z.rename('Monte Carlo')
	
	u.Measurement.set_method("Min-Max")
	# This option will limit the error calculation to using the derivative
	# formula
	
	z = x-y
	z.rename("Min-Max")



	
	