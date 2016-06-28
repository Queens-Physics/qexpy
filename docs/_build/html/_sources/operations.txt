Formatting
==========

Naming
------

In addition to containing a mean and standard deviation, Measurement
objects can also have a string name and unit associated with it.
These can then be used both in printing values and in labelling any plots
created with these values.  By default, Measured objects are named
unnamed_var0, with a unique number assigned to each object.
The name and units of a Measured object can be declared either when the
object is created or altered after.

.. nbinput:: ipython3

   import qexpy.error as e
	
   x = e.Measurement(10, 1, name='Length', units='cm')
   # This value can be changed using the following method
	
   x.rename(name='Cable Length', units='m')
   # Note that units are only a marker and changing units does not change
   # any values with a Measurement
	
   print(x)
	
.. nboutput:: ipython3

   Cable Length = 10 +/- 1

Units
-----

Values which have more complicated units can also be entered using the
following syntax.  Consider a measurement of acceleration, with units of
m/s^2 or meters per second squared, this can be entered as a list of the
unit letters followed by the exponent of the unit, for every base unit,
such as meter order second:
	
.. nbinput:: ipython3

   import qexpy.error as e
	
   t = e.Measurement(3,0.25, name='Time', units='s')
   a = e.Measurement(10, 1, name='Acceleration', units=['m',1,'s',-1])

This also allows for the units of values produced by operations such as
multiplication to be generated automatically.  Consider the calculation of
the velocity of some object that accelerates at a for t seconds:

.. nbinput:: ipython3

   v = a*t
   print(v.units)
	
.. nboutput:: ipython3

	['m',1,'s','1']
	
This unit list, when used in a plot will appear as:

.. code-block:: python

   'm^1s^-1'

Print Styles
------------

The default format of printing a value with an uncertainty is:
   
.. nbinput:: ipython3
   
   import qexpy.error as e
   x = e.Measurement(10, 1)
   print(x)

.. nboutput:: ipython3

   10 +/- 1
	
However, there are three ways of outputting a Measurement object.
Furthermore, each method also allows for a specific number of significant
digits to be shown.

One method is called scientific and will output the number in scientific
notation with the error being shown as a value with only a single whole 
digit.  In order to change between any printing method, the following
function will change how the package prints a Measurement object:

.. nbinput:: ipython3

   import qexpy.error as e
	
   x = e.Measurement(122, 10)
   e.Measurement.print_style("Scientific")
   print(x)
	
.. nboutput:: ipython3

   (12 +/- 1)*10**1
	
The same process is used for a print style called Latex which, as the name
suggests, is formatted for use in Latex documents.  This may be useful in
the creation of labs by allowing variables to be copied and pasted
directly into a Latex document.

.. nbinput:: ipython3

   import qexpy.error as e
	
   x = e.Measurement(122, 10)
   e.Measurement.print_style("Latex")
   print(x)
	
.. nboutput:: ipython3

   (12 \pm 1)\e1
