Operations Module
=================

The operations module of this package contains the code which allows for the
propagation of errors with Measurement objects, created in the uncertainties
module. To improve the user experience, all of these methods are available
through the uncertainties module. Thus, this module will not be directly used
by most users. There are several techniques used in this module which are the
basis on which this package operates. Thus, said methods will be covered in this
section of the documentation.

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
