.. _user_guide.units:

==================
Working with Units
==================

You can specify the unit of a physical quantity using a string:

.. ipython:: python

   import qexpy as q
   a = q.Measurement(5, 0.5, unit="kg*m^2/s^2A^2")
   a

Internally, qexpy parses the input string and stores the unit in a dictiaonry:

.. ipython:: python

   print(dict(a.unit))
