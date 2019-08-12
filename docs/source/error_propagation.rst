=================
Error Propagation
=================

Error propagation is implemented as a child class of :py:class:`ExperimentalValue` called :py:class:`DerivedValue`. When working with QExPy, the result of all computations are stored as instances of this class.

The DerivedValue Object
=======================

.. autoclass:: qexpy.data.data.DerivedValue

Properties
----------

.. autoattribute:: qexpy.data.data.DerivedValue.value
.. autoattribute:: qexpy.data.data.DerivedValue.error
.. autoattribute:: qexpy.data.data.DerivedValue.relative_error
.. autoattribute:: qexpy.data.data.DerivedValue.error_method

Methods
-------

.. automethod:: qexpy.data.data.DerivedValue.reset_error_method
.. automethod:: qexpy.data.data.DerivedValue.recalculate
