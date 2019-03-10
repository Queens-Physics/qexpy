============================
The ExperimentalValue Object
============================

The core of QExPy is a data structure used to store quantities with a center value and an uncertainty. Each quantity is stored in an instance of :py:class:`.ExperimentalValue`, which has some basic attributes. This class is not to be instantiated directly. To learn about how to take measurements, refer to the class :py:class:`.MeasuredValue` for more information.

.. autoclass:: qexpy.data.data.ExperimentalValue

Properties
----------

.. autoattribute:: qexpy.data.data.ExperimentalValue.value
.. autoattribute:: qexpy.data.data.ExperimentalValue.error
.. autoattribute:: qexpy.data.data.ExperimentalValue.relative_error
.. autoattribute:: qexpy.data.data.ExperimentalValue.name
.. autoattribute:: qexpy.data.data.ExperimentalValue.unit

Methods
-------

.. automethod:: qexpy.data.data.ExperimentalValue.derivative
