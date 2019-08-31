===========================
The MeasurementArray Object
===========================

Using QExPy, the user is able to record a series of measurements, and store them in an array. This feature is implemented in QExPy as a wrapper around :py:class:`numpy.ndarray`. The :py:class:`.ExperimentalValueArray` class, also given the alias :py:class:`.MeasurementArray` stores an array of values with uncertainties, and it also comes with methods for some basic data processing.

.. autoclass:: qexpy.data.datasets.ExperimentalValueArray

Properties
==========

.. autoattribute:: qexpy.data.datasets.ExperimentalValueArray.values
.. autoattribute:: qexpy.data.datasets.ExperimentalValueArray.errors
.. autoattribute:: qexpy.data.datasets.ExperimentalValueArray.name
.. autoattribute:: qexpy.data.datasets.ExperimentalValueArray.unit

Methods
=======

.. automethod:: qexpy.data.datasets.ExperimentalValueArray.mean
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.std
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.sum
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.error_on_mean
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.error_weighted_mean
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.propagated_error
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.append
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.delete
.. automethod:: qexpy.data.datasets.ExperimentalValueArray.insert
