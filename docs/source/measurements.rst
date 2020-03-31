======================
The Measurement Object
======================

To record values with an uncertainty, we use the :py:class:`.MeasuredValue` object. It is a child class of :py:class:`.ExperimentalValue`, so it inherits all attributes and methods from the :py:class:`.ExperimentalValue` class.

.. autoclass:: qexpy.data.data.MeasuredValue

Repeated Measurements
=====================

To record a value as the mean of a series of repeated measurements, use :py:class:`.RepeatedlyMeasuredValue`

.. autoclass:: qexpy.data.data.RepeatedlyMeasuredValue

Properties
----------

.. autoattribute:: qexpy.data.data.RepeatedlyMeasuredValue.raw_data
.. autoattribute:: qexpy.data.data.RepeatedlyMeasuredValue.mean
.. autoattribute:: qexpy.data.data.RepeatedlyMeasuredValue.error_weighted_mean
.. autoattribute:: qexpy.data.data.RepeatedlyMeasuredValue.std
.. autoattribute:: qexpy.data.data.RepeatedlyMeasuredValue.error_on_mean
.. autoattribute:: qexpy.data.data.RepeatedlyMeasuredValue.propagated_error

Methods
-------

.. automethod:: qexpy.data.data.RepeatedlyMeasuredValue.use_std_for_uncertainty
.. automethod:: qexpy.data.data.RepeatedlyMeasuredValue.use_error_on_mean_for_uncertainty
.. automethod:: qexpy.data.data.RepeatedlyMeasuredValue.use_error_weighted_mean_as_value
.. automethod:: qexpy.data.data.RepeatedlyMeasuredValue.use_propagated_error_for_uncertainty
.. automethod:: qexpy.data.data.RepeatedlyMeasuredValue.show_histogram

Correlated Measurements
=======================

Sometimes in experiments, two measured quantities can be correlated, and this correlation needs to be accounted for during error propagation. QExPy provides methods that allows users to specify the correlation between two measurements, and it will be taken into account automatically during computations.

.. autofunction:: qexpy.data.data.set_correlation
.. autofunction:: qexpy.data.data.get_correlation
.. autofunction:: qexpy.data.data.set_covariance
.. autofunction:: qexpy.data.data.get_covariance

There are also shortcuts to the above methods implemented in :py:class:`.ExperimentalValue`.

.. automethod:: qexpy.data.data.MeasuredValue.set_correlation
.. automethod:: qexpy.data.data.MeasuredValue.get_correlation
.. automethod:: qexpy.data.data.MeasuredValue.set_covariance
.. automethod:: qexpy.data.data.MeasuredValue.get_covariance
