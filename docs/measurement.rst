The Measurement Object
======================

The propagation of measurements, including the propagation of errors is at
the heart of this package. This section will describe how Measurement
objects are created as well as all the methods available to operate on Measurements.

.. autoclass:: qexpy.error.Measurement

Properties
----------

Measurements have a few properties that you can get and set. Each property has an example of how to get and set it.

.. autoattribute:: qexpy.error.Measurement.error_on_mean
.. autoattribute:: qexpy.error.Measurement.name
.. autoattribute:: qexpy.error.Measurement.relative_error
.. autoattribute:: qexpy.error.Measurement.std
.. autoattribute:: qexpy.error.Measurement.units

Functions
---------

Measurements also have a number of functions that can be used to get, set or display information about the Measurement object.

.. automethod:: qexpy.error.Measurement.get_covariance
.. automethod:: qexpy.error.Measurement.get_data_array
.. automethod:: qexpy.error.Measurement.get_derivative
.. automethod:: qexpy.error.Measurement.get_units_str
.. automethod:: qexpy.error.Measurement.print_deriv_error
.. automethod:: qexpy.error.Measurement.print_mc_error
.. automethod:: qexpy.error.Measurement.print_min_max_error
.. automethod:: qexpy.error.Measurement.rename
.. automethod:: qexpy.error.Measurement.set_correlation
.. automethod:: qexpy.error.Measurement.set_covariance
.. automethod:: qexpy.error.Measurement.show_error_contribution
.. automethod:: qexpy.error.Measurement.show_histogram
.. automethod:: qexpy.error.Measurement.show_MC_histogram
