The Measurement Object
======================

The propagation of measurements, including the propagation of errors is at
the heart of this package. This section will describe how Measurement
objects are created as well as all the methods available to operate on Measurements.

.. autoclass:: qexpy.error.Measurement
   :members: print_mc_error, print_min_max_error, print_deriv_error, get_derivative, mean, std, error_on_mean, name, relative_error, error_method, get_data_array, get_units, show_histogram, show_MC_histogram, show_error_contribution, set_correlation, set_covariance, get_covariance, rename
