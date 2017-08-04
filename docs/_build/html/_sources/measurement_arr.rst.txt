The MeasurementArray Object
===========================

When making a series of Measurements that need to be grouped together (i.e. for plotting or group calculations) a MeasurementArray should be used. This section describes the uses of MeasurementArray objects.

.. autofunction:: qexpy.error.MeasurementArray 
.. autoclass:: qexpy.error.Measurement_Array

Properties
----------
MeasurementArrays have a few properties that you can get and set. Each property has an example of how to get and set it.

.. autoattribute:: qexpy.error.Measurement_Array.error_weighted_mean
.. autoattribute:: qexpy.error.Measurement_Array.mean
.. autoattribute:: qexpy.error.Measurement_Array.means
.. automethod:: qexpy.error.Measurement_Array.std
.. autoattribute:: qexpy.error.Measurement_Array.stds
.. autoattribute:: qexpy.error.Measurement_Array.units

Functions
---------
MeasurementArrays also have a number of functions that can be used to change the MeasurementArray or get information about it.

.. automethod:: qexpy.error.Measurement_Array.append(meas)
.. automethod:: qexpy.error.Measurement_Array.delete
.. automethod:: qexpy.error.Measurement_Array.insert
.. automethod:: qexpy.error.Measurement_Array.get_units_str
.. automethod:: qexpy.error.Measurement_Array.show_table