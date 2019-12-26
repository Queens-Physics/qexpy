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
.. autoattribute:: qexpy.data.data.DerivedValue.mc

Methods
-------

.. automethod:: qexpy.data.data.DerivedValue.reset_error_method
.. automethod:: qexpy.data.data.DerivedValue.recalculate
.. automethod:: qexpy.data.data.DerivedValue.show_error_contributions

The MonteCarloSettings Object
=============================

QExPy provides users with many options to customize Monte Carlo error propagation. Each :py:class:`DerivedValue` object stores a :py:class:`MonteCarloSettings` object that contains some settings for the Monte Carlo error propagation of this value.

.. autoclass:: qexpy.data.utils.MonteCarloSettings

Properties
----------

.. autoattribute:: qexpy.data.utils.MonteCarloSettings.sample_size
.. autoattribute:: qexpy.data.utils.MonteCarloSettings.confidence
.. autoattribute:: qexpy.data.utils.MonteCarloSettings.xrange

Methods
-------

.. automethod:: qexpy.data.utils.MonteCarloSettings.set_xrange
.. automethod:: qexpy.data.utils.MonteCarloSettings.use_mode_with_confidence
.. automethod:: qexpy.data.utils.MonteCarloSettings.use_mean_and_std
.. automethod:: qexpy.data.utils.MonteCarloSettings.show_histogram
.. automethod:: qexpy.data.utils.MonteCarloSettings.samples
.. automethod:: qexpy.data.utils.MonteCarloSettings.use_custom_value_and_error
