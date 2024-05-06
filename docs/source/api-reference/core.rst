.. _api.core:

====
Core
====

.. currentmodule:: qexpy.core

Base Class
----------

.. autosummary::
   :nosignatures:
   :toctree: api/

   ExperimentalValue

.. _api.core.measurement:

Measurements
------------

The :py:class:`Measurement` class extends :py:class:`~ExperimentalValue`. A measurement can be
recorded as a single value or a series of repeated takes, which
is interpreted as a single quantity measured multiple times to mitigate the uncertainty. In this
case, a subclass of :py:class:`Measurement` is created, that is the :py:class:`RepeatedMeasurement`.

.. autosummary::
   :nosignatures:
   :toctree: api/

   Measurement
   RepeatedMeasurement

.. _api.core.derived_value:

DerivedValue
------------

The result of all calculations performed with :py:class:`ExperimentalValue` objects are wrapped in
this class. The DerivedValue supports different methods of error propagation.

.. autosummary::
   :nosignatures:
   :toctree: api/

   DerivedValue
   MonteCarloConfig

.. _api.core.constants:

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: api/

   Constant
