.. _getting_started.tutorial:

===================
Beginner's Tutorial
===================

Start by importing ``qexpy`` into your code like this:

.. ipython:: python

    import qexpy as q

The top level namespace contains most functionalities of qexpy.

Measurements
------------

To record a measurement, use :class:`q.Measurement <qexpy.core.Measurement>`:

.. ipython:: python

    m = q.Measurement(12.3, 0.25)
    m

Optionally, a measurement can be recorded with a name and a unit:

.. ipython:: python

    m = q.Measurement(12.3, 0.25, unit="kg", name="mass")
    m

A common error mitigation technique is to take repeated measurements of the
same quantity and combine them into a single measured value:

.. ipython:: python

   t = q.Measurement([5, 4.9, 5.3, 4.7, 4.8, 5.3], unit="s", name="time")
   t

The estimated value of this quantity is given by the mean of the samples. If
individual measurement uncertainties are unknown, the estimated uncertainty
is given by the standard error on the mean of the samples. Alternatively, if 
individual measurement uncertainties are known, they can be provided to obtain
a more accurate estimate of the value and uncertainty:

.. ipython:: python

    t = q.Measurement(
        [5, 4.9, 5.3, 4.7, 4.8, 5.3],
        error=[0.5, 0.25, 0.5, 0.5, 0.25, 0.5],
        unit="s",
        name="time",
    )
    t

In this case, the estimated value of this quantity is the error-weighted mean
of the samples, and the error is also derived from the individual measurement
uncertainties.
