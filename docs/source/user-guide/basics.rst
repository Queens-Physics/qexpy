.. _user_guide.basics:

==========
The Basics
==========

This is an instroduction to the fundamentals of qexpy. We will cover how to
record measurements, perform calculations, and manage value objects.

Measurements
------------

The first step to experimental data analysis is to record measurements. A
measurement can be recorded by creating a :class:`q.Measurement <qexpy.core.Measurement>`:

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
of the samples:

.. math::
    \mu_x = \frac{\sum_i w_i x_i}{\sum_i w_i}

where :math:`w_i` is defined as

.. math::
    w_i \equiv \frac{1}{\sigma_{x_i}^2}

where :math:`\sigma_{x_i}` is the uncertainty of the :math:`i`-th measurement. 
The estimated uncertainty of the measured quantity is given by:

.. math::
    \sigma_x = \sqrt{\frac{1}{\sum_i{w_i}}}

This method gives more importance to the more precise measurements, and it is
generally the recommended method when you have a reliable estimate of each
individual measurement's uncertainty. Alternatively, you can choose to diregard
the individual errors completely and estimate the uncertainty from the observed
scatter of the samples:

.. ipython:: python

    t.use_standard_error()
    t

Format and Style
----------------

The display format of values can be customized. For example, to display values
using the scientific notation, you can set ``options.format.value`` to ``"scientific"``:

.. ipython:: python

    q.options.format.value = "scientific"
    q.G

By default, the uncertainty is displayed with 1 significant figure, and the
value matches the digit of the uncertainty. You can change this behaviour:

.. ipython:: python

    q.options.format.precision.sigfigs = 2
    q.G

The display format of the units can also be customized:

.. ipython:: python

    q.options.format.unit = "product"
    q.G
