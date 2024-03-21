"""Defines measurements"""

# pylint: disable=too-many-arguments

from __future__ import annotations

from numbers import Real

from qexpy._typing import ArrayLikeT
from .experimental_value import ExperimentalValue


class Measurement(ExperimentalValue):
    """A measurement recorded with an uncertainty

    Stores a measured value and the uncertainty of the measurement on a single quantity. A
    measurement can be recorded as a single measurement or a series of repeated measurements
    taken on the same quantity.

    The latter case is not to be confused with an array of measurements, which should be taken
    with :py:func:`qexpy.MeasurementArray`. Think of this as recording a single data point,
    measured multiple times to reduce the uncertainty.

    Parameters
    ----------

    data : float or array
        The measured value(s) of this quantity.
    error : float or array, default=0
        The uncertainty of the measurement. If ``data`` is an array, this can be a single value
        that represents the uncertainty of all measurements, or an array that correspond to the
        uncertainties of each measurement in ``data``.
    relative_error : float or array, optional
        The relative error of the measurement. If provided, it will override ``error``.
    name : str, optional
        The name of this quantity (e.g. ``"mass"``, ``"voltage"``).
    unit : str, optional
        The unit of this quantity (e.g. ``"kg*m/s^2"``, ``"m^1s^-2"``).

    Examples
    --------

    >>> import qexpy as q
    >>> a = q.Measurement(5, 0.5, name='mass', unit='kg')
    >>> a
    mass = 5.0 +/- 0.5 [kg]

    And to take an array of repeated measurements:

    >>> b = q.Measurement([4.9, 4.8, 5])
    >>> b
    4.90 +/- 0.06

    See Also
    --------

    :class:`~qexpy.core.Measurement`
    :class:`~qexpy.core.RepeatedMeasurement`

    """

    def __new__(
        cls,
        data: Real | ArrayLikeT,
        error: Real | ArrayLikeT = 0.0,
        relative_error: Real | ArrayLikeT = None,
        name: str = "",
        unit: str = "",
    ):
        if isinstance(data, Real) and isinstance(error, Real):
            return object.__new__(Measurement)

        if isinstance(data, ArrayLikeT):
            return object.__new__(RepeatedMeasurement)

        raise TypeError("Invalid data types for a measurement!")

    def __init__(
        self,
        value: Real,
        error: Real = 0.0,
        relative_error: Real = None,
        name: str = "",
        unit: str = "",
    ):
        self._value = float(value)
        error = float(error) if relative_error is None else float(relative_error) * float(value)
        if error < 0:
            raise ValueError("The error must be non-negative!")
        self._error = error
        super().__init__(name=name, unit=unit)

    @property
    def value(self) -> float:
        return self._value

    @property
    def error(self) -> float:
        return self._error

    @property
    def std(self) -> float:
        """The standard deviation of the measured value

        For a single measurement, the standard deviation is just the error.

        :type: float

        """
        return self.error


class RepeatedMeasurement(Measurement):
    """A single quantity measured in multiple takes

    The ``RepeatedMeasurement`` stores the array of repeated measurements. By default, its value
    and error are the mean and standard error (error on mean) of the samples.

    """
