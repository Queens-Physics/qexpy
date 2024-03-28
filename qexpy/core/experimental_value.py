"""Defines the base class for all experimental values."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import qexpy as q
from qexpy.utils.formatter import format_value_error
from qexpy.utils.units import Unit


class ExperimentalValue(ABC):
    """Base class for a value with an uncertainty.

    Measurements recorded with QExPy and the result of calculations performed with QExPy values
    are stored as instances of this class.

    Attributes
    ----------

    value : float
    error : float
    relative_error : float
    name : str
    unit : Unit

    Examples
    --------

    Values are recorded as a :func:`~qexpy.Measurement`.

    >>> import qexpy as q
    >>> distance = q.Measurement(5, 0.1, name="distance", unit="m")

    The string representation of a value contains the name and the unit.

    >>> print(distance)
    distance = 5.0 +/- 0.1 [m]

    You can also access the centre value and the uncertainty individually.

    >>> print(distance.value)
    5.0
    >>> print(distance.error)
    0.1

    When using `ExperimentalValue` objects in calculations, the results will have properly
    propagated uncertainties. The units will also participate in calculations.

    >>> time = q.Measurement(10, 0.1, name="time", unit="s")
    >>> speed = distance / time
    >>> print(speed)
    0.50 +/- 0.01 [m/s]

    """

    def __init__(self, name: str = "", unit: str = ""):
        self.name = name
        self.unit = unit

    def __str__(self):
        name = f"{self.name} = " if self.name else ""
        unit = f" [{self.unit}]" if self.unit else ""
        return f"{name}{format_value_error(self.value, self.error)}{unit}"

    def __repr__(self):
        return self.__str__()

    @property
    @abstractmethod
    def value(self) -> float:
        """The centre value of this quantity.

        :type: float

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def error(self) -> float:
        """The uncertainty on the value.

        :type: float

        """
        raise NotImplementedError

    @property
    def relative_error(self) -> float:
        """The ratio between the error and the value

        The relative error is defined as ``abs(error / value)``

        :type: float

        """
        return np.abs(self.error / self.value) if self.value != 0 else 0.0

    @property
    def name(self) -> str:
        """The name of the value

        If ``name`` is specified, it will be included in the string representation.

        :type: str

        Examples
        --------

        >>> import qexpy as q
        >>> a = q.Measurement(5, 0.1, name="x")
        >>> print(a)
        x = 5.0 +/- 0.1

        """
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("The name must be a string!")
        self._name = name

    @property
    def unit(self) -> Unit:
        """The unit of the quantity.

        The unit is internally stored as a dictionary, but represented as a string. The user can
        specify the unit of a quantity, and it will participate in future calculations. Units can
        be specified and displayed in different ways.

        :type: Unit

        Examples
        --------

        >>> import qexpy as q

        Units can be specified when recording a measurement:

        >>> a = q.Measurement(5, 0.5, unit="kg*m/s^2")
        >>> a.unit
        'kg⋅m/s^2'
        >>> print(a)
        5.0 +/- 0.1 [kg⋅m/s^2]

        Units can also be updated post-measurement:

        >>> b = q.Measurement(5, 0.5)
        >>> b.unit = "kg^2m^2s^-4"
        >>> b.unit
        'kg^2⋅m^2/s^4'

        By default, units with negative exponents are displayed as denominators of a fraction,
        but this can be changed by setting the ``format.style.unit`` option to "exponent".

        >>> q.options.format.style.unit = "exponent"
        >>> b.unit
        'kg^2⋅m^2⋅s^-4'

        The units will also participate in calculations.

        >>> c = b / a
        >>> c.unit
        'kg⋅m⋅s^-2'

        """
        return self._unit

    @unit.setter
    def unit(self, unit: Unit | str):
        if not isinstance(unit, (Unit, str)):
            raise TypeError(f"The unit must be a string or a Unit object, got: {type(unit)}")
        self._unit = unit if isinstance(unit, Unit) else Unit.from_string(unit)
