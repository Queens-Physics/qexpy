"""Defines the base class for a quantity in experimental data analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from qexpy.core.operations import (
    absolute,
    add,
    array_ufunc,
    divide,
    multiply,
    negate,
    power,
    rdivide,
    rpower,
    rsubtract,
    subtract,
)
from qexpy.format import format_value_error
from qexpy.typing import Number
from qexpy.units import Unit, UnitLike


class Quantity(ABC):
    """Base class for a value with an uncertainty."""

    def __init__(self, name: str = "", unit: UnitLike = ""):
        if not isinstance(name, str):
            raise TypeError(f"The name must be a string, got {type(name)}.")
        self._name = name
        if not isinstance(unit, UnitLike):
            raise TypeError(f"The unit mast be a str, got {type(unit)}.")
        self._unit = Unit(unit)

    @property
    @abstractmethod
    def value(self) -> float:
        """The value of this quantity.

        :type: float

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def error(self) -> float:
        """The uncertainty on the value.

        :type: float

        """

    @property
    def relative_error(self) -> float:
        """The ration between the error and the centre value.

        :type: float

        The relative error is defined as ``abs(error / value)``.

        """
        if self.error == 0:
            return 0.0
        if self.value == 0:
            return np.inf
        return np.abs(self.error / self.value)

    @property
    def name(self) -> str:
        """The name of this quantity.

        :type: str

        """
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError(f"The name must be a string, got {type(name)}.")
        self._name = name

    @property
    def unit(self) -> Unit:
        """The unit of this quantity.

        :type: Unit

        """
        return self._unit

    @unit.setter
    def unit(self, unit):
        if not isinstance(unit, UnitLike):
            raise TypeError(f"The unit mast be a str, got {type(unit)}.")
        self._unit = Unit(unit)

    def __str__(self) -> str:
        name = f"{self.name} = " if self.name else ""
        unit = f" [{self.unit}]" if self.unit else ""
        return f"{name}{format_value_error(self.value, self.error)}{unit}"

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, Number):
            return self.value == other
        if isinstance(other, Quantity):
            return self.value == other.value
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, Number):
            return self.value < other
        if isinstance(other, Quantity):
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Number):
            return self.value > other
        if isinstance(other, Quantity):
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Number):
            return self.value <= other
        if isinstance(other, Quantity):
            return self.value <= other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Number):
            return self.value >= other
        if isinstance(other, Quantity):
            return self.value >= other.value
        return NotImplemented

    def _derivative(self, x):
        return 0.0

    def __abs__(self) -> Quantity:
        return absolute(self)

    def __add__(self, other) -> Quantity:
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other) -> Quantity:
        return subtract(self, other)

    def __rsub__(self, other) -> Quantity:
        return rsubtract(self, other)

    def __mul__(self, other) -> Quantity:
        return multiply(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other) -> Quantity:
        return divide(self, other)

    def __rtruediv__(self, other) -> Quantity:
        return rdivide(self, other)

    def __pow__(self, other) -> Quantity:
        return power(self, other)

    def __rpow__(self, other) -> Quantity:
        return rpower(self, other)

    def __neg__(self) -> Quantity:
        return negate(self)

    def __array_ufunc__(self, ufunc: Callable, _: str, *inputs, **__) -> Quantity:
        return array_ufunc(self, ufunc, *inputs)
