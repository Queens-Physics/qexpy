"""Defines the base class for a quantity in experimental data analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qexpy.format import format_value_error
from qexpy.typing import Number
from qexpy.units import Unit, UnitLike


class Quantity(ABC):
    """Base class for a value with an uncertainty."""

    def __init__(self, name: str, unit: UnitLike):
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
