"""Defines the DerivedValue class"""

from functools import cached_property

import qexpy as q
from qexpy.core.experimental_value import ExperimentalValue
from qexpy.core.formula import _Formula
from qexpy.utils import Unit


class DerivedValue(ExperimentalValue):
    """A calculated value with a propagated uncertainty"""

    def __init__(self, formula: _Formula):
        self._formula = formula
        super().__init__("", None)

    def __copy__(self):
        obj = object.__new__(DerivedValue)
        obj._formula = self._formula
        obj._name = self._name
        return obj

    @property
    def value(self) -> float:
        return self._value

    @property
    def error(self) -> float:
        return self._error

    @property
    def unit(self) -> Unit:
        return self._unit

    @cached_property
    def _value(self):
        return self._formula.value

    @cached_property
    def _error(self):
        return self._formula.error

    @cached_property
    def _unit(self):  # pylint: disable=method-hidden
        return self._formula.unit
