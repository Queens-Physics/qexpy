"""Define the data structure for derived values."""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import override

from qexpy._config.config import options
from qexpy.core.monte_carlo import MonteCarloController
from qexpy.units import Unit, UnitLike

from .formula import (
    OP_TO_FORMULA,
    Formula,
    _Add,
    _Divide,
    _Multiply,
    _NegativeOp,
    _Power,
    _Subtract,
    to_formula,
)
from .operations import (
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
from .quantity import Quantity


class ErrorMethod(StrEnum):
    """The method of error propagation."""

    DERIVATIVE = "derivative"
    MONTE_CARLO = "monte-carlo"
    AUTO = "auto"


class DerivedValue(Quantity):
    """A value derived from other quantities."""

    def __init__(self, formula: Formula):
        self._formula = formula
        self._error_method = ErrorMethod.AUTO
        self._mc = MonteCarloController(formula)
        super().__init__()

    @property
    @override
    def value(self) -> float:
        if self.error_method == ErrorMethod.MONTE_CARLO:
            return self.mc.value
        return float(self._formula.value)

    @property
    @override
    def error(self) -> float:
        if self.error_method == ErrorMethod.MONTE_CARLO:
            return self.mc.error
        return float(self._formula.error)

    @property
    @override
    def unit(self) -> Unit:
        return self._formula.unit

    @property
    def mc(self) -> MonteCarloController:
        """The controller for the Monte Carlo error method."""
        return self._mc

    @property
    def error_method(self) -> ErrorMethod:
        """The method of error propagation used for this value."""
        if self._error_method == ErrorMethod.AUTO:
            return options.error.method
        return self._error_method

    @error_method.setter
    def error_method(self, method: str):
        """Sets the method of error propagation for this value."""
        try:
            self._error_method = ErrorMethod(method)
        except ValueError as e:
            raise ValueError(
                f"{method} is not a valid error method. Accepted values are: "
                "'derivative', 'monte-carlo', or 'auto'."
            ) from e

    @unit.setter
    def unit(self, unit: UnitLike):
        if not isinstance(unit, UnitLike):
            raise TypeError(f"The unit mast be a str, got {type(unit)}.")
        self._unit = Unit(unit)


@to_formula.register
def _(obj: DerivedValue):
    return obj._formula


@add.register
def _(var1: Quantity, var2):
    formula = _Add(to_formula(var1), to_formula(var2))
    return DerivedValue(formula)


@subtract.register
def _(var1: Quantity, var2):
    formula = _Subtract(to_formula(var1), to_formula(var2))
    return DerivedValue(formula)


@rsubtract.register
def _(var1: Quantity, var2):
    formula = _Subtract(to_formula(var2), to_formula(var1))
    return DerivedValue(formula)


@multiply.register
def _(var1: Quantity, var2):
    formula = _Multiply(to_formula(var1), to_formula(var2))
    return DerivedValue(formula)


@divide.register
def _(var1: Quantity, var2):
    formula = _Divide(to_formula(var1), to_formula(var2))
    return DerivedValue(formula)


@rdivide.register
def _(var1: Quantity, var2):
    formula = _Divide(to_formula(var2), to_formula(var1))
    return DerivedValue(formula)


@power.register
def _(var1: Quantity, var2):
    formula = _Power(to_formula(var1), to_formula(var2))
    return DerivedValue(formula)


@rpower.register
def _(var1: Quantity, var2):
    formula = _Power(to_formula(var2), to_formula(var1))
    return DerivedValue(formula)


@negate.register
def _(var: Quantity):
    formula = _NegativeOp(to_formula(var))
    return DerivedValue(formula)


@absolute.register
def _(var: Quantity):
    return -var if var < 0 else DerivedValue(to_formula(var))


@array_ufunc.register
def _(var: Quantity, ufunc, *inputs):
    if ufunc not in OP_TO_FORMULA:
        return NotImplemented
    inputs = (to_formula(v) for v in inputs)
    formula = OP_TO_FORMULA[ufunc](*inputs)
    return DerivedValue(formula)
