"""Define the data structure for derived values."""

from __future__ import annotations

from typing_extensions import override

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


class DerivedValue(Quantity):
    """A value derived from other quantities."""

    def __init__(self, formula: Formula):
        self._formula = formula
        super().__init__()

    @property
    @override
    def value(self) -> float:
        raise NotImplementedError

    @property
    @override
    def error(self) -> float:
        raise NotImplementedError


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


@array_ufunc.register
def _(var: Quantity, ufunc, *inputs):
    if ufunc not in OP_TO_FORMULA:
        return NotImplemented
    inputs = (to_formula(v) for v in inputs)
    formula = OP_TO_FORMULA[ufunc](*inputs)
    return DerivedValue(formula)
