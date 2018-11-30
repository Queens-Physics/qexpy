"""Operations for ExperimentalValue objects

This file implements the different operation overloads on ExperimentalValue objects. The
operations will return a DerivedValue object, with errors propagated properly. They are helper
methods which will be used in data.py

The operator methods all return the error propagated results used to construct DerivedValue
objects defined in data.py. For more details please refer to the default constructor of the
DerivedValue class

"""

import math as m
import warnings

from . import data
import qexpy.settings.literals as lit


def differentiator(operator):
    """Gets the derivative formula for an operator

    The differentiator is a lambda function calculates the derivative of an expression with
    respect to a target value. The first argument is a reference to the target value, and the
    rest of the arguments are the operands for this operation

    Usage:
        differentiator("MUL")(x, a, b) would return the derivative of the expression "a * b"
        with respect to the value x

    Args:
        operator (str): the operator of this expression (use string literals to avoid typos)

    """
    return differentiators[operator]


def execute(operator, operands):
    """Executes an operation with propagated results"""
    operation = operations[operator]
    return {
        lit.DERIVATIVE_PROPAGATED: (operation(*operands), propagate_error_derivative(operator, operands))
        # TODO: implement monte carlo error propagation
    }


def propagate_error_derivative(operator, operands):
    quadratures = map(lambda operand: (operand.error * differentiator(operator)(operand, *operands)) ** 2, operands)
    return m.sqrt(sum(quadratures))


def propagate_units(operator, operands):
    units = {}
    if operator == lit.MUL:
        for operand in operands:
            for unit, exponent in operand.get_units().items():
                units[unit] = (0 if unit not in units else units[unit]) + exponent
        return units
    elif operator == lit.DIV:
        for unit, exponent in operands[0].get_units().items():
            units[unit] = (0 if unit not in units else units[unit]) + exponent
        for unit, exponent in operands[1].get_units().items():
            units[unit] = (0 if unit not in units else units[unit]) - exponent
        return units
    elif operator in [lit.ADD, lit.SUB]:
        if len(operands[0].get_units()) == 0 and len(operands[1].get_units()) > 0:
            return operands[1].get_units()
        elif len(operands[1].get_units()) == 0 and len(operands[0].get_units()) > 0:
            return operands[0].get_units()
        elif operands[1].get_units() != operands[0].get_units():
            warnings.warn("You're trying to add/subtract two values with mismatching units, returning empty unit")
            return units
        else:
            return operands[0].get_units()
    elif operator == lit.NEG:
        return operands[0].get_units()


operations = {
    lit.NEG: lambda x: -x.value,
    lit.ADD: lambda a, b: a.value + b.value,
    lit.SUB: lambda a, b: a.value - b.value,
    lit.MUL: lambda a, b: a.value * b.value,
    lit.DIV: lambda a, b: a.value / b.value
}

# the usage of the differentiators are documented under the method differentiator
differentiators = {
    lit.NEG: lambda other, x: -x.derivative(other),
    lit.ADD: lambda other, a, b: a.derivative(other) + b.derivative(other),
    lit.SUB: lambda other, a, b: a.derivative(other) - b.derivative(other),
    lit.MUL: lambda other, a, b: a.derivative(other) * b.value + b.derivative(other) * a.value,
    lit.DIV: lambda other, a, b: (b.value * a.derivative(other) - a.value * b.derivative(other)) / (b.value ** 2)
}
