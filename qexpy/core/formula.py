"""Defines the data structure for a formula"""

# pylint: disable=protected-access

from __future__ import annotations

import itertools
from abc import abstractmethod, ABC
from numbers import Real
from typing import Iterable, Set

import numpy as np

import qexpy as q
from qexpy.utils import Unit


class _Formula(ABC):

    @classmethod
    def _wraps(cls, operand):
        """Construct a formula from an operand or return the operand itself."""
        if isinstance(operand, q.core.DerivedValue):
            return operand._formula
        if isinstance(operand, Real):
            return q.core.Constant(operand)
        if isinstance(operand, _Formula):
            return operand
        raise TypeError(f"Cannot perform operation with variable of type {type(operand)}!")

    @property
    @abstractmethod
    def value(self) -> float | np.ndarray:
        """The value of this formula."""
        raise NotImplementedError

    @property
    def error(self) -> float:
        """Error of this formula propagated using the derivative method."""
        # Find the measurements at the root of this formula tree
        sources = _find_measurements(self)
        # Calculate variance by adding quadratures
        quadratures = [(x.error * self._derivative(x)) ** 2 for x in sources]
        covariances = list(_covariance_terms(self, sources))
        variance = np.sum(quadratures) + np.sum(covariances)
        return np.sqrt(variance)

    @property
    @abstractmethod
    def unit(self) -> Unit:
        """The unit of this formula."""
        raise NotImplementedError

    @abstractmethod
    def _derivative(self, x: _Formula) -> float:
        """The derivative with respect to the given variable."""
        raise NotImplementedError


class _Operation(_Formula, ABC):

    @property
    @abstractmethod
    def operands(self) -> Iterable[_Formula]:
        """The operands of this operation."""
        raise NotImplementedError


class _UnaryOp(_Operation, ABC):

    def __init__(self, operand: _Formula):
        self.operand = operand

    @property
    def unit(self):
        return Unit({})

    @property
    def operands(self) -> Iterable[_Formula]:
        return (self.operand,)


class _BinaryOp(_Operation, ABC):

    left: _Formula
    right: _Formula

    def __init__(self, left: _Formula, right: _Formula):
        self.left = left
        self.right = right

    @property
    def operands(self) -> Iterable[_Formula]:
        return self.left, self.right


class _Add(_BinaryOp):

    @property
    def value(self):
        return self.left.value + self.right.value

    @property
    def unit(self):
        return self.left.unit + self.right.unit

    def _derivative(self, x):
        return self.left._derivative(x) + self.right._derivative(x)


class _Subtract(_BinaryOp):

    @property
    def value(self):
        return self.left.value - self.right.value

    @property
    def unit(self):
        return self.left.unit - self.right.unit

    def _derivative(self, x):
        return self.left._derivative(x) - self.right._derivative(x)


class _Multiply(_BinaryOp):

    @property
    def value(self):
        return self.left.value * self.right.value

    @property
    def unit(self):
        return self.left.unit * self.right.unit

    def _derivative(self, x):
        d1 = self.left.value * self.right._derivative(x)
        d2 = self.right.value * self.left._derivative(x)
        return d1 + d2


class _Divide(_BinaryOp):

    @property
    def value(self):
        return self.left.value / self.right.value

    @property
    def unit(self):
        return self.left.unit / self.right.unit

    def _derivative(self, x):
        num1 = self.right.value * self.left._derivative(x)
        num2 = self.left.value * self.right._derivative(x)
        return (num1 - num2) / self.right.value**2


class _Power(_BinaryOp):

    @property
    def value(self):
        return self.left.value**self.right.value

    @property
    def unit(self):
        return self.left.unit**self.right.value

    def _derivative(self, x):
        """Derivative of f(x)^g(x) with respect to x

        The derivative of f(x)^g(x) includes a term that involves taking the log of f(x). This
        is not necessary if g(x) is a constant. In some cases where f(x) is negative, the log of
        f(x) would return `nan`, making the whole expression `nan`. This should not need to happen
        if d/dx of g(x) is 0, which eliminates the nan term. Since in Python nan * 0 returns nan
        instead of 0, this helper function is written so that this is properly handled

        .. math::

            d/dx f(x)^{g(x)} = f(x)^{(g(x)-1)}(g(x)f'(x)+f(x)log(f(x))g'(x))

        """
        base = self.left
        exponent = self.right
        leading = base.value ** (exponent.value - 1)
        second = exponent.value * base._derivative(x)
        if exponent._derivative(x) != 0:
            second += base.value * np.log(base.value) * exponent._derivative(x)
        return leading * second


class _NegativeOp(_UnaryOp):

    @property
    def value(self):
        return -self.operand.value

    @property
    def unit(self):
        return self.operand.unit

    def _derivative(self, x):
        return -self.operand._derivative(x)


class _Sqrt(_UnaryOp):

    @property
    def value(self):
        return np.sqrt(self.operand.value)

    @property
    def unit(self):
        return self.operand.unit ** (1 / 2)

    def _derivative(self, x):
        return self.operand._derivative(x) / (2 * np.sqrt(self.operand.value))


class _Sin(_UnaryOp):

    @property
    def value(self):
        return np.sin(self.operand.value)

    def _derivative(self, x):
        return np.cos(self.operand.value) * self.operand._derivative(x)


class _Cos(_UnaryOp):

    @property
    def value(self):
        return np.cos(self.operand.value)

    def _derivative(self, x):
        return -np.sin(self.operand.value) * self.operand._derivative(x)


class _Tan(_UnaryOp):

    @property
    def value(self):
        return np.tan(self.operand.value)

    def _derivative(self, x):
        return 1 / np.cos(self.operand.value) ** 2 * self.operand._derivative(x)


class _Asin(_UnaryOp):

    @property
    def value(self):
        return np.arcsin(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / np.sqrt(1 - self.operand.value**2)


class _Acos(_UnaryOp):

    @property
    def value(self):
        return np.arccos(self.operand.value)

    def _derivative(self, x):
        return -self.operand._derivative(x) / np.sqrt(1 - self.operand.value**2)


class _Atan(_UnaryOp):

    @property
    def value(self):
        return np.arctan(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / (self.operand.value**2 + 1)


class _Atan2(_BinaryOp):

    @property
    def value(self):
        return np.arctan2(self.left.value, self.right.value)

    def _derivative(self, x):
        return (
            self.right.value * self.left._derivative(x)
            - self.left.value * self.right._derivative(x)
        ) / (self.left.value**2 + self.right.value**2)

    @property
    def unit(self) -> Unit:
        return Unit({})


class _Sinh(_UnaryOp):

    @property
    def value(self):
        return np.sinh(self.operand.value)

    def _derivative(self, x):
        return np.cosh(self.operand.value) * self.operand._derivative(x)


class _Cosh(_UnaryOp):

    @property
    def value(self):
        return np.cosh(self.operand.value)

    def _derivative(self, x):
        return np.sinh(self.operand.value) * self.operand._derivative(x)


class _Tanh(_UnaryOp):

    @property
    def value(self):
        return np.tanh(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / np.cosh(self.operand.value) ** 2


class _Asinh(_UnaryOp):

    @property
    def value(self):
        return np.arcsinh(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / np.sqrt(self.operand.value**2 + 1)


class _Acosh(_UnaryOp):

    @property
    def value(self):
        return np.arccosh(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / np.sqrt(self.operand.value**2 - 1)


class _Atanh(_UnaryOp):

    @property
    def value(self):
        return np.arctanh(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / (1 - self.operand.value**2)


class _Exp(_UnaryOp):

    @property
    def value(self):
        return np.exp(self.operand.value)

    def _derivative(self, x):
        return np.exp(self.operand.value) * self.operand._derivative(x)


class _Log2(_UnaryOp):

    @property
    def value(self):
        return np.log2(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / (np.log(2) * self.operand.value)


class _Log10(_UnaryOp):

    @property
    def value(self):
        return np.log10(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / (np.log(10) * self.operand.value)


class _Ln(_UnaryOp):

    @property
    def value(self):
        return np.log(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / self.operand.value


def _find_measurements(formula: _Formula) -> Set[q.core.Measurement]:
    """Find the measurements that this formula depends on."""

    if isinstance(formula, q.core.Measurement):
        return {formula}

    if isinstance(formula, _Operation):
        return set.union(*[_find_measurements(operand) for operand in formula.operands])

    return set()


def _covariance_terms(formula: _Formula, sources: Set[q.core.Measurement]) -> Iterable:
    """Finds the contributing covariance terms for the derivative method"""

    for var1, var2 in itertools.combinations(sources, 2):
        corr = q.correlation(var1, var2)
        # Re-calculate the covariance between two measurements, because in the
        # case of repeated measurements, sometimes the covariance is calculated
        # from the raw measurements, which is closely coupled with the standard
        # deviation of these samples. This is misleading because with repeated
        # measurements, we use the error on the mean, not the standard deviation
        # of the raw measurements, as the error. Essentially, with repeatedly
        # measured values, we are ignoring the array of raw measurements, and
        # treating its value and error as the mean and standard deviation just
        # like we would with any other single measurements.
        cov = corr * var1.error * var2.error
        if cov != 0:
            yield 2 * cov * formula._derivative(var1) * formula._derivative(var2)


OP_TO_FORMULA = {
    np.sqrt: _Sqrt,
    np.sin: _Sin,
    np.cos: _Cos,
    np.tan: _Tan,
    np.arcsin: _Asin,
    np.arccos: _Acos,
    np.arctan: _Atan,
    np.sinh: _Sinh,
    np.cosh: _Cosh,
    np.tanh: _Tanh,
    np.arcsinh: _Asinh,
    np.arccosh: _Acosh,
    np.arctanh: _Atanh,
    np.arctan2: _Atan2,
    np.exp: _Exp,
    np.log2: _Log2,
    np.log10: _Log10,
    np.log: _Ln,
    np.add: _Add,
    np.subtract: _Subtract,
    np.multiply: _Multiply,
    np.divide: _Divide,
    np.power: _Power,
    np.negative: _NegativeOp,
}
