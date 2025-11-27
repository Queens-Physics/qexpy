"""The module contains the internal data structure for an expression tree."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import singledispatch
from typing import Protocol, runtime_checkable

import numpy as np

from qexpy.units import Unit

from .functions import correlation
from .measurements import Measurement


@singledispatch
def to_formula(obj) -> Formula:
    """Wrap an object in a Formula."""
    if isinstance(obj, Formula):
        return obj
    raise TypeError(f"Variable of type {type(obj)} is not supported in operations.")


@runtime_checkable
class Formula(Protocol):
    """An abstract expression tree that represents a formula."""

    @property
    def value(self) -> float | np.ndarray:
        """The value of this expression."""
        raise NotImplementedError

    @property
    def error(self) -> float:
        """The uncertainty of this expression."""
        raise NotImplementedError

    @property
    def unit(self) -> Unit:
        """The unit of this expression."""
        raise NotImplementedError

    def _derivative(self, x: Formula) -> float:
        """The derivative of this expression with respect to a variable."""
        raise NotImplementedError


class _Operation(ABC):
    """An operation node in an expression tree."""

    @abstractmethod
    def __init__(self, *args: Formula):
        """Dummy initializer defined for syntax inspection."""

    @property
    @abstractmethod
    def value(self) -> float | np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def operands(self) -> tuple[Formula, ...]:
        """The operands of this operation."""
        raise NotImplementedError

    @property
    def error(self) -> float:
        """Error of this formula propagated using the derivative method."""
        # Find the measurements at the root of this formula tree
        sources = _collect_measurements(self)
        # Calculate the variance by adding quadratures
        quadratures = ((x.error * self._derivative(x)) ** 2 for x in sources)
        sum_quadratures = np.sum(np.fromiter(quadratures, float))
        covariances = _covariance_terms(self, sources)
        sum_covariances = np.sum(np.fromiter(covariances, float))
        return np.sqrt(sum_quadratures + sum_covariances)

    @property
    @abstractmethod
    def unit(self) -> Unit:
        raise NotImplementedError

    @abstractmethod
    def _derivative(self, x: Formula) -> float:
        raise NotImplementedError


class _UnaryOp(_Operation):
    """A unary operation node."""

    def __init__(self, operand: Formula):
        self.operand = operand
        super().__init__()

    @property
    def unit(self):
        return Unit({})

    @property
    def operands(self) -> tuple[Formula]:
        return (self.operand,)


class _BinaryOp(_Operation):
    """A binary operation node."""

    left: Formula
    right: Formula

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right
        super().__init__()

    @property
    def operands(self) -> tuple[Formula, Formula]:
        return self.left, self.right


class _Add(_BinaryOp):
    """The add operation."""

    @property
    def value(self):
        return self.left.value + self.right.value

    @property
    def unit(self):
        return self.left.unit + self.right.unit

    def _derivative(self, x):
        return self.left._derivative(x) + self.right._derivative(x)


class _Subtract(_BinaryOp):
    """The subtract operation."""

    @property
    def value(self):
        return self.left.value - self.right.value

    @property
    def unit(self):
        return self.left.unit - self.right.unit

    def _derivative(self, x):
        return self.left._derivative(x) - self.right._derivative(x)


class _Multiply(_BinaryOp):
    """The multiply operation."""

    @property
    def value(self):
        return self.left.value * self.right.value

    @property
    def unit(self):
        return self.left.unit * self.right.unit

    def _derivative(self, x):
        d1 = self.left.value * self.right._derivative(x)
        d2 = self.right.value * self.left._derivative(x)
        return float(d1 + d2)


class _Divide(_BinaryOp):
    """The division operation."""

    @property
    def value(self):
        return self.left.value / self.right.value

    @property
    def unit(self):
        return self.left.unit / self.right.unit

    def _derivative(self, x):
        num1 = self.right.value * self.left._derivative(x)
        num2 = self.left.value * self.right._derivative(x)
        return float((num1 - num2) / self.right.value**2)


class _Power(_BinaryOp):
    """The power operation."""

    @property
    def value(self):
        return self.left.value**self.right.value

    @property
    def unit(self):
        return self.left.unit**self.right.value

    def _derivative(self, x):
        """Derivative of f(x)^g(x) with respect to x.

        The derivative of f(x)^g(x) includes a term that involves taking the log of
        f(x). This is not necessary if g(x) is a constant. In some cases where f(x)
        is negative, the log of f(x) would return `nan`, making the whole expression
        `nan`. This should not need to happen if d/dx of g(x) is 0, which eliminates
        the nan term. Since in Python nan * 0 returns nan instead of 0, this helper
        function is written so that this is properly handled

        .. math::

            d/dx f(x)^{g(x)} = f(x)^{(g(x)-1)}(g(x)f'(x)+f(x)log(f(x))g'(x))

        """
        base = self.left
        exponent = self.right
        leading = base.value ** (exponent.value - 1)
        second = exponent.value * base._derivative(x)
        if exponent._derivative(x) != 0:
            second += base.value * np.log(base.value) * exponent._derivative(x)
        return float(leading * second)


class _NegativeOp(_UnaryOp):
    """The negation operation."""

    @property
    def value(self):
        return -self.operand.value

    @property
    def unit(self):
        return self.operand.unit

    def _derivative(self, x):
        return -self.operand._derivative(x)


class _Sqrt(_UnaryOp):
    """The square root operation."""

    @property
    def value(self):
        return np.sqrt(self.operand.value)

    @property
    def unit(self):
        return self.operand.unit ** (1 / 2)

    def _derivative(self, x):
        return self.operand._derivative(x) / float(2 * np.sqrt(self.operand.value))


class _Sin(_UnaryOp):
    """The sine function."""

    @property
    def value(self):
        return np.sin(self.operand.value)

    def _derivative(self, x):
        return float(np.cos(self.operand.value) * self.operand._derivative(x))


class _Cos(_UnaryOp):
    """The cosine function."""

    @property
    def value(self):
        return np.cos(self.operand.value)

    def _derivative(self, x):
        return float(-np.sin(self.operand.value) * self.operand._derivative(x))


class _Tan(_UnaryOp):
    """The tangent function."""

    @property
    def value(self):
        return np.tan(self.operand.value)

    def _derivative(self, x):
        return float(1 / np.cos(self.operand.value) ** 2 * self.operand._derivative(x))


class _Asin(_UnaryOp):
    """The arcsin function."""

    @property
    def value(self):
        return np.arcsin(self.operand.value)

    def _derivative(self, x):
        return float(self.operand._derivative(x) / np.sqrt(1 - self.operand.value**2))


class _Acos(_UnaryOp):
    """The arccos function."""

    @property
    def value(self):
        return np.arccos(self.operand.value)

    def _derivative(self, x):
        return float(-self.operand._derivative(x) / np.sqrt(1 - self.operand.value**2))


class _Atan(_UnaryOp):
    """The arctan function."""

    @property
    def value(self):
        return np.arctan(self.operand.value)

    def _derivative(self, x):
        return float(self.operand._derivative(x) / (self.operand.value**2 + 1))


class _Atan2(_BinaryOp):
    """The arctan2 function."""

    @property
    def value(self):
        return np.arctan2(self.left.value, self.right.value)

    def _derivative(self, x):
        return float(
            (
                self.right.value * self.left._derivative(x)
                - self.left.value * self.right._derivative(x)
            )
            / (self.left.value**2 + self.right.value**2)
        )

    @property
    def unit(self) -> Unit:
        return Unit({})


class _Sinh(_UnaryOp):
    """The sinh function."""

    @property
    def value(self):
        return np.sinh(self.operand.value)

    def _derivative(self, x):
        return float(np.cosh(self.operand.value) * self.operand._derivative(x))


class _Cosh(_UnaryOp):
    """The cosh function."""

    @property
    def value(self):
        return np.cosh(self.operand.value)

    def _derivative(self, x):
        return float(np.sinh(self.operand.value) * self.operand._derivative(x))


class _Tanh(_UnaryOp):
    """The tanh function."""

    @property
    def value(self):
        return np.tanh(self.operand.value)

    def _derivative(self, x):
        return float(self.operand._derivative(x) / np.cosh(self.operand.value) ** 2)


class _Asinh(_UnaryOp):
    """The asinh function."""

    @property
    def value(self):
        return np.arcsinh(self.operand.value)

    def _derivative(self, x):
        return float(self.operand._derivative(x) / np.sqrt(self.operand.value**2 + 1))


class _Acosh(_UnaryOp):
    """The arccosh function."""

    @property
    def value(self):
        return np.arccosh(self.operand.value)

    def _derivative(self, x):
        return float(self.operand._derivative(x) / np.sqrt(self.operand.value**2 - 1))


class _Atanh(_UnaryOp):
    """The atanh function."""

    @property
    def value(self):
        return np.arctanh(self.operand.value)

    def _derivative(self, x):
        return float(self.operand._derivative(x) / (1 - self.operand.value**2))


class _Exp(_UnaryOp):
    """The exponential function."""

    @property
    def value(self):
        return np.exp(self.operand.value)

    def _derivative(self, x):
        return float(np.exp(self.operand.value) * self.operand._derivative(x))


class _Log2(_UnaryOp):
    """The base-2 log function."""

    @property
    def value(self):
        return np.log2(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / (np.log(2) * self.operand.value)


class _Log10(_UnaryOp):
    """The base-10 log function."""

    @property
    def value(self):
        return np.log10(self.operand.value)

    def _derivative(self, x):
        return self.operand._derivative(x) / (np.log(10) * self.operand.value)


class _Ln(_UnaryOp):
    """The base-e log function."""

    @property
    def value(self):
        return np.log(self.operand.value)

    def _derivative(self, x):
        return float(self.operand._derivative(x) / self.operand.value)


def _collect_measurements(formula: Formula) -> set[Measurement]:
    """Return the set of measurements that the formula is composed of."""

    if isinstance(formula, Measurement):
        return {formula}

    if isinstance(formula, _Operation):
        return set.union(*[_collect_measurements(op) for op in formula.operands])

    return set()


def _covariance_terms(formula: Formula, sources: set[Measurement]) -> Iterator[float]:
    """Finds the contributing covariance terms for the derivative method."""

    for var1, var2 in itertools.combinations(sources, 2):
        corr = correlation(var1, var2)
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
            # pylint: disable=protected-access
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
