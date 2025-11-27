"""Tests for the internal Formula module."""

import numpy as np
import pytest

from qexpy.core import Constant, DerivedValue, Measurement
from qexpy.core.formula import (
    Formula,
    _Acos,
    _Acosh,
    _Add,
    _Asin,
    _Asinh,
    _Atan,
    _Atan2,
    _Atanh,
    _Cos,
    _Cosh,
    _Divide,
    _Exp,
    _Ln,
    _Log2,
    _Log10,
    _Multiply,
    _NegativeOp,
    _Operation,
    _Power,
    _Sin,
    _Sinh,
    _Sqrt,
    _Subtract,
    _Tan,
    _Tanh,
    to_formula,
)
from qexpy.units import Unit


@pytest.mark.parametrize(
    "original, expected_type",
    [
        (1, Constant),
        (1.2, Constant),
        (Measurement(5, 0.1), Measurement),
        (DerivedValue(_NegativeOp(Measurement(5, 0.1))), Formula),
    ],
)
def test_to_formula(original, expected_type):
    """Tests wrapping an object in a Formula."""
    assert isinstance(to_formula(original), expected_type)


def test_wraps_fail():
    """Tests wrapping an unsupported object in a Formula."""
    with pytest.raises(TypeError, match="not supported in operations."):
        to_formula("abc")


class TestUnitaryOps:
    """Tests for unitary operations."""

    def test_negative_op(self):
        """Tests the _NegativeOp class."""

        m = Measurement(5, 0.1, unit="kg*m/s^2")
        op = _NegativeOp(m)
        assert op.value == -5
        assert op.error == 0.1
        assert op._derivative(m) == -1
        assert op.unit == {"kg": 1, "m": 1, "s": -2}

        op2 = _NegativeOp(op)
        assert op2.value == 5
        assert op2.error == 0.1
        assert op2._derivative(m) == 1
        assert op2.unit == {"kg": 1, "m": 1, "s": -2}

    def test_sqrt(self):
        """Tests the _Sqrt class."""

        m = Measurement(5, 0.1, unit="kg*m/s^2")
        op = _Sqrt(m)
        assert op.value == np.sqrt(5)
        assert op.error == 0.1 / (2 * np.sqrt(5))
        assert op._derivative(m) == 1 / (2 * np.sqrt(5))
        assert op.unit == {"kg": 1 / 2, "m": 1 / 2, "s": -1}

        op2 = _Sqrt(op)
        assert np.isclose(op2.value, 5 ** (1 / 4))
        assert np.isclose(op2.error, 0.1 / (4 * 5 ** (3 / 4)))
        assert np.isclose(op2._derivative(m), 1 / (4 * 5 ** (3 / 4)))
        assert op2.unit == {"kg": 1 / 4, "m": 1 / 4, "s": -1 / 2}

    @pytest.mark.parametrize("theta", [0, np.pi / 3, np.pi / 2, np.pi, 2 * np.pi - 0.1])
    def test_sin(self, theta):
        """Tests the _Sin class."""

        m = Measurement(theta, 0.02, unit="kg*m/s^2")
        op = _Sin(m)
        assert op.value == np.sin(theta)
        assert op.error == np.abs(0.02 * np.cos(theta))
        assert op._derivative(m) == np.cos(theta)
        assert op.unit == {}

        op2 = _Sin(op)
        assert np.isclose(op2.value, np.sin(np.sin(theta)))
        assert np.isclose(
            op2.error, np.abs(0.02 * np.cos(np.sin(theta)) * np.cos(theta))
        )
        assert np.isclose(op2._derivative(m), np.cos(np.sin(theta)) * np.cos(theta))
        assert op2.unit == {}

    @pytest.mark.parametrize("theta", [0, np.pi / 3, np.pi / 2, np.pi, 2 * np.pi - 0.1])
    def test_cos(self, theta):
        """Tests the _Cos class."""

        m = Measurement(theta, 0.02, unit="kg*m/s^2")
        op = _Cos(m)
        assert op.value == np.cos(theta)
        assert op.error == np.abs(0.02 * np.sin(theta))
        assert op._derivative(m) == -np.sin(theta)
        assert op.unit == {}

        op2 = _Cos(op)
        assert np.isclose(op2.value, np.cos(np.cos(theta)))
        assert np.isclose(
            op2.error, np.abs(0.02 * np.sin(np.cos(theta)) * np.sin(theta))
        )
        assert op2._derivative(m) == np.sin(np.cos(theta)) * np.sin(theta)
        assert op2.unit == {}

    @pytest.mark.parametrize("theta", [0, np.pi / 3, np.pi, 2 * np.pi - 0.1])
    def test_tan(self, theta):
        """Tests the _Tan class."""

        m = Measurement(theta, 0.02, unit="kg*m/s^2")
        op = _Tan(m)
        assert op.value == np.tan(theta)
        assert op.error == np.abs(0.02 / (np.cos(theta) ** 2))
        assert op._derivative(m) == 1 / (np.cos(theta) ** 2)
        assert op.unit == {}

        op2 = _Tan(op)
        assert np.isclose(op2.value, np.tan(np.tan(theta)))
        assert np.isclose(
            op2.error, np.abs(0.02 / np.cos(np.tan(theta)) ** 2 / np.cos(theta) ** 2)
        )
        assert op2._derivative(m) == 1 / np.cos(np.tan(theta)) ** 2 / np.cos(theta) ** 2
        assert op2.unit == {}

    @pytest.mark.parametrize("theta", [-np.pi / 2 + 0.1, 0, np.pi / 3, np.pi / 2 - 0.1])
    def test_asin(self, theta):
        """Tests the _Asin class."""

        m = Measurement(np.sin(theta), 0.02, unit="kg*m/s^2")
        op = _Asin(m)
        assert np.isclose(op.value, theta)
        assert np.isclose(op.error, 0.02 / np.sqrt(1 - np.sin(theta) ** 2))
        assert np.isclose(op._derivative(m), 1 / np.sqrt(1 - np.sin(theta) ** 2))
        assert op.unit == {}

    @pytest.mark.parametrize("theta", [0.1, np.pi / 3, np.pi / 2 + 0.1, np.pi - 0.1])
    def test_acos(self, theta):
        """Tests the _Acos class."""

        m = Measurement(np.cos(theta), 0.02, unit="kg*m/s^2")
        op = _Acos(m)
        assert np.isclose(op.value, theta)
        assert np.isclose(op.error, np.abs(0.02 / np.sqrt(1 - np.cos(theta) ** 2)))
        assert np.isclose(op._derivative(m), -1 / np.sqrt(1 - np.cos(theta) ** 2))
        assert op.unit == {}

    @pytest.mark.parametrize("theta", [-np.pi / 2 + 0.1, 0, np.pi / 3, np.pi / 2 - 0.1])
    def test_atan(self, theta):
        """Tests the _Atan class."""

        m = Measurement(np.tan(theta), 0.02, unit="kg*m/s^2")
        op = _Atan(m)
        assert np.isclose(op.value, theta)
        assert np.isclose(op.error, 0.02 / (1 + np.tan(theta) ** 2))
        assert np.isclose(op._derivative(m), 1 / (1 + np.tan(theta) ** 2))
        assert op.unit == {}

    @pytest.mark.parametrize("x", [-1, 0, 1])
    def test_sinh(self, x):
        """Tests the _Sinh and _Asinh class."""

        m = Measurement(x, 0.02, unit="kg*m/s^2")
        op = _Sinh(m)
        assert np.isclose(op.value, np.sinh(x))
        assert np.isclose(op.error, 0.02 * np.cosh(x))
        assert op._derivative(m) == np.cosh(x)
        assert op.unit == {}

        op2 = _Asinh(op)
        assert np.isclose(op2.value, x)
        assert np.isclose(op2.error, 0.02)
        assert op2._derivative(m) == np.cosh(x) / np.sqrt(np.sinh(x) ** 2 + 1)
        assert op2.unit == {}

    @pytest.mark.parametrize("x", [-1, 0.1, 1])
    def test_cosh(self, x):
        """Tests the _Cosh ans _Acosh class."""

        m = Measurement(x, 0.02, unit="kg*m/s^2")
        op = _Cosh(m)
        assert np.isclose(op.value, np.cosh(x))
        assert np.isclose(op.error, np.abs(0.02 * np.sinh(x)))
        assert op._derivative(m) == np.sinh(x)
        assert op.unit == {}

        op2 = _Acosh(op)
        assert np.isclose(op2.value, np.abs(x))
        assert np.isclose(op2.error, 0.02)
        assert op2._derivative(m) == np.sinh(x) / np.sqrt(np.cosh(x) ** 2 - 1)
        assert op2.unit == {}

    @pytest.mark.parametrize("x", [-1, 0, 1])
    def test_tanh(self, x):
        """Tests the _Tanh and _Atanh class."""

        m = Measurement(x, 0.02, unit="kg*m/s^2")
        op = _Tanh(m)
        assert np.isclose(op.value, np.tanh(x))
        assert np.isclose(op.error, np.abs(0.02 / (np.cosh(x) ** 2)))
        assert op._derivative(m) == 1 / (np.cosh(x) ** 2)
        assert op.unit == {}

        op2 = _Atanh(op)
        assert np.isclose(op2.value, x)
        assert np.isclose(op2.error, 0.02)
        assert np.isclose(
            op2._derivative(m), 1 / (1 - np.tanh(x) ** 2) / (np.cosh(x) ** 2)
        )
        assert op2.unit == {}

    def test_exp(self):
        """Tests the _Exp class."""

        m = Measurement(5, 0.02, unit="kg*m/s^2")
        op = _Exp(m)
        assert np.isclose(op.value, np.exp(5))
        assert np.isclose(op.error, 0.02 * np.exp(5))
        assert op._derivative(m) == np.exp(5)
        assert op.unit == {}

        op2 = _Exp(op)
        assert np.isclose(op2.value, np.exp(np.exp(5)))
        assert np.isclose(op2.error, 0.02 * np.exp(np.exp(5)) * np.exp(5))
        assert op2._derivative(m) == np.exp(np.exp(5)) * np.exp(5)
        assert op2.unit == {}

    def test_log(self):
        """Tests the logarithm functions."""

        m = Measurement(5, 0.02, unit="kg*m/s^2")
        op = _Log2(m)
        assert np.isclose(op.value, np.log2(5))
        assert np.isclose(op.error, 0.02 / 5 / np.log(2))
        assert op._derivative(m) == 1 / np.log(2) / 5
        assert op.unit == {}

        op = _Log10(m)
        assert np.isclose(op.value, np.log10(5))
        assert np.isclose(op.error, 0.02 / 5 / np.log(10))
        assert op._derivative(m) == 1 / np.log(10) / 5
        assert op.unit == {}

        op = _Ln(m)
        assert np.isclose(op.value, np.log(5))
        assert np.isclose(op.error, 0.02 / 5)
        assert op._derivative(m) == 1 / 5
        assert op.unit == {}


class _MockFormula(_Operation):
    """A mock formula with a derivative."""

    def __init__(self, m, d):
        self.m = m
        self.d = d
        super().__init__()

    @property
    def value(self) -> float:
        """The value of the formula."""
        return self.m.value * self.d

    @property
    def unit(self) -> Unit:
        """The unit of the formula."""
        return Unit({})  # pragma: no cover

    def _derivative(self, x: Formula) -> float:
        return self.d if x is self.m else 0

    @property
    def operands(self) -> tuple[Formula]:
        """The operands of the formula."""
        return (self.m,)


class TestBinaryOps:
    """Tests binary operations."""

    def test_add(self):
        """Tests the _Add class."""

        m1 = Measurement(1.23, 0.02, unit="kg*m/s^2")
        m2 = Measurement(4.56, 0.03, unit="")
        op = _Add(m1, m2)
        assert np.isclose(op.value, 1.23 + 4.56)
        assert np.isclose(op.error, np.sqrt(0.02**2 + 0.03**2))
        assert op._derivative(m1) == 1
        assert op._derivative(m2) == 1
        assert op.unit == {"kg": 1, "m": 1, "s": -2}

        f1 = _MockFormula(m1, 0.12)
        f2 = _MockFormula(m2, 0.23)
        op = _Add(f1, f2)
        assert np.isclose(op.value, 1.23 * 0.12 + 4.56 * 0.23)
        assert np.isclose(op.error, np.sqrt((0.02 * 0.12) ** 2 + (0.03 * 0.23) ** 2))
        assert op._derivative(m1) == 0.12
        assert op._derivative(m2) == 0.23

    def test_sub(self):
        """Tests the _Sub class."""

        m1 = Measurement(1.23, 0.02, unit="kg*m/s^2")
        m2 = Measurement(4.56, 0.03, unit="")
        op = _Subtract(m2, m1)
        assert np.isclose(op.value, 4.56 - 1.23)
        assert np.isclose(op.error, np.sqrt(0.02**2 + 0.03**2))
        assert op._derivative(m1) == -1
        assert op._derivative(m2) == 1
        assert op.unit == {"kg": 1, "m": 1, "s": -2}

        f1 = _MockFormula(m1, 0.12)
        f2 = _MockFormula(m2, 0.23)
        op = _Subtract(f2, f1)
        assert np.isclose(op.value, 4.56 * 0.23 - 1.23 * 0.12)
        assert np.isclose(op.error, np.sqrt((0.02 * 0.12) ** 2 + (0.03 * 0.23) ** 2))
        assert op._derivative(m1) == -0.12
        assert op._derivative(m2) == 0.23

    def test_multiply(self):
        """Tests the _Multiply class."""

        m1 = Measurement(1.23, 0.02, unit="m/s^2")
        m2 = Measurement(4.56, 0.03, unit="kg")
        op = _Multiply(m1, m2)
        assert np.isclose(op.value, 1.23 * 4.56)
        assert np.isclose(op.error, np.sqrt((0.02 * 4.56) ** 2 + (0.03 * 1.23) ** 2))
        assert op._derivative(m1) == 4.56
        assert op._derivative(m2) == 1.23
        assert op.unit == {"kg": 1, "m": 1, "s": -2}

        f1 = _MockFormula(m1, 0.12)
        f2 = _MockFormula(m2, 0.23)
        op = _Multiply(f1, f2)
        assert np.isclose(op.value, 1.23 * 0.12 * 4.56 * 0.23)
        assert op._derivative(m1) == 0.12 * 0.23 * 4.56
        assert op._derivative(m2) == 0.23 * 0.12 * 1.23
        assert np.isclose(
            op.error,
            np.sqrt(
                (0.02 * 0.12 * 0.23 * 4.56) ** 2 + (0.03 * 0.12 * 0.23 * 1.23) ** 2
            ),
        )

    def test_divide(self):
        """Test the _Division class."""

        m1 = Measurement(1.23, 0.02, unit="m/s")
        m2 = Measurement(4.56, 0.03, unit="s")
        op = _Divide(m1, m2)
        assert np.isclose(op.value, 1.23 / 4.56)
        assert op._derivative(m1) == 1 / 4.56
        assert op._derivative(m2) == -1.23 / 4.56**2
        assert np.isclose(
            op.error, np.sqrt((0.02 / 4.56) ** 2 + (0.03 * 1.23 / 4.56**2) ** 2)
        )
        assert op.unit == {"m": 1, "s": -2}

        f1 = _MockFormula(m1, 0.12)
        f2 = _MockFormula(m2, 0.23)
        op = _Divide(f1, f2)
        assert np.isclose(op.value, 1.23 * 0.12 / 4.56 / 0.23)
        assert np.isclose(op._derivative(m1), 0.12 / (0.23 * 4.56))
        assert np.isclose(op._derivative(m2), -1.23 * 0.12 / 0.23 / 4.56**2)
        assert np.isclose(
            op.error,
            np.sqrt(
                (0.02 * 0.12 / (4.56 * 0.23)) ** 2
                + (0.03 * 0.12 / 0.23 * 1.23 / 4.56**2) ** 2
            ),
        )

    def test_power(self):
        """Tests the _Pow operation."""

        m1 = Measurement(1.2, 0.02, unit="kg*m/s^2")
        m2 = Measurement(4.5, 0.03, unit="A*s")
        op = _Power(m1, m2)
        assert np.isclose(op.value, 1.2**4.5)
        assert np.isclose(op._derivative(m1), 4.5 * 1.2 ** (4.5 - 1))
        assert np.isclose(op._derivative(m2), 1.2**4.5 * np.log(1.2))
        assert np.isclose(
            op.error,
            np.sqrt(
                (4.5 * 1.2 ** (4.5 - 1) * 0.02) ** 2
                + (1.2**4.5 * np.log(1.2) * 0.03) ** 2
            ),
        )
        assert op.unit == {"kg": 4.5, "m": 4.5, "s": -9}

        f1 = _MockFormula(m1, 0.12)
        f2 = _MockFormula(m2, 0.23)
        op2 = _Power(f1, f2)
        assert np.isclose(op2.value, (0.12 * 1.2) ** (0.23 * 4.5))
        assert np.isclose(
            op2._derivative(m1), 0.23 * 4.5 * (0.12 * 1.2) ** (0.23 * 4.5 - 1) * 0.12
        )
        assert np.isclose(
            op2._derivative(m2),
            (0.12 * 1.2) ** (0.23 * 4.5) * np.log(0.12 * 1.2) * 0.23,
        )
        assert np.isclose(
            op2.error,
            np.sqrt(
                (0.23 * 4.5 * (0.12 * 1.2) ** (0.23 * 4.5 - 1) * 0.12 * 0.02) ** 2
                + ((0.12 * 1.2) ** (0.23 * 4.5) * np.log(0.12 * 1.2) * 0.23 * 0.03) ** 2
            ),
        )

    def test_power_negative_base(self):
        """Tests that the _Pow operation works with a negative base."""

        m1 = Measurement(-1.2, 0.02, unit="kg*m/s^2")
        m2 = Constant(4)
        op = _Power(m1, m2)
        assert np.isclose(op.value, (-1.2) ** 4)
        assert np.isclose(op._derivative(m1), 4 * (-1.2) ** (4 - 1))
        assert np.isclose(op.error, np.abs(4 * (-1.2) ** (4 - 1) * 0.02))

    @pytest.mark.parametrize(
        "a1, a2, expected",
        [
            (0, 1, 0),
            (1, 1, np.pi / 4),
            (1, -1, 3 * np.pi / 4),
            (-1, 1, -np.pi / 4),
            (-1, -1, -3 * np.pi / 4),
        ],
    )
    def test_atan2(self, a1, a2, expected):
        """Tests the _Atan2 class."""

        m1 = Measurement(a1, 0.02, unit="kg*m/s^2")
        m2 = Measurement(a2, 0.03, unit="kg*m/s^2")
        op = _Atan2(m1, m2)
        assert np.isclose(op.value, expected)
        assert np.isclose(
            op.error,
            np.sqrt(
                (0.02 * a2 / (a1**2 + a2**2)) ** 2 + (0.03 * a1 / (a1**2 + a2**2)) ** 2
            ),
        )
        assert np.isclose(op._derivative(m1), a2 / (a1**2 + a2**2))
        assert np.isclose(op._derivative(m2), -a1 / (a1**2 + a2**2))
        assert op.unit == {}
