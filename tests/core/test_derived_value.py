"""Tests for the DerivedValue class."""

import numpy as np
import pytest

import qexpy as q
from qexpy.core import DerivedValue
from qexpy.core.formula import (
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
    _NegativeOp,
    _Power,
    _Sin,
    _Sinh,
    _Sqrt,
    _Subtract,
    _Tan,
    _Tanh,
)


class TestOperations:
    """Tests the operator overloads of the ExperimentalValue."""

    @pytest.mark.parametrize(
        "v",
        [
            q.Measurement(1.23, 0.01, name="x", unit="kg*m"),
            q.Measurement(-1.23, 0.01, name="x", unit="kg*m"),
            q.Measurement([4.9, 5, 5.1], name="x", unit="kg*m/s^2"),
            q.Measurement(1.23, 0.02, unit="kg*m") - q.Measurement(4.56, 0.03),
        ],
    )
    def test_abs(self, v):
        """Tests taking the absolute value of a value."""

        res = abs(v)
        assert res is not v
        assert res.value == np.abs(v.value)
        assert res.error == v.error
        assert res.unit == v.unit

    def test_add(self):
        """Tests adding two values."""

        m1 = q.Measurement(1.23, 0.02, unit="kg*m/s^2")
        m2 = q.Measurement(4.56, 0.03)
        res = m1 + m2
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 + 4.56)
        assert np.isclose(res.error, np.sqrt(0.02**2 + 0.03**2))
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

        res = m2 + 1.23
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 + 4.56)
        assert res.error == 0.03
        assert res.unit == {}

        res = 4.56 + m1
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 + 4.56)
        assert res.error == 0.02
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

    def test_sub(self):
        """Tests subtracting two values."""

        m1 = q.Measurement(1.23, 0.02, unit="kg*m/s^2")
        m2 = q.Measurement(4.56, 0.03)

        res = m2 - m1
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 4.56 - 1.23)
        assert np.isclose(res.error, np.sqrt(0.02**2 + 0.03**2))
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

        res = m2 - 1.23
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 4.56 - 1.23)
        assert res.error == 0.03
        assert res.unit == {}

        res = 4.56 - m1
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 4.56 - 1.23)
        assert res.error == 0.02
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

    def test_mul(self):
        """Tests multiplying two values."""

        m1 = q.Measurement(1.23, 0.02, unit="m/s^2")
        m2 = q.Measurement(4.56, 0.03, unit="kg")

        res = m1 * m2
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 * 4.56)
        assert np.isclose(res.error, np.sqrt((4.56 * 0.02) ** 2 + (1.23 * 0.03) ** 2))
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

        res = m1 * 4.56
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 * 4.56)
        assert res.error == 0.02 * 4.56
        assert res.unit == {"m": 1, "s": -2}

        res = 1.23 * m2
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 * 4.56)
        assert res.error == 0.03 * 1.23
        assert res.unit == {"kg": 1}

    def test_div(self):
        """Tests dividing two values."""

        m1 = q.Measurement(1.23, 0.02, unit="m/s")
        m2 = q.Measurement(4.56, 0.03, unit="s")

        res = m1 / m2
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 / 4.56)
        assert np.isclose(
            res.error, np.sqrt((0.02 / 4.56) ** 2 + (1.23 * 0.03 / 4.56**2) ** 2)
        )
        assert res.unit == {"m": 1, "s": -2}

        res = m1 / 4.56
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 / 4.56)
        assert res.error == 0.02 / 4.56
        assert res.unit == {"m": 1, "s": -1}

        res = 1.23 / m2
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.23 / 4.56)
        assert res.error == 1.23 * 0.03 / 4.56**2
        assert res.unit == {"s": -1}

    def test_pow(self):
        """Tests the power operator."""

        m1 = q.Measurement(1.2, 0.02, unit="kg*m/s^2")
        m2 = q.Measurement(4.5, 0.03, unit="A*s")

        res = m1**m2
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.2**4.5)
        assert np.isclose(
            res.error,
            np.sqrt(
                (4.5 * 1.2 ** (4.5 - 1) * 0.02) ** 2
                + (1.2**4.5 * np.log(1.2) * 0.03) ** 2
            ),
        )
        assert res.unit == {"kg": 4.5, "m": 4.5, "s": -9}

        res = m1**4.5
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.2**4.5)
        assert np.isclose(res.error, 4.5 * 1.2 ** (4.5 - 1) * 0.02)
        assert res.unit == {"kg": 4.5, "m": 4.5, "s": -9}

        res = 1.2**m2
        assert isinstance(res, DerivedValue)
        assert np.isclose(res.value, 1.2**4.5)
        assert np.isclose(res.error, 1.2**4.5 * np.log(1.2) * 0.03)
        assert res.unit == {}

    @pytest.mark.parametrize(
        "func, val, formula_type",
        [
            (np.sqrt, 1.23, _Sqrt),
            (np.sin, np.pi / 3, _Sin),
            (np.cos, np.pi / 3, _Cos),
            (np.tan, np.pi / 3, _Tan),
            (np.arcsin, np.sin(np.pi / 3), _Asin),
            (np.arccos, np.cos(np.pi / 3), _Acos),
            (np.arctan, np.tan(np.pi / 3), _Atan),
            (np.sinh, 1.23, _Sinh),
            (np.cosh, 1.23, _Cosh),
            (np.tanh, 1.23, _Tanh),
            (np.arcsinh, 1.23, _Asinh),
            (np.arccosh, 1.23, _Acosh),
            (np.arctanh, 0.23, _Atanh),
            (np.exp, 1.23, _Exp),
            (np.log2, 1.23, _Log2),
            (np.log10, 1.23, _Log10),
            (np.log, 1.23, _Ln),
            (np.negative, 1.23, _NegativeOp),
        ],
    )
    def test_ufunc(self, func, val, formula_type):
        """Tests compatibility with numpy ufuncs."""

        m1 = q.Measurement(val, 0.02, unit="kg*m/s^2")
        res = func(m1)
        assert isinstance(res, DerivedValue)
        assert res.value == func(val)
        assert isinstance(res._formula, formula_type)

    @pytest.mark.parametrize(
        "func, val1, val2, formula_type",
        [
            (np.add, 1.23, 4.56, _Add),
            (np.arctan2, 1.23, 4.56, _Atan2),
            (np.subtract, 1.23, 4.56, _Subtract),
            (np.divide, 1.23, 4.56, _Divide),
            (np.power, 1.23, 4.56, _Power),
        ],
    )
    def test_binary_ufunc(self, func, val1, val2, formula_type):
        """Tests compatibility with binary ufuncs."""

        m1 = q.Measurement(val1, 0.02)
        m2 = q.Measurement(val2, 0.03)

        res = func(m1, m2)
        assert isinstance(res, DerivedValue)
        assert res.value == func(val1, val2)
        assert isinstance(res._formula, formula_type)

        res = func(val1, m2)
        assert isinstance(res, DerivedValue)
        assert res.value == func(val1, val2)
        assert isinstance(res._formula, formula_type)

        res = func(m1, val2)
        assert isinstance(res, DerivedValue)
        assert res.value == func(val1, val2)
        assert isinstance(res._formula, formula_type)

    def test_composite_formula(self):
        """Tests a derived value constructed with a composite formula."""

        a = q.Measurement(5, 0.1)
        b = q.Measurement(20, 0.5)
        c = q.Measurement(8, 0.5)

        res = b / a + c
        assert res.value == 20 / 5 + 8
        assert np.isclose(
            res.error, np.sqrt((1 / 5 * 0.5) ** 2 + (20 / (5**2) * 0.1) ** 2 + 0.5**2)
        )

    def test_correlated_measurements(self):
        """Tests a derived value with correlated measurements."""

        m1 = q.Measurement([
            399.3,
            404.6,
            394.6,
            396.3,
            399.6,
            404.9,
            387.4,
            404.9,
            398.2,
            407.2,
        ])
        m2 = q.Measurement([
            193.2,
            205.1,
            192.6,
            194.2,
            196.6,
            201.0,
            184.7,
            215.2,
            203.6,
            207.8,
        ])
        m3 = np.array([
            399.3 + 193.2,
            404.6 + 205.1,
            394.6 + 192.6,
            396.3 + 194.2,
            399.6 + 196.6,
            404.9 + 201.0,
            387.4 + 184.7,
            404.9 + 215.2,
            398.2 + 203.6,
            407.2 + 207.8,
        ])
        q.set_correlation(m1, m2)
        res = m1 + m2
        assert np.isclose(res.value, np.mean(m3))
        assert np.isclose(res.error, np.std(m3, ddof=1) / np.sqrt(len(m3)))
