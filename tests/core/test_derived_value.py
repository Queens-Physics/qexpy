"""Tests for ExperimentalValue arithmetic"""

import numpy as np
import pytest

import qexpy as q
from qexpy.core.formula import (
    _Sin,
    _Cos,
    _Tan,
    _Asin,
    _Acos,
    _Atan,
    _Sinh,
    _Cosh,
    _Tanh,
    _Acosh,
    _Asinh,
    _Sqrt,
    _Atanh,
    _Atan2,
    _Exp,
    _Log2,
    _Log10,
    _Ln,
    _NegativeOp,
    _Add,
    _Subtract,
    _Divide,
    _Power,
)


class TestOperations:
    """Tests the operator overloads of the ExperimentalValue class"""

    def test_abs(self):
        """Tests taking the absolute value of a value"""

        m = q.Measurement(1.23, 0.02, name="x", unit="kg*m")
        res = abs(m)
        assert isinstance(res, q.core.Measurement)
        assert res is not m
        assert res.value == 1.23
        assert res.error == 0.02
        assert res.unit == {"kg": 1, "m": 1}
        assert res.name == "x"
        assert res._id != m._id

        m = q.Measurement(-1.23, 0.02, name="x", unit="kg*m")
        res = abs(m)
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == 1.23
        assert res.error == 0.02
        assert res.unit == {"kg": 1, "m": 1}
        assert res.name == ""

        res2 = abs(res)
        assert isinstance(res2, q.core.DerivedValue)
        assert res2 is not res
        assert res2.value == 1.23
        assert res2.error == 0.02
        assert res2.unit == {"kg": 1, "m": 1}

        val = q.Measurement(1.23, 0.02, unit="kg*m") - q.Measurement(4.56, 0.03, unit="kg*m")
        res3 = abs(val)
        assert isinstance(res3, q.core.DerivedValue)
        assert res3.value == np.abs(val.value)
        assert res3.error == val.error
        assert res3.unit == {"kg": 1, "m": 1}

        m = q.Measurement([4.9, 5, 5.1], name="x", unit="kg*m/s^2")
        res4 = abs(m)
        assert isinstance(res4, q.core.RepeatedMeasurement)
        assert res4 is not m
        assert np.equal(res4._data, m._data).all()
        assert res4._unit == m._unit
        assert res4.value == m.value
        assert res4.error == m.error
        assert res4._id != m._id

    def test_add(self):
        """Tests adding two values"""

        m1 = q.Measurement(1.23, 0.02, unit="kg*m/s^2")
        m2 = q.Measurement(4.56, 0.03)
        res = m1 + m2
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 + 4.56)
        assert res.error == pytest.approx(np.sqrt(0.02**2 + 0.03**2))
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

        res = m2 + 1.23
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 + 4.56)
        assert res.error == 0.03
        assert res.unit == {}

        res = 4.56 + m1
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 + 4.56)
        assert res.error == 0.02
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

    def test_sub(self):
        """Tests subtracting two values"""

        m1 = q.Measurement(1.23, 0.02, unit="kg*m/s^2")
        m2 = q.Measurement(4.56, 0.03)

        res = m2 - m1
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(4.56 - 1.23)
        assert res.error == pytest.approx(np.sqrt(0.02**2 + 0.03**2))
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

        res = m2 - 1.23
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(4.56 - 1.23)
        assert res.error == 0.03
        assert res.unit == {}

        res = 4.56 - m1
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(4.56 - 1.23)
        assert res.error == 0.02
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

    def test_mul(self):
        """Tests multiplying two values"""

        m1 = q.Measurement(1.23, 0.02, unit="m/s^2")
        m2 = q.Measurement(4.56, 0.03, unit="kg")

        res = m1 * m2
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 * 4.56)
        assert res.error == pytest.approx(np.sqrt((4.56 * 0.02) ** 2 + (1.23 * 0.03) ** 2))
        assert res.unit == {"kg": 1, "m": 1, "s": -2}

        res = m1 * 4.56
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 * 4.56)
        assert res.error == 0.02 * 4.56
        assert res.unit == {"m": 1, "s": -2}

        res = 1.23 * m2
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 * 4.56)
        assert res.error == 0.03 * 1.23
        assert res.unit == {"kg": 1}

    def test_div(self):
        """Tests dividing two values"""

        m1 = q.Measurement(1.23, 0.02, unit="m/s")
        m2 = q.Measurement(4.56, 0.03, unit="s")

        res = m1 / m2
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 / 4.56)
        assert res.error == pytest.approx(
            np.sqrt((0.02 / 4.56) ** 2 + (1.23 * 0.03 / 4.56**2) ** 2)
        )
        assert res.unit == {"m": 1, "s": -2}

        res = m1 / 4.56
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 / 4.56)
        assert res.error == 0.02 / 4.56
        assert res.unit == {"m": 1, "s": -1}

        res = 1.23 / m2
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.23 / 4.56)
        assert res.error == 1.23 * 0.03 / 4.56**2
        assert res.unit == {"s": -1}

    def test_pow(self):
        """Tests the power operator"""

        m1 = q.Measurement(1.2, 0.02, unit="kg*m/s^2")
        m2 = q.Measurement(4.5, 0.03, unit="A*s")

        res = m1**m2
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.2**4.5)
        assert res.error == pytest.approx(
            np.sqrt((4.5 * 1.2 ** (4.5 - 1) * 0.02) ** 2 + (1.2**4.5 * np.log(1.2) * 0.03) ** 2)
        )
        assert res.unit == {"kg": 4.5, "m": 4.5, "s": -9}

        res = m1**4.5
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.2**4.5)
        assert res.error == pytest.approx(4.5 * 1.2 ** (4.5 - 1) * 0.02)
        assert res.unit == {"kg": 4.5, "m": 4.5, "s": -9}

        res = 1.2**m2
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == pytest.approx(1.2**4.5)
        assert res.error == pytest.approx(1.2**4.5 * np.log(1.2) * 0.03)
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
        """Tests compatibility with numpy ufuncs"""

        m1 = q.Measurement(val, 0.02, unit="kg*m/s^2")
        res = func(m1)
        assert isinstance(res, q.core.DerivedValue)
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
        """Tests compatibility with binary ufuncs"""

        m1 = q.Measurement(val1, 0.02)
        m2 = q.Measurement(val2, 0.03)

        res = func(m1, m2)
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == func(val1, val2)
        assert isinstance(res._formula, formula_type)

        res = func(val1, m2)
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == func(val1, val2)
        assert isinstance(res._formula, formula_type)

        res = func(m1, val2)
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == func(val1, val2)
        assert isinstance(res._formula, formula_type)


# pylint: disable=too-few-public-methods
class TestErrorPropagation:
    """Tests that the derived values have correctly propagated errors"""

    @pytest.mark.parametrize(
        "op_func",
        [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
        ],
    )
    def test_correlated_measurements(self, op_func):
        """Tests that error propagation works correctly for correlated measurements"""

        arr1 = np.array([399.3, 404.6, 394.6, 396.3, 399.6, 404.9, 387.4, 404.9, 398.2, 407.2])
        arr2 = np.array([193.2, 205.1, 192.6, 194.2, 196.6, 201.0, 184.7, 215.2, 203.6, 207.8])
        arr_expected = op_func(arr1, arr2)

        m1 = q.Measurement(arr1)
        m2 = q.Measurement(arr2)
        m_expected = q.Measurement(arr_expected)

        m1.set_covariance(m2)
        m = op_func(m1, m2)
        assert m.value == pytest.approx(m_expected.value, rel=0.02)
        assert m.error == pytest.approx(m_expected.error, rel=0.02)
