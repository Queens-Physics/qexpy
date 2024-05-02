"""Tests for ExperimentalValue arithmetic"""

# pylint: disable=no-value-for-parameter

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


class TestErrorPropagation:
    """Tests that the derived values have correctly propagated errors"""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleans up global configurations"""
        yield
        q.reset_option()
        q.clear_unit_definitions()

    def test_error_method(self):
        """Tests updating the error method"""

        m1 = q.Measurement(1.23, 0.02)
        m2 = q.Measurement(4.56, 0.03)
        res = m1 + m2

        q.options.error.mc.sample_size = 100

        assert isinstance(res, q.core.DerivedValue)
        assert res.error_method == "derivative"
        assert res.value == pytest.approx(5.79)

        q.options.error.method = "monte-carlo"
        assert res.error_method == "monte-carlo"
        assert res.value == res.mc.samples.mean()

        q.options.error.method = "derivative"
        assert res.error_method == "derivative"
        assert res.value == pytest.approx(5.79)

        res.error_method = "monte-carlo"
        assert res.error_method == "monte-carlo"
        assert res.value == res.mc.samples.mean()

        res.error_method = "auto"
        assert res.error_method == "derivative"
        assert res.value == pytest.approx(5.79)

    def test_invalid_error_method(self):
        """Tests setting an invalid error method"""

        m1 = q.Measurement(1.23, 0.02)
        m2 = q.Measurement(4.56, 0.03)
        res = m1 + m2

        with pytest.raises(ValueError, match="The error method can only be"):
            res.error_method = "invalid"

    @pytest.mark.parametrize(
        "op_func",
        [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
        ],
    )
    @pytest.mark.parametrize("error_method", ["derivative", "monte-carlo"])
    def test_correlated_measurements(self, op_func, error_method):
        """Tests that error propagation works correctly for correlated measurements"""

        arr1 = np.array([399.3, 404.6, 394.6, 396.3, 399.6, 404.9, 387.4, 404.9, 398.2, 407.2])
        arr2 = np.array([193.2, 205.1, 192.6, 194.2, 196.6, 201.0, 184.7, 215.2, 203.6, 207.8])
        arr_expected = op_func(arr1, arr2)

        m1 = q.Measurement(arr1)
        m2 = q.Measurement(arr2)
        m_expected = q.Measurement(arr_expected)

        m1.set_covariance(m2)
        m = op_func(m1, m2)
        m.error_method = error_method
        assert m.value == pytest.approx(m_expected.value, rel=0.02)
        assert m.error == pytest.approx(m_expected.error, rel=0.05)

    @pytest.mark.parametrize(
        "op_func",
        [
            lambda x, y, z: x + y + z,
            lambda x, y, z: x - y - z,
            lambda x, y, z: x * y - z,
            lambda x, y, z: x - y / z,
        ],
    )
    @pytest.mark.parametrize("error_method", ["derivative", "monte-carlo"])
    def test_multiple_correlated_measurements(self, op_func, error_method):
        """Tests that error propagation works for multiple correlated measurements"""

        arr1 = np.array([399.3, 404.6, 394.6, 396.3, 399.6, 404.9, 387.4, 404.9, 398.2, 407.2])
        arr2 = np.array([193.2, 205.1, 192.6, 194.2, 196.6, 201.0, 184.7, 215.2, 203.6, 207.8])
        arr3 = np.array([93.1, 105.1, 92.7, 94.2, 96.6, 101.0, 84.6, 115.3, 103.6, 107.7])
        arr_expected = op_func(arr1, arr2, arr3)

        m1 = q.Measurement(arr1)
        m2 = q.Measurement(arr2)
        m3 = q.Measurement(arr3)
        m_expected = q.Measurement(arr_expected)

        m1.set_covariance(m2)
        m1.set_covariance(m3)
        m2.set_covariance(m3)
        m = op_func(m1, m2, m3)
        m.error_method = error_method
        assert m.value == pytest.approx(m_expected.value, rel=0.02)
        assert m.error == pytest.approx(m_expected.error, rel=0.05)

    @pytest.mark.parametrize("error_method", ["derivative", "monte-carlo"])
    def test_composite_formula(self, error_method):
        """Integration test for error propagation with composite formula"""

        q.define_unit("F", "C^2/(N*m)")
        q.define_unit("N", "kg*m/s^2")

        q1 = q.Measurement(1.23e-6, relative_error=0.01, unit="C")
        q2 = q.Measurement(2.34e-5, relative_error=0.01, unit="C")
        r = q.Measurement(0.12, relative_error=0.01, unit="m")

        force = 1 / (4 * q.pi * q.eps0) * q1 * q2 / r**2
        force.error_method = error_method

        eps0 = 8.8541878128e-12
        expected_value = 1 / (4 * np.pi * eps0) * 1.23e-6 * 2.34e-5 / 0.12**2
        expected_error = np.sqrt(
            (0.01 * 1.23e-6 / (4 * np.pi * eps0) * 2.34e-5 / 0.12**2) ** 2
            + (0.01 * 2.34e-5 / (4 * np.pi * eps0) * 1.23e-6 / 0.12**2) ** 2
            + (0.01 * 0.12 * 2 / (4 * np.pi * eps0) * 1.23e-6 * 2.34e-5 / 0.12**3) ** 2
            + (0.0000000013e-12 / (4 * np.pi * eps0**2) * 1.23e-6 * 2.34e-5 / 0.12**2) ** 2
        )

        assert isinstance(force, q.core.DerivedValue)
        assert force.unit == {"kg": 1, "m": 1, "s": -2}
        assert str(force.unit) == "N"

        if force.error_method == "derivative":
            assert force.value == pytest.approx(expected_value)
            assert force.error == pytest.approx(expected_error)
        else:
            assert force.value == pytest.approx(expected_value, rel=0.001)
            assert force.error == pytest.approx(expected_error, rel=0.005)
