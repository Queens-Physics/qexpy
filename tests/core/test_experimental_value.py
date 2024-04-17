"""Unit tests for the ExperimentalValue class"""

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
from qexpy.utils import Unit


class TestAttributes:
    """Tests accessing the attributes of the ExperimentalValue class"""

    def test_name(self):
        """Tests the name attribute"""

        x = q.Measurement(1.23, 0.15, name="x")
        assert x.name == "x"

        x.name = "y"
        assert x.name == "y"

        with pytest.raises(TypeError):
            x.name = 1

    def test_unit(self):
        """Tests the unit attribute"""

        x = q.Measurement(1.23, 0.15, unit="kg*m/s^2")
        assert x.unit == Unit({"kg": 1, "m": 1, "s": -2})
        assert str(x.unit) == "kg⋅m/s^2"

        x.unit = "kg^2m^2s^-4"
        assert x.unit == Unit({"kg": 2, "m": 2, "s": -4})
        assert str(x.unit) == "kg^2⋅m^2/s^4"

        with pytest.raises(TypeError):
            x.unit = 1

    @pytest.mark.parametrize(
        "args, kwargs, expected",
        [
            ((1.23, 0.15), {}, "1.23 +/- 0.15"),
            ((1.23,), {"relative_error": 0.15}, "1.23 +/- 0.18"),
            ((1.23, 0.15), {"name": "x", "unit": "kg*m/s^2"}, "x = 1.23 +/- 0.15 [kg⋅m/s^2]"),
            ((1.23,), {"relative_error": 0.13, "name": "x"}, "x = 1.23 +/- 0.16"),
            ((1.23, 0.15), {"unit": "kg^2m^2s^-4"}, "1.23 +/- 0.15 [kg^2⋅m^2/s^4]"),
        ],
    )
    def test_str(self, args, kwargs, expected):
        """Tests the __str__ method of ExperimentalValue"""

        with q.option_context(
            "format.style.unit",
            "fraction",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "error",
        ):
            x = q.Measurement(*args, **kwargs)
            assert str(x) == expected
            assert repr(x) == expected


class TestCompare:
    """Tests for comparing experimental values"""

    def test_equal(self):
        """Tests the equal comparison methods of ExperimentalValue"""

        x = q.Measurement(1.23, 0.15)
        y = q.Measurement(1.23, 0.13)
        assert x == y
        assert x == 1.23
        assert 1.23 == y

        z = q.Measurement(1.22, 0.15)
        assert x != z
        assert x != 1.22
        assert 1.22 != x

        assert not x == "a"

    def test_not_equal(self):
        """Tests the not equal methods of ExperimentalValue"""

        x = q.Measurement(2.23, 0.13)
        y = q.Measurement(1.23, 0.15)

        assert x > y
        assert x >= y
        assert y < x
        assert y <= x

        assert x > 1
        assert x < 3
        assert x <= 2.23
        assert x >= 2.23

    def test_not_defined(self):
        """Test comparisons that are not defined"""

        x = q.Measurement(2.23, 0.13)
        with pytest.raises(TypeError):
            print(x < "a")
        with pytest.raises(TypeError):
            print(x > "a")
        with pytest.raises(TypeError):
            print(x <= "a")
        with pytest.raises(TypeError):
            print(x >= "a")


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

        res = func(val1, m2)
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == func(val1, val2)

        res = func(m1, val2)
        assert isinstance(res, q.core.DerivedValue)
        assert res.value == func(val1, val2)
