"""Tests for operations between experimental values"""

import pytest
import qexpy as q

from qexpy.data.data import ExperimentalValue
from qexpy.utils.exceptions import UndefinedOperationError


class TestArithmetic:
    """tests for basic arithmetic operations"""

    def test_value_comparison(self):
        """tests for comparing values"""

        a = q.Measurement(4, 0.5, unit="m")
        b = q.Measurement(10, 2, unit="m")
        c = q.Measurement(10, 1)

        assert a < b
        assert a <= b
        assert b >= a
        assert a > 2
        assert 3 < b
        assert a == 4
        assert 10 == b
        assert b == c

    def test_elementary_operations(self):
        """tests for elementary arithmetic operations"""

        a = q.Measurement(4, 0.5, unit="m")
        b = q.Measurement(10, 2, unit="m")

        c = a + b
        assert c.value == 14
        assert c.error == pytest.approx(2.0615528128088303)
        assert str(c) == "14 +/- 2 [m]"

        c2 = a + 2
        assert c2.value == 6
        assert c2.error == 0.5
        assert str(c2) == "6.0 +/- 0.5 [m]"

        c3 = 5 + a
        assert c3.value == 9
        assert c3.error == 0.5

        c4 = a + (10, 2)
        assert c4.value == 14
        assert c4.error == pytest.approx(2.0615528128088303)
        assert str(c4) == "14 +/- 2"

        c5 = -a
        assert str(c5) == "-4.0 +/- 0.5 [m]"

        h = b - a
        assert h.value == 6
        assert h.error == pytest.approx(2.0615528128088303)
        assert str(h) == "6 +/- 2 [m]"

        h1 = a - 2
        assert h1.value == 2
        assert h1.error == 0.5
        assert str(h1) == "2.0 +/- 0.5 [m]"

        h2 = 5 - a
        assert h2.value == 1
        assert h2.error == 0.5

        f = q.Measurement(4, 0.5, unit="kg*m/s^2")
        d = q.Measurement(10, 2, unit="m")

        e = f * d
        assert e.value == 40
        assert e.error == pytest.approx(9.433981132056603)
        assert str(e) == "40 +/- 9 [kg⋅m^2⋅s^-2]"

        e1 = f * 2
        assert e1.value == 8
        assert str(e1) == "8 +/- 1 [kg⋅m⋅s^-2]"

        e2 = 2 * f
        assert e2.value == 8

        s = q.Measurement(10, 2, unit="m")
        t = q.Measurement(4, 0.5, unit="s")

        v = s / t
        assert v.value == 2.5
        assert v.error == pytest.approx(0.5896238207535377)
        assert str(v) == "2.5 +/- 0.6 [m⋅s^-1]"

        v1 = 20 / s
        assert v1.value == 2
        assert str(v1) == "2.0 +/- 0.4 [m^-1]"

        v2 = s / 2
        assert v2.value == 5

        with pytest.raises(UndefinedOperationError):
            s + 'a'

        k = q.Measurement(5, 0.5, unit="m")

        m = k ** 2
        assert str(m) == "25 +/- 5 [m^2]"

        n = 2 ** k
        assert n.value == 32
        assert n.error == pytest.approx(11.09035488895912495)
        assert str(n) == "30 +/- 10"

    def test_vectorized_arithmetic(self):
        """tests for arithmetic with experimental value arrays"""

        a = q.MeasurementArray([1, 2, 3, 4, 5], 0.5, unit="s")

        res = a + 2
        assert all(res.values == [3, 4, 5, 6, 7])
        assert all(res.errors == [0.5, 0.5, 0.5, 0.5, 0.5])
        assert res.unit == "s"

        res = 2 + a
        assert all(res.values == [3, 4, 5, 6, 7])
        assert all(res.errors == [0.5, 0.5, 0.5, 0.5, 0.5])
        assert res.unit == "s"

        res = a + (2, 0.5)
        assert all(res.values == [3, 4, 5, 6, 7])

        res = (2, 0.5) + a
        assert all(res.values == [3, 4, 5, 6, 7])

        res = q.Measurement(2, 0.5) + a
        assert all(res.values == [3, 4, 5, 6, 7])

        res = a + [1, 2, 3, 4, 5]
        assert all(res.values == [2, 4, 6, 8, 10])

        res = [1, 2, 3, 4, 5] + a
        assert all(res.values == [2, 4, 6, 8, 10])

        res = a - 1
        assert all(res.values == [0, 1, 2, 3, 4])

        res = 10 - a
        assert all(res.values == [9, 8, 7, 6, 5])

        res = q.Measurement(10, 0.5) - a
        assert all(res.values == [9, 8, 7, 6, 5])

        res = a - [1, 2, 3, 4, 5]
        assert all(res.values == [0, 0, 0, 0, 0])

        res = [1, 2, 3, 4, 5] - a
        assert all(res.values == [0, 0, 0, 0, 0])

        res = a * 2
        assert all(res.values == [2, 4, 6, 8, 10])

        res = 2 * a
        assert all(res.values == [2, 4, 6, 8, 10])

        res = q.Measurement(2, 0.5) * a
        assert all(res.values == [2, 4, 6, 8, 10])

        b = q.MeasurementArray([10, 20, 30, 40, 50], 0.5, unit="m")

        res = b * a
        assert all(res.values == [10, 40, 90, 160, 250])
        assert res.unit == "m⋅s"

        res = [1, 2, 3, 4, 5] * a
        assert all(res.values == [1, 4, 9, 16, 25])

        res = a / 2
        assert all(res.values == [0.5, 1, 1.5, 2, 2.5])

        res = 2 / a
        assert all(res.values == [2, 1, 2 / 3, 2 / 4, 2 / 5])

        res = q.Measurement(2, 0.5) / a
        assert all(res.values == [2, 1, 2 / 3, 2 / 4, 2 / 5])

        res = b / a
        assert all(res.values == [10, 10, 10, 10, 10])
        assert res.unit == "m⋅s^-1"

        res = [1, 2, 3, 4, 5] / a
        assert all(res.values == [1, 1, 1, 1, 1])

        res = a ** 2
        assert all(res.values == [1, 4, 9, 16, 25])

        res = 2 ** a
        assert all(res.values == [2, 4, 8, 16, 32])

        res = q.Measurement(2, 0.5) ** a
        assert all(res.values == [2, 4, 8, 16, 32])

        res = a ** [2, 2, 2, 2, 2]
        assert all(res.values == [1, 4, 9, 16, 25])
        assert res.unit == "s^2"

        res = [2, 2, 2, 2, 2] ** a
        assert all(res.values == [2, 4, 8, 16, 32])

    def test_composite_operations(self):
        """tests combining several operations"""

        d = q.Measurement(5, 0.1, unit="m")
        m = q.Measurement(10, 0.5, unit="kg")
        t = q.Measurement(2, 0.1, unit="s")

        v = d / t
        e = 1 / 2 * m * (v ** 2)

        assert e.value == 31.25
        assert e.error == pytest.approx(3.7107319021993495716)
        assert e.unit == "kg⋅m^2⋅s^-2"

        res = (d ** 2) ** (1 / 3)
        assert res.unit == "m^(2/3)"


class TestMathFunctions:
    """tests for math function wrappers"""

    def test_math_functions(self):
        """tests for math functions on single values"""

        a = q.Measurement(4, 0.5, unit="m")

        res = q.sqrt(a)
        assert res.value == 2
        assert res.error == 0.125
        assert res.unit == "m^(1/2)"
        assert q.sqrt(4) == 2

        with pytest.raises(UndefinedOperationError):
            q.sqrt("a")

        res = q.exp(a)
        assert res.value == pytest.approx(54.598150033144239)
        assert res.error == pytest.approx(27.299075016572120)

        res = q.log(a)
        assert res.value == pytest.approx(1.3862943611198906)
        assert res.error == 0.125

        res = q.log(2, a)
        assert res.value == 2
        assert res.error == pytest.approx(0.1803368801111204)

        res = q.log10(a)
        assert res.value == pytest.approx(0.6020599913279624)
        assert res.error == pytest.approx(0.0542868102379065)

        with pytest.raises(TypeError):
            q.log(2, a, 2)

    def test_trig_functions(self):
        """tests for trigonometric functions"""

        a = q.Measurement(0.7853981633974483)
        b = q.Measurement(45)
        c = q.Measurement(0.5)

        res = q.sin(a)
        assert res.value == pytest.approx(0.7071067811865475244)

        res = q.sind(b)
        assert res.value == pytest.approx(0.7071067811865475244)

        res = q.cos(a)
        assert res.value == pytest.approx(0.7071067811865475244)

        res = q.cosd(b)
        assert res.value == pytest.approx(0.7071067811865475244)

        res = q.tan(a)
        assert res.value == pytest.approx(1)

        res = q.tand(b)
        assert res.value == pytest.approx(1)

        res = q.sec(a)
        assert res.value == pytest.approx(1.4142135623730950488)

        res = q.secd(b)
        assert res.value == pytest.approx(1.4142135623730950488)

        res = q.csc(a)
        assert res.value == pytest.approx(1.4142135623730950488)

        res = q.cscd(b)
        assert res.value == pytest.approx(1.4142135623730950488)

        res = q.cot(a)
        assert res.value == pytest.approx(1)

        res = q.cotd(b)
        assert res.value == pytest.approx(1)

        res = q.asin(c)
        assert res.value == pytest.approx(0.523598775598298873077)

        res = q.acos(c)
        assert res.value == pytest.approx(1.047197551196597746154)

        res = q.atan(c)
        assert res.value == pytest.approx(0.463647609000806116214)

    def test_vectorized_functions(self):
        """tests for functions on experimental value arrays"""

        a = q.MeasurementArray([1, 2, 3, 4, 5])
        assert q.mean(a) == 3
        assert q.std(a) == pytest.approx(1.58113883008419)
        assert q.sum(a) == 15

        b = [1, 2, 3, 4, 5]

        res = q.mean(b)

        assert res == 3
        assert not isinstance(res, ExperimentalValue)

        assert q.std(b) == pytest.approx(1.58113883008419)
        assert q.sum(b) == 15
