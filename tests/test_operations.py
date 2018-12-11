"""Unit tests for operations on ExperimentalValue objects

This file contains tests for operations with ExperimentalValue objects. It checks if the
result of the operations are of the proper type. It also checks if errors and units of the
values are propagated properly

"""

import pytest
import qexpy as q
import numbers


class TestArithmetic:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        q.reset_default_configuration()

    def test_value_comparison(self):
        a = q.Measurement(4, 0.5, unit="m")
        b = q.Measurement(10, 2, unit="m")
        assert a < b
        assert a > 2
        assert 3 < b
        assert a == 4
        assert 10 == b

    def test_elementary_operations(self):
        q.set_unit_style(q.UnitStyle.FRACTION)
        q.set_sig_figs_for_error(2)

        a = q.Measurement(4, 0.5, unit="m")
        b = q.Measurement(10, 2, unit="m")
        c = a + b
        assert c.value == 14
        assert c.error == pytest.approx(2.0615528128088303)
        assert str(c) == "14.0 +/- 2.1 [m]"
        c2 = a + 2
        assert c2.value == 6
        assert c2.error == 0.5
        c3 = 5 + a
        assert c3.value == 9
        assert c3.error == 0.5

        h = b - a
        assert h.value == 6
        assert h.error == pytest.approx(2.0615528128088303)
        assert str(h) == "6.0 +/- 2.1 [m]"
        h1 = a - 2
        assert h1.value == 2
        assert h1.error == 0.5
        h2 = 5 - a
        assert h2.value == 1
        assert h2.error == 0.5

        f = q.Measurement(4, 0.5, unit="kg*m/s^2")
        d = q.Measurement(10, 2, unit="m")
        e = f * d
        assert e.value == 40
        assert e.error == pytest.approx(9.433981132056603)
        assert str(e) == "40.0 +/- 9.4 [kg⋅m^2/s^2]"
        e1 = f * 2
        assert e1.value == 8
        e2 = 2 * f
        assert e2.value == 8

        s = q.Measurement(10, 2, unit="m")
        t = q.Measurement(4, 0.5, unit="s")
        v = s / t
        assert v.value == 2.5
        assert v.error == pytest.approx(0.5896238207535377)
        assert str(v) == "2.50 +/- 0.59 [m/s]"
        v1 = 20 / s
        assert v1.value == 2
        v2 = s / 2
        assert v2.value == 5


class TestFunctions:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        q.reset_default_configuration()

    def test_simple_functions(self):
        a = q.Measurement(5, 0.5)
        res_sqrt = q.sqrt(a)
        assert res_sqrt.value == pytest.approx(2.2360679775)
        assert res_sqrt.error == pytest.approx(0.11180339887498948)
        res_sqrt_const = q.sqrt(5)
        assert res_sqrt_const == pytest.approx(2.2360679775)
        assert isinstance(res_sqrt_const, numbers.Real)

        res_exp = q.exp(a)
        assert res_exp.value == pytest.approx(148.4131591025766)
        assert res_exp.error == pytest.approx(74.2065795512883)
        res_exp_const = q.exp(5)
        assert res_exp_const == pytest.approx(148.4131591025766)
        assert isinstance(res_exp_const, numbers.Real)

    def test_trig_functions(self):
        a = q.Measurement(1, 0.5)
        res_sin = q.sin(a)
        assert res_sin.value == pytest.approx(0.8414709848078965)
        assert res_sin.error == pytest.approx(0.2701511529340699)

        ad = q.Measurement(30, 0.5)
        res_sind = q.sind(ad)
        assert res_sind.value == pytest.approx(0.5)
        assert res_sind.error == pytest.approx(0.007557497350975909)

        b = q.Measurement(0.5, 0.02)
        res_asin = q.asin(b)
        assert res_asin.value == pytest.approx(0.5235987755982988)
        assert res_asin.error == pytest.approx(0.023094010767585035)

        res_sec = q.sec(b)
        assert res_sec.value == pytest.approx(1.139493927324549)
        assert res_sec.error == pytest.approx(0.012450167393185607)

    def test_more_advanced_functions(self):
        a = q.Measurement(2, 0.5)
        b = q.Measurement(5, 0.5)
        res_pow = b ** a
        assert res_pow.value == 25
        assert res_pow.error == pytest.approx(20.72999937432251)

        res_ln = q.log(a)
        assert res_ln.value == pytest.approx(0.6931471805599453)
        assert res_ln.error == pytest.approx(0.25)

        res_log = q.log(a, b)
        assert res_log.value == pytest.approx(2.321928094887362)
        assert res_log.error == pytest.approx(0.8497943815525582)
