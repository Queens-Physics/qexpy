"""Unit tests for operations on ExperimentalValue objects

This file contains tests for operations with ExperimentalValue objects. It checks if the
result of the operations are of the proper type. It also checks if errors and units of the
values are propagated properly

"""

import pytest
import qexpy as q


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
