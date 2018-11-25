"""Unit tests for operations on ExperimentalValue objects

This file contains tests for operations with ExperimentalValue objects. It
checks if the result of the operations are of the proper type. It also checks
if errors are propagated properly

"""

# noinspection PyPackageRequirements
import pytest
import qexpy as q


class TestOperands:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        q.reset_default_configuration()

    def test_elementary_operations(self):
        q.set_unit_style(q.UnitStyle.FRACTION)
        q.set_sig_figs_for_error(2)

        a = q.Measurement(4, 0.5, unit="m")
        b = q.Measurement(10, 2, unit="m")
        c = a + b
        assert c.value == 14
        assert c.error == pytest.approx(2.0615528128088303)
        assert str(c) == "14.0 +/- 2.1 [m]"

        h = b - a
        assert h.value == 6
        assert h.error == pytest.approx(2.0615528128088303)
        assert str(h) == "6.0 +/- 2.1 [m]"

        f = q.Measurement(4, 0.5, unit="kg*m/s^2")
        d = q.Measurement(10, 2, unit="m")
        e = f * d
        assert e.value == 40
        assert e.error == pytest.approx(9.433981132056603)
        assert str(e) == "40.0 +/- 9.4 [kg⋅m^2/s^2]"

        s = q.Measurement(10, 2, unit="m")
        t = q.Measurement(4, 0.5, unit="s")
        v = s / t
        assert v.value == 2.5
        assert v.error == pytest.approx(0.5896238207535377)
        assert str(v) == "2.50 +/- 0.59 [m/s]"
