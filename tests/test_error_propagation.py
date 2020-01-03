"""Tests for different error propagation methods"""

import pytest
import qexpy as q

from qexpy.data.data import ExperimentalValue, MeasuredValue
from qexpy.utils.exceptions import IllegalArgumentError


class TestDerivedValue:
    """tests for the derived value class"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        """resets all default configurations"""
        q.get_settings().reset()
        q.reset_correlations()

    def test_derivative_method(self):
        """tests error propagation using the derivative method"""

        a = q.Measurement(5, 0.5)
        b = q.Measurement(2, 0.2)

        res = q.sqrt((a + b) / 2)
        assert res.error == pytest.approx(0.0719622917128924443443)
        assert str(res) == "1.87 +/- 0.07"

    def test_monte_carlo_method(self):
        """tests error propagation using the monte carlo method"""

        q.set_error_method(q.ErrorMethod.MONTE_CARLO)

        a = q.Measurement(5, 0.5)
        b = q.Measurement(2, 0.2)

        res = q.sqrt((a + b) / 2)
        assert res.error == pytest.approx(0.071962291712, abs=1e-2)
        assert str(res) == "1.87 +/- 0.07"

        res.mc.sample_size = 10000000
        assert res.mc.samples().size == 10000000
        assert res.error == pytest.approx(0.071962291712, abs=1e-3)

        res.mc.reset_sample_size()
        assert res.mc.sample_size == 10000

        with pytest.raises(ValueError):
            res.mc.sample_size = -1

        G = 6.67384e-11  # the gravitational constant
        m1 = q.Measurement(40e4, 2e4, name="m1", unit="kg")
        m2 = q.Measurement(30e4, 10e4, name="m2", unit="kg")
        r = q.Measurement(3.2, 0.5, name="distance", unit="m")

        f = G * m1 * m2 / (r ** 2)

        f.mc.confidence = 0.68
        assert f.mc.confidence == 0.68

        f.mc.use_mode_with_confidence()
        assert f.value == pytest.approx(0.68, abs=0.15)
        assert f.error == pytest.approx(0.36, abs=0.15)

        with pytest.raises(ValueError):
            f.mc.confidence = -1
        with pytest.raises(TypeError):
            f.mc.confidence = '1'

        f.mc.use_mode_with_confidence(0.3)
        assert f.error == pytest.approx(0.15, abs=0.15)

        f.mc.confidence = 0.3
        assert f.error == pytest.approx(0.15, abs=0.15)

        f.mc.use_mean_and_std()
        assert f.value == pytest.approx(0.848, abs=0.15)
        assert f.error == pytest.approx(0.435, abs=0.15)

        f.mc.sample_size = 10000000
        f.mc.set_xrange(-1, 4)

        assert f.value == pytest.approx(0.848, abs=0.05)
        assert f.error == pytest.approx(0.435, abs=0.05)

        with pytest.raises(TypeError):
            f.mc.set_xrange('1')
        with pytest.raises(ValueError):
            f.mc.set_xrange(4, 1)

        f.mc.set_xrange()
        assert f.mc.xrange == ()

        f.mc.use_custom_value_and_error(0.8, 0.4)
        assert f.value == 0.8
        assert f.error == 0.4

        with pytest.raises(TypeError):
            f.mc.use_custom_value_and_error('a', 0.4)
        with pytest.raises(TypeError):
            f.mc.use_custom_value_and_error(0.8, 'a')
        with pytest.raises(ValueError):
            f.mc.use_custom_value_and_error(0.8, -0.5)

        f.recalculate()
        assert f.value == pytest.approx(0.848, abs=0.15)
        assert f.error == pytest.approx(0.435, abs=0.15)

        k = q.Measurement(0.01, 0.1)
        res = q.log(k)
        with pytest.warns(UserWarning):
            assert res.value != pytest.approx(-4.6)

    def test_correlated_measurements(self):
        """tests error propagation for correlated measurements"""

        a = q.Measurement(5, 0.5)
        b = q.Measurement(2, 0.2)

        q.set_covariance(a, b, 0.08)

        res = q.sqrt((a + b) / 2)
        assert res.error == pytest.approx(0.08964214570007952299766)

        res.error_method = q.ErrorMethod.MONTE_CARLO
        assert res.error == pytest.approx(0.0896421457001, abs=1e-2)

    def test_manipulate_derived_value(self):
        """unit tests for the derived value class"""

        a = q.Measurement(5, 0.5)
        b = q.Measurement(2, 0.5)

        res = a + b
        assert res.value == 7
        assert res.error == pytest.approx(0.7071067811865476)
        assert res.relative_error == pytest.approx(0.1010152544552210749)

        assert res.derivative(a) == 1

        with pytest.raises(IllegalArgumentError):
            res.derivative(1)

        res.error_method = q.ErrorMethod.MONTE_CARLO
        assert res.error_method == q.ErrorMethod.MONTE_CARLO

        res.error_method = "derivative"
        assert res.error_method == q.ErrorMethod.DERIVATIVE

        res.reset_error_method()
        assert res.error_method == q.ErrorMethod.DERIVATIVE

        with pytest.raises(ValueError):
            res.error_method = "hello"

        with pytest.raises(TypeError):
            res.value = 'a'
        with pytest.raises(TypeError):
            res.error = 'a'
        with pytest.raises(ValueError):
            res.error = -1
        with pytest.raises(TypeError):
            res.relative_error = 'a'
        with pytest.raises(ValueError):
            res.relative_error = -1

        with pytest.warns(UserWarning):
            res.value = 6
        assert res.value == 6
        assert isinstance(res, MeasuredValue)

        res = a + b
        with pytest.warns(UserWarning):
            res.error = 0.5
        assert res.error == 0.5
        assert isinstance(res, MeasuredValue)

        res = a + b
        with pytest.warns(UserWarning):
            res.relative_error = 0.5
        assert res.relative_error == 0.5
        assert isinstance(res, MeasuredValue)
