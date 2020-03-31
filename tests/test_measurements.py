"""Unit tests for recording individual and arrays of measurements"""

import pytest

import numpy as np
import qexpy as q

from qexpy.data.data import RepeatedlyMeasuredValue, MeasuredValue, UndefinedActionError
from qexpy.data.datasets import ExperimentalValueArray
from qexpy.utils.exceptions import IllegalArgumentError


class TestMeasuredValue:
    """Tests for a single measurement"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        """restores all default configurations before each testcase"""
        q.get_settings().reset()

    def test_measurement(self):
        """tests for single measurements"""

        a = q.Measurement(5)
        assert a.value == 5
        assert a.error == 0
        assert str(a) == "5 +/- 0"
        assert repr(a) == "MeasuredValue(5 +/- 0)"

        b = q.Measurement(5, 0.5)
        assert b.value == 5
        assert b.error == 0.5
        assert b.relative_error == 0.1
        assert b.std == 0.5
        assert str(b) == "5.0 +/- 0.5"
        assert repr(b) == "MeasuredValue(5.0 +/- 0.5)"

        c = q.Measurement(12.34, 0.05, name="energy", unit="kg*m^2*s^-2")
        assert str(c) == "energy = 12.34 +/- 0.05 [kg⋅m^2⋅s^-2]"

        q.set_sig_figs_for_error(2)
        assert str(c) == "energy = 12.340 +/- 0.050 [kg⋅m^2⋅s^-2]"
        q.set_sig_figs_for_value(4)
        assert str(c) == "energy = 12.34 +/- 0.05 [kg⋅m^2⋅s^-2]"
        q.set_unit_style(q.UnitStyle.FRACTION)
        assert str(c) == "energy = 12.34 +/- 0.05 [kg⋅m^2/s^2]"

        assert c.derivative(c) == 1
        assert c.derivative(b) == 0

        with pytest.raises(IllegalArgumentError):
            c.derivative(1)

        with pytest.raises(IllegalArgumentError):
            q.Measurement('12.34')

        with pytest.raises(IllegalArgumentError):
            q.Measurement(12.34, '0.05')

        with pytest.raises(TypeError):
            q.Measurement(5, unit=1)

        with pytest.raises(TypeError):
            q.Measurement(5, name=1)

    def test_measurement_setters(self):
        """tests for changing values in a measured value"""

        a = q.Measurement(12.34, 0.05)
        a.value = 50
        assert a.value == 50
        a.error = 0.02
        assert a.error == 0.02
        a.relative_error = 0.05
        assert a.relative_error == 0.05
        assert a.error == 2.5
        a.name = "energy"
        assert a.name == "energy"
        a.unit = "kg*m^2/s^2"
        assert a.unit == "kg⋅m^2⋅s^-2"

        with pytest.raises(TypeError):
            a.value = '1'

        with pytest.raises(TypeError):
            a.error = '1'

        with pytest.raises(ValueError):
            a.error = -1

        with pytest.raises(TypeError):
            a.relative_error = '1'

        with pytest.raises(ValueError):
            a.relative_error = -1

        with pytest.raises(TypeError):
            a.name = 1

        with pytest.raises(TypeError):
            a.unit = 1

    def test_repeated_measurement(self):
        """test recording repeatedly measured values"""

        a = q.Measurement([10, 9.8, 9.9, 10.1, 10.2])
        assert isinstance(a, RepeatedlyMeasuredValue)
        assert a.value == 10
        assert a.error == pytest.approx(0.070710730438880)
        assert a.std == pytest.approx(0.158114)
        a.use_std_for_uncertainty()
        assert a.error == pytest.approx(0.158114)

        assert isinstance(a.raw_data, np.ndarray)
        assert not isinstance(a.raw_data, ExperimentalValueArray)

        b = q.Measurement([10, 9.8, 9.9, 10.1, 10.2], [0.5, 0.3, 0.1, 0.2, 0.2])
        assert isinstance(b, RepeatedlyMeasuredValue)
        assert b.mean == 10
        assert b.value == 10
        assert b.error == pytest.approx(0.070710730438880)

        assert isinstance(b.raw_data, ExperimentalValueArray)
        assert all(b.raw_data == [10, 9.8, 9.9, 10.1, 10.2])

        with pytest.raises(ValueError):
            q.Measurement([10, 9.8, 9.9, 10.1, 10.2], [0.5, 0.3, 0.1, 0.2])

    def test_repeated_measurement_setters(self):
        """test that the setters for repeated measurements behave correctly"""

        a = q.Measurement([10, 9.8, 9.9, 10.1, 10.2], [0.5, 0.3, 0.1, 0.2, 0.2])
        assert isinstance(a, RepeatedlyMeasuredValue)
        a.use_error_weighted_mean_as_value()
        assert a.error_weighted_mean == 9.971399730820997
        assert a.value == 9.971399730820997
        a.use_error_on_mean_for_uncertainty()
        assert a.error_on_mean == pytest.approx(0.070710678118654)
        assert a.error == pytest.approx(0.070710678118654)
        a.use_propagated_error_for_uncertainty()
        assert a.propagated_error == 0.0778236955614928
        assert a.error == 0.0778236955614928

        with pytest.raises(TypeError):
            a.value = '15'

        with pytest.warns(UserWarning):
            a.value = 15

        assert not isinstance(a, RepeatedlyMeasuredValue)
        assert isinstance(a, MeasuredValue)
        assert a.value == 15

    def test_correlation_for_repeated_measurements(self):
        """test covariance and correlation settings between repeated measurements"""

        a = q.Measurement([0.8, 0.9, 1, 1.1])
        b = q.Measurement([2, 2.2, 2.1, 2.3])

        assert q.get_correlation(a, b) == 0
        assert q.get_covariance(a, b) == 0

        q.set_correlation(a, b)
        assert q.get_correlation(a, b) == pytest.approx(0.8)
        assert q.get_covariance(a, b) == pytest.approx(0.01333333333)

        q.set_correlation(a, b, 0)
        assert q.get_correlation(a, b) == 0
        assert q.get_covariance(a, b) == 0

        a.set_covariance(b)
        assert q.get_correlation(a, b) == pytest.approx(0.8)
        assert q.get_covariance(a, b) == pytest.approx(0.01333333333)

        q.set_covariance(a, b, 0)
        assert q.get_correlation(a, b) == 0
        assert q.get_covariance(a, b) == 0

        d = a + b
        with pytest.raises(IllegalArgumentError):
            a.get_covariance(0)
        with pytest.raises(IllegalArgumentError):
            a.get_correlation(0)
        with pytest.raises(IllegalArgumentError):
            a.set_covariance(0, 0)
        with pytest.raises(IllegalArgumentError):
            a.set_correlation(0, 0)
        with pytest.raises(IllegalArgumentError):
            a.set_covariance(d, 0)
        with pytest.raises(IllegalArgumentError):
            a.set_correlation(d, 0)

        c = q.Measurement([0, 1, 2])
        with pytest.raises(IllegalArgumentError):
            q.set_covariance(a, c)
        with pytest.raises(IllegalArgumentError):
            q.set_correlation(a, c)

    def test_correlation_for_single_measurements(self):
        """test covariance and correlation between single measurements"""

        a = q.Measurement(5, 0.5)
        b = q.Measurement(6, 0.2)
        c = q.Measurement(5)
        d = a + b

        assert a.get_covariance(a) == 0.25
        assert a.get_correlation(a) == 1
        assert a.get_covariance(c) == 0
        assert a.get_correlation(c) == 0

        assert d.get_covariance(a) == 0
        assert d.get_correlation(a) == 0
        assert a.get_covariance(d) == 0
        assert a.get_correlation(d) == 0
        assert q.get_covariance(a, d) == 0
        assert q.get_correlation(a, d) == 0

    def test_illegal_correlation_settings(self):
        """test illegal correlation and covariance settings"""

        a = q.Measurement(5, 0.5)
        b = q.Measurement(6, 0.2)
        c = q.Measurement(5)
        d = a + b

        with pytest.raises(IllegalArgumentError):
            q.set_correlation(a, 0, 0)
        with pytest.raises(IllegalArgumentError):
            q.set_covariance(a, 0, 0)
        with pytest.raises(IllegalArgumentError):
            a.set_correlation(0, 0)
        with pytest.raises(IllegalArgumentError):
            a.set_covariance(0, 0)
        with pytest.raises(UndefinedActionError):
            d.set_correlation(a, 0)
        with pytest.raises(UndefinedActionError):
            d.set_covariance(a, 0)
        with pytest.raises(IllegalArgumentError):
            a.set_covariance(d, 0)
        with pytest.raises(IllegalArgumentError):
            a.set_correlation(d, 0)
        with pytest.raises(ArithmeticError):
            a.set_covariance(c, 1)
        with pytest.raises(ArithmeticError):
            a.set_correlation(c, 1)

        with pytest.raises(IllegalArgumentError):
            a.get_correlation(0)
        with pytest.raises(IllegalArgumentError):
            a.get_covariance(0)
        with pytest.raises(IllegalArgumentError):
            q.get_correlation(a, 0)
        with pytest.raises(IllegalArgumentError):
            q.get_covariance(a, 0)

        with pytest.raises(ValueError):
            q.set_covariance(a, b, 100)
        with pytest.raises(ValueError):
            q.set_correlation(a, b, 2)
