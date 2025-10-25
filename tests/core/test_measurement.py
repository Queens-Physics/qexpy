"""Unit tests for the Measurement."""

import numpy as np
import pytest
import scipy

import qexpy as q
from qexpy.core.measurements import RepeatedMeasurement


class TestMeasurement:
    """Tests for the Measurement base class."""

    def test_simple_measurement(self):
        """Tests taking a single measurement."""

        m = q.Measurement(1.23, 0.25, name="force", unit="kg*m/s^2")
        assert m.value == 1.23
        assert m.error == 0.25
        assert np.isclose(m.relative_error, 0.25 / 1.23)
        assert m.name == "force"
        assert m.unit == {"kg": 1, "m": 1, "s": -2}
        assert str(m) == "force = 1.2 +/- 0.2 [kg⋅m/s^2]"

    def test_error_resolution(self):
        """Tests different ways to specify the error."""

        m = q.Measurement(1.23, 0.3)
        assert m.error == 0.3
        assert np.isclose(m.relative_error, 0.3 / 1.23)

        m = q.Measurement(1.23, relative_error=0.3)
        assert m.relative_error == 0.3
        assert np.isclose(m.error, 0.3 * 1.23)

        m = q.Measurement(0.0, 0.1)
        assert m.error == 0.1
        assert m.relative_error == np.inf

    def test_negative_error(self):
        """Test that negative error raises ValueError."""

        with pytest.raises(ValueError, match="The error must be non-negative"):
            q.Measurement(4.2, -0.1)

        with pytest.raises(ValueError, match="The relative error must be non-negative"):
            q.Measurement(4.2, relative_error=-0.1)

    def test_repeated_measurement(self):
        """Tests taking a series of repeated measurements."""

        samples = [1.23, 1.2, 1.9, 1.25, 1.93]
        m = q.Measurement(samples)
        assert isinstance(m, RepeatedMeasurement)

        assert np.isclose(m.value, np.mean(samples))
        assert np.isclose(m.error, scipy.stats.sem(samples))

        m.use_standard_deviation()
        assert np.isclose(m.error, np.std(samples, ddof=1))

        m.use_standard_error()
        assert np.isclose(m.error, scipy.stats.sem(samples))

    def test_repeated_measurement_with_error(self):
        """Tests taking repeated measurements with errors."""

        samples = [1.23, 1.2, 1.9, 1.25, 1.93]
        errors = [0.05, 0.1, 0.2, 0.02, 0.01]
        m = q.Measurement(samples, errors)

        expected_value, expected_error = _weighted_mean_and_error(samples, errors)

        assert np.isclose(m.value, expected_value)
        assert np.isclose(m.error, expected_error)

        m.use_standard_error()
        assert np.isclose(m.value, np.mean(samples))
        assert np.isclose(m.error, scipy.stats.sem(samples))

        m.use_error_weighted_mean()
        assert np.isclose(m.value, expected_value)
        assert np.isclose(m.error, expected_error)

    def test_repeated_error_resolution(self):
        """Tests different ways to specify errors to a repeated measurement."""

        samples = [1.23, 1.2, 1.9, 1.25, 1.93]
        m = q.Measurement(samples, 0.05)
        errors = np.ones_like(samples) * 0.05
        expected_value, expected_error = _weighted_mean_and_error(samples, errors)
        assert np.isclose(m.value, expected_value)
        assert np.isclose(m.error, expected_error)

        m = q.Measurement(samples, relative_error=0.05)
        errors = np.asarray(samples) * 0.05
        expected_value, expected_error = _weighted_mean_and_error(samples, errors)
        assert np.isclose(m.value, expected_value)
        assert np.isclose(m.error, expected_error)

        relative_errors = [0.01, 0.05, 0.02, 0.03, 0.04]
        m = q.Measurement(samples, relative_error=relative_errors)
        errors = np.asarray(samples) * np.asarray(relative_errors)
        expected_value, expected_error = _weighted_mean_and_error(samples, errors)
        assert np.isclose(m.value, expected_value)
        assert np.isclose(m.error, expected_error)

    def test_invalid_errors(self):
        """Tests that the correct error is raised when the errors is invalid."""

        with pytest.raises(TypeError, match="Cannot provide an array of errors"):
            q.Measurement(0.5, [0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="must be non-negative"):
            q.Measurement(0.5, -0.1)
        with pytest.raises(ValueError, match="must be non-negative"):
            q.Measurement(0.5, relative_error=-0.1)
        with pytest.raises(ValueError, match="must be non-negative"):
            q.Measurement([0.5, 0.2], -0.1)
        with pytest.raises(ValueError, match="must be non-negative"):
            q.Measurement([0.5, 0.2], relative_error=-0.1)

        with pytest.raises(ValueError, match="The length of the error"):
            q.Measurement([0.1, 0.2, 0.3], [0.04, 0.05])
        with pytest.raises(ValueError, match="The length of the relative_error"):
            q.Measurement([0.1, 0.2, 0.3], relative_error=[0.04, 0.05])


def _weighted_mean_and_error(samples, errors):
    weights = 1 / (np.array(errors) ** 2)
    expected_value = np.sum(weights * samples) / np.sum(weights)
    expected_error = 1 / np.sqrt(np.sum(weights))
    return expected_value, expected_error
