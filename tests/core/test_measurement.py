"""Unit tests for the Measurement."""

import numpy as np
import pytest

import qexpy as q


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
        """Tests different ways to resolve the error."""

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
