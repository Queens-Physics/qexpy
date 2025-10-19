"""Unit tests for the basic functionalities of the Quantity."""

import numpy as np

import qexpy as q


class TestQuantity:
    """Tests the basic properties of a Quantity."""

    def test_attributes(self):
        """Tests accessing attributes."""

        f = q.Constant(-1.234, 0.023, "force", "kg*m/s^2")
        assert f.name == "force"
        assert f.unit == {"kg": 1, "m": 1, "s": -2}
        assert f.value == -1.234
        assert f.error == 0.023
        assert np.isclose(f.relative_error, 0.023 / 1.234)

    def test_mutations(self):
        """Tests setters for name and unit."""

        f = q.Constant(-1.234, 0.023)
        f.name = "force"
        f.unit = "kg*m/s^2"
        assert f.name == "force"
        assert f.unit == {"kg": 1, "m": 1, "s": -2}

    def test_relative_error_zero_value(self):
        """Tests that the relative_error is defined."""

        f = q.Constant(0.0, 0.1)
        assert f.relative_error == np.inf

        f = q.Constant(1.0, 0)
        assert f.relative_error == 0

    def test_str(self):
        """Tests the __str__ method of the Quantity class."""

        f = q.Constant(-1.234, 0.023)
        assert str(f) == "-1.23 +/- 0.02"

        f.name = "force"
        f.unit = "kg*m/s^2"
        assert str(f) == "force = -1.23 +/- 0.02 [kg⋅m/s^2]"

    def test_comparisons(self):
        """Tests comparing quantities."""

        x = q.Constant(1.23, 0.15)
        y = q.Constant(1.23, 0.25)
        z = q.Constant(2.34, 0.15)

        assert x == y
        assert x != z
