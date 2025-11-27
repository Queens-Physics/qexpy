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
        assert x != "a"

        assert x == 1.23
        assert x != 2.34

        assert x < z
        assert x <= y
        assert x < 2.34
        assert x <= 2.34
        assert z > 1.23
        assert z >= 2.23

        assert z > x
        assert y >= x
        assert x < 2.34
        assert x <= 2.34
        assert z > 1.23
        assert z >= 2.23


class TestConstantOperations:
    """Tests that constant operations produce constants."""

    def test_add(self):
        """Test constant additions."""

        a = q.Constant(5, 0.1)
        b = q.Constant(2, 0.1)

        res = a + b
        assert isinstance(res, q.Constant)
        assert res == 7

        res = a + 2
        assert isinstance(res, q.Constant)
        assert res == 7

        res = 5 + b
        assert isinstance(res, q.Constant)
        assert res == 7

    def test_subtract(self):
        """Test constant subtractions."""

        a = q.Constant(5, 0.1)
        b = q.Constant(2, 0.1)

        res = a - b
        assert isinstance(res, q.Constant)
        assert res == 3

        res = a - 2
        assert isinstance(res, q.Constant)
        assert res == 3

        res = 5 - b
        assert isinstance(res, q.Constant)
        assert res == 3

    def test_multiply(self):
        """Test constant multiplication."""

        a = q.Constant(5, 0.1)
        b = q.Constant(2, 0.2)

        res = a * b
        assert isinstance(res, q.Constant)
        assert res == 10

        res = a * 2
        assert isinstance(res, q.Constant)
        assert res == 10

        res = 5 * b
        assert isinstance(res, q.Constant)
        assert res == 10

    def test_division(self):
        """Test constant division."""

        a = q.Constant(5, 0.1)
        b = q.Constant(2, 0.2)

        res = a / b
        assert isinstance(res, q.Constant)
        assert res == 2.5

        res = a / 2
        assert isinstance(res, q.Constant)
        assert res == 2.5

        res = 5 / b
        assert isinstance(res, q.Constant)
        assert res == 2.5

    def test_power(self):
        """Test taking constant powers."""

        a = q.Constant(5, 0.1)
        b = q.Constant(2, 0.2)

        res = a**b
        assert isinstance(res, q.Constant)
        assert res == 25

        res = a**2
        assert isinstance(res, q.Constant)
        assert res == 25

        res = 5**b
        assert isinstance(res, q.Constant)
        assert res == 25

    def test_negate(self):
        """Test negative of a constant."""

        a = q.Constant(5, 0.1)
        res = -a
        assert isinstance(res, q.Constant)
        assert res == -5

    def test_ufunc(self):
        """Test numpy functions on constants."""

        a = q.Constant(25)
        res = np.sqrt(a)
        assert isinstance(res, q.Constant)
        assert res == 5

        a = q.Constant(5)
        b = q.Constant(2)

        res = np.power(a, b)
        assert isinstance(res, q.Constant)
        assert res == 25

        res = np.power(a, 2)
        assert isinstance(res, q.Constant)
        assert res == 25

        res = np.power(5, b)
        assert isinstance(res, q.Constant)
        assert res == 25
