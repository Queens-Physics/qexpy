"""Tests for the internal unit module."""

import numpy as np
import pytest

from qexpy.units import Unit

PREDEFINED = {
    "empty": {},
    "hertz": {"s": -1},
    "newton": {"kg": 1, "m": 1, "s": -2},
    "pascal": {"kg": 1, "m": -1, "s": -2},
    "joule": {"kg": 1, "m": 2, "s": -2},
    "coloumb": {"s": 1, "A": 1},
    "inv-coloumb": {"s": -1, "A": -1},
    "volt": {"kg": 1, "m": 2, "s": -3, "A": -1},
    "farad": {"kg": -1, "m": -2, "s": 4, "A": 2},
    "foo": {"kg": 4, "m": 2, "Pa": 1, "L": -3, "s": -2, "A": -2},
    "bar": {"kg": 1 / 2, "m": 1 / 3, "s": -1.234, "A": -2.345},
}


def assert_unit_equal(actual, expected):
    """Compare two dictionaries representing units."""
    assert actual.keys() == expected.keys()
    for k in actual:
        assert np.allclose(actual[k], expected[k])


@pytest.mark.parametrize(
    "unit_str, name",
    [
        ("", "empty"),
        ("s^-1", "hertz"),
        ("1/s", "hertz"),
        ("kg*m/s^2", "newton"),
        ("kg^1m^1s^-2", "newton"),
        ("kg/(m*s^2)", "pascal"),
        ("kg/m^1s^2", "pascal"),
        ("kg*m^2/s^2", "joule"),
        ("kg(m/s^2)m", "joule"),
        ("s*A", "coloumb"),
        ("1/(s*A)", "inv-coloumb"),
        ("s^-1A^-1", "inv-coloumb"),
        ("kg*m^2/s^3A^1", "volt"),
        ("s^4A^2kg^-1m^-2", "farad"),
        ("kg^4m^2Pa^1L^-3s^-2A^-2", "foo"),
        ("kg^4m^2Pa/L^3s^2A^2", "foo"),
        ("(kg^4*m^2*Pa)/(L^3*s^2*A^2)", "foo"),
        ("kg^1/2m^(1/3)/s^(1.234)A^2.345", "bar"),
    ],
)
def test_unit_from_string(unit_str: str, name: str):
    """Tests constructing a unit expression from a string."""
    actual = Unit(unit_str)._unit
    assert_unit_equal(actual, PREDEFINED[name])


class TestUnitOperations:
    """Tests performing operations with units."""

    def test_unit_addition_and_subtraction(self):
        """Tests adding and subtracting two units."""

        unit1 = Unit({"kg": 1, "s": -2})
        unit2 = Unit({"kg": 2, "s": 2, "m": 1})

        with pytest.warns(UserWarning, match="mismatching units"):
            assert unit1 + unit2 == Unit({})
            assert unit2 - unit1 == Unit({})

        unit3 = Unit({"kg": 1, "s": -2})
        assert unit1 + unit3 == Unit({"kg": 1, "s": -2})
        assert unit1 - unit3 == Unit({"kg": 1, "s": -2})

    def test_unit_multiplication_and_division(self):
        """Tests multiplying and dividing two units."""

        unit1 = Unit({"kg": 1, "s": -2})
        unit2 = Unit({"kg": 2, "s": 2, "m": 1})

        assert unit1 * unit2 == Unit({"kg": 3, "m": 1})
        assert unit1 / unit2 == Unit({"kg": -1, "s": -4, "m": -1})
        assert 1 / unit2 == Unit({"kg": -2, "s": -2, "m": -1})

    def test_unit_exponentiation(self):
        """Tests exponents of a unit."""

        unit = Unit({"kg": 1, "m": 2, "s": -2})
        assert_unit_equal((unit**2)._unit, {"kg": 2, "m": 4, "s": -4})
        assert_unit_equal((unit**0.5)._unit, {"kg": 0.5, "m": 1, "s": -1})
