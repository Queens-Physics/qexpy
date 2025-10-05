"""Tests for the internal unit module."""

import numpy as np
import pytest

import qexpy as q
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
        ("(s^2A^1)^2/(kg*m^2)", "farad"),
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


@pytest.mark.parametrize(
    "unit_str",
    [
        "k2m^2",
        "m^2*/kg^3",
        "m^2/^3",
        "(kg*)m",
        "m*1/kg",
        "m*kg)",
        "(m^2kg^3",
        "m^2*",
    ],
)
def test_invalid_unit_expression(unit_str: str):
    """Tests that an error is raised when the unit is invalid."""
    with pytest.raises(ValueError):
        Unit(unit_str)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("empty", ""),
        ("hertz", "1/s"),
        ("newton", "kg⋅m/s^2"),
        ("pascal", "kg/(m⋅s^2)"),
        ("joule", "kg⋅m^2/s^2"),
        ("coloumb", "s⋅A"),
        ("inv-coloumb", "1/(s⋅A)"),
        ("volt", "kg⋅m^2/(s^3⋅A)"),
        ("farad", "s^4⋅A^2/(kg⋅m^2)"),
        ("foo", "kg^4⋅m^2⋅Pa/(L^3⋅s^2⋅A^2)"),
        ("bar", "kg^(1/2)⋅m^(1/3)/(s^1.23⋅A^2.35)"),
    ],
)
def test_unit_to_fraction(name: str, expected: str):
    """Tests formatting a unit to a string in the fraction form."""
    unit = Unit(PREDEFINED[name])
    with q.set_option_context("format.unit", "fraction"):
        assert str(unit) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("empty", ""),
        ("hertz", "s^-1"),
        ("newton", "kg⋅m⋅s^-2"),
        ("pascal", "kg⋅m^-1⋅s^-2"),
        ("joule", "kg⋅m^2⋅s^-2"),
        ("coloumb", "s⋅A"),
        ("inv-coloumb", "s^-1⋅A^-1"),
        ("volt", "kg⋅m^2⋅s^-3⋅A^-1"),
        ("farad", "kg^-1⋅m^-2⋅s^4⋅A^2"),
        ("foo", "kg^4⋅m^2⋅Pa⋅L^-3⋅s^-2⋅A^-2"),
        ("bar", "kg^(1/2)⋅m^(1/3)⋅s^-1.23⋅A^-2.35"),
    ],
)
def test_unit_to_product(name: str, expected: str):
    """Tests formatting a unit to a string in the product form."""
    unit = Unit(PREDEFINED[name])
    with q.set_option_context("format.unit", "product"):
        assert str(unit) == expected


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


class TestUnitAliases:
    """Tests defining aliases for compound units."""

    @pytest.fixture(autouse=True)
    def teardown(self):
        """Reset everything."""
        q.clear_unit_aliases()

    def test_define_unit(self):
        """Tests defining a unit alias."""

        q.define_unit("N", "kg^1m^1s^-2")
        assert str(Unit("kg*m/s^2")) == "N"
        assert str(Unit("kg^2m^2s^-4")) == "N^2"
        assert str(Unit("kg^(-1/2)m^(-1/2)s^1")) == "1/N^(1/2)"
        assert str(Unit("kg^3m^2s^-4")) == "kg^3⋅m^2/s^4"

    def test_unit_operations(self):
        """Tests unit operations with unit aliases."""

        q.define_unit("F", "C^2/(N*m)")
        q.define_unit("N", "kg*m/s^2")

        unit_q = Unit({"C": 1})
        unit_r = Unit({"m": 1})
        unit_eps = Unit({"F": 1, "m": -1})

        res = unit_q * unit_q / unit_eps / unit_r**2
        assert res == Unit({"kg": 1, "m": 1, "s": -2})
        assert str(res) == "N"

    def test_unit_not_unpacked_if_unnecessary(self):
        """Tests that pre-defined units are not unpacked when not necessary."""

        q.define_unit("N", "kg*m/s^2")

        unit_1 = Unit({"N": 1})

        assert unit_1 + Unit({}) == Unit({"N": 1})
        assert unit_1 - Unit({}) == Unit({"N": 1})
        assert unit_1 * Unit({}) == Unit({"N": 1})
        assert Unit({}) * unit_1 == Unit({"N": 1})
        assert unit_1 / Unit({}) == Unit({"N": 1})
        assert Unit({}) / unit_1 == Unit({"N": -1})

    def test_define_unit_circular_reference(self):
        """Tests that an error is raised with circular unit definitions."""

        q.define_unit("A", "B*C")
        q.define_unit("B", "X*V/A")

        unit_1 = Unit({"A": 1})
        unit_2 = Unit({"B": 1})

        with pytest.raises(RecursionError, match="circular reference"):
            _ = unit_1 + unit_2
