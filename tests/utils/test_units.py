"""Unit tests for unit parsing, constructing unit strings, and unit operations"""

import pytest

import qexpy as q
from qexpy.utils import Unit

PREDEFINED = {
    "joule": {"kg": 1, "m": 2, "s": -2},
    "pascal": {"kg": 1, "m": -1, "s": -2},
    "coulomb": {"A": 1, "s": 1},
    "hello": {"A": -1, "s": -1},
    "world": {"kg": 4, "m": 2, "Pa": 1, "L": -3, "s": -2, "A": -2},
    "ugly": {"kg": 1 / 2, "m": 1 / 3, "s": -2.003123, "A": -0.02312},
}

STRINGS_TO_UNITS = [
    ("kg*m^2/s^2", PREDEFINED["joule"]),
    ("kg^1m^2s^-2", PREDEFINED["joule"]),
    ("kg/(m*s^2)", PREDEFINED["pascal"]),
    ("kg/m^1s^2", PREDEFINED["pascal"]),
    ("kg^1m^-1s^-2", PREDEFINED["pascal"]),
    ("A*s", PREDEFINED["coulomb"]),
    ("A^-1s^-1", PREDEFINED["hello"]),
    ("kg^4m^2Pa^1L^-3s^-2A^-2", PREDEFINED["world"]),
    ("kg^4m^2Pa/L^3s^2A^2", PREDEFINED["world"]),
    ("(kg^4*m^2*Pa)/(L^3*s^2*A^2)", PREDEFINED["world"]),
    ("kg^1/2m^(1/3)/s^(2.003123)A^0.02312", PREDEFINED["ugly"]),
]

UNITS_TO_STRINGS = [
    ({}, "", ""),
    (PREDEFINED["joule"], "kg⋅m^2⋅s^-2", "kg⋅m^2/s^2"),
    (PREDEFINED["pascal"], "kg⋅m^-1⋅s^-2", "kg/(m⋅s^2)"),
    (PREDEFINED["coulomb"], "A⋅s", "A⋅s"),
    (PREDEFINED["hello"], "A^-1⋅s^-1", "1/(A⋅s)"),
    (PREDEFINED["world"], "kg^4⋅m^2⋅Pa⋅L^-3⋅s^-2⋅A^-2", "kg^4⋅m^2⋅Pa/(L^3⋅s^2⋅A^2)"),
    (
        PREDEFINED["ugly"],
        "kg^(1/2)⋅m^(1/3)⋅s^-2.0031⋅A^-0.023",
        "kg^(1/2)⋅m^(1/3)/(s^2.0031⋅A^0.023)",
    ),
]


class TestUnits:
    """Tests for parsing and constructing unit strings"""

    @pytest.mark.parametrize("string, expected", STRINGS_TO_UNITS)
    def test_parse_unit_string(self, string, expected):
        """Tests parsing a unit string"""
        assert Unit.from_string(string) == expected

    @pytest.mark.parametrize("unit_dict, exp_str, frac_str", UNITS_TO_STRINGS)
    def test_construct_unit_string(self, unit_dict, exp_str, frac_str):
        """Tests constructing a unit string"""

        units = Unit(unit_dict)
        with q.option_context("format.style.unit", "fraction"):
            assert str(units) == frac_str
            assert repr(units) == frac_str
        with q.option_context("format.style.unit", "exponent"):
            assert str(units) == exp_str
            assert repr(units) == exp_str

    def test_invalid_string(self):
        """Tests invalid unit strings"""

        with pytest.raises(ValueError):
            Unit.from_string("m2kg4/A2")

        with pytest.raises(ValueError):
            q.define_unit("1", "m2kg4/A2")

    def test_define_unit(self):
        """Tests user defined units"""

        q.define_unit("N", "kg*m/s^2")
        assert q.utils.units._registered_units == {"N": {"kg": 1, "m": 1, "s": -2}}

        unit = Unit({"kg": 1, "m": 1, "s": -2})
        assert str(unit) == "N"

        unit = Unit({"kg": 2, "m": 2, "s": -4})
        assert str(unit) == "N^2"

        unit = Unit({"kg": -2, "m": -2, "s": 4})
        assert str(unit) == "1/N^2"

        unit = Unit({"kg": 2, "m": 2, "s": -3})
        assert str(unit) == "kg^2⋅m^2/s^3"

        unit = Unit({"kg": 2, "m": 2})
        assert str(unit) == "kg^2⋅m^2"

        unit = Unit({"kg": 2, "m": 2, "A": -2})
        assert str(unit) == "kg^2⋅m^2/A^2"

        q.clear_unit_definitions()
        assert q.utils.units._registered_units == {}
