"""Unit tests for unit parsing, constructing unit strings, and unit operations"""

import pytest

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


class TestUnits:
    """Tests for parsing and constructing unit strings"""

    @pytest.mark.parametrize("string, expected", STRINGS_TO_UNITS)
    def test_parse_unit_string(self, string, expected):
        """Tests parsing a unit string"""
        assert Unit.from_string(string) == expected
