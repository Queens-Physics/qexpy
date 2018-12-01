"""Tests for utility methods

This file contains test cases for methods in sub-package utils, which contains regular
utility methods as well as value printing methods

"""

import pytest

import qexpy.utils.printing as printing
import qexpy.utils.units as units
import qexpy.utils.utils as utils
import qexpy.settings.settings as settings


class TestUtils:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        settings.reset_default_configuration()

    def test_count_sig_figs(self):
        assert utils.count_significant_figures(1.23) == 3
        assert utils.count_significant_figures(243) == 3
        assert utils.count_significant_figures(2.101) == 4
        assert utils.count_significant_figures("3.100") == 4
        assert utils.count_significant_figures("0012.1") == 3

    def test_load_data_from_file(self):
        data = utils.load_data_from_file("./resources/sample_data.csv")
        assert len(data) == 4
        for data_set in data:
            assert len(data_set) == 30
            assert isinstance(data_set[0], float)

    def test_construct_unit_string(self):
        units_for_newton = {
            "kg": 1,
            "m": 2,
            "s": -2
        }
        units_for_henry = {
            "kg": 1,
            "m": 2,
            "s": -2,
            "A": -2
        }
        units_for_idk_what = {
            "kg": 1,
            "m": 2
        }
        units_for_another_idk_what = {
            "kg": -1,
            "s": -2
        }
        # for exponent style printing
        assert units.construct_unit_string(units_for_newton) == "kg⋅m^2⋅s^-2"
        assert units.construct_unit_string(units_for_henry) == "kg⋅m^2⋅s^-2⋅A^-2"
        assert units.construct_unit_string(units_for_idk_what) == "kg⋅m^2"
        assert units.construct_unit_string(units_for_another_idk_what) == "kg^-1⋅s^-2"
        # for fraction style printing
        settings.set_unit_style(settings.UnitStyle.FRACTION)
        assert units.construct_unit_string(units_for_newton) == "kg⋅m^2/s^2"
        assert units.construct_unit_string(units_for_henry) == "kg⋅m^2/(s^2⋅A^2)"
        assert units.construct_unit_string(units_for_idk_what) == "kg⋅m^2"
        assert units.construct_unit_string(units_for_another_idk_what) == "1/(kg⋅s^2)"

    def test_parse_units(self):
        units_for_watt = {
            "kg": 1,
            "m": 2,
            "s": -2
        }
        assert units.parse_units("kg*m^2/s^2") == units_for_watt
        assert units.parse_units("kg^1m^2s^-2") == units_for_watt
        assert units.parse_units("kg^1*m^2/s^2") == units_for_watt
        assert units.parse_units("(kg^1m^2)/s^2") == units_for_watt
        units_for_henry = {
            "kg": 1,
            "m": 2,
            "s": -2,
            "A": -2
        }
        assert units.parse_units("kg*m^2/(s^2A^2)") == units_for_henry
        assert units.parse_units("kg^1m^2s^-2A^-2") == units_for_henry
        assert units.parse_units("(kg*m^2)/s^2A^2") == units_for_henry
        assert units.parse_units("kg/s^2A^2*m^2") == units_for_henry
        units_for_idk_what = {
            "kg": 4,
            "m": 2,
            "L": -3,
            "Pa": 1,
            "s": -2,
            "A": -2
        }
        assert units.parse_units("m^2kg^4/s^2A^2L^3*Pa") == units_for_idk_what
        assert units.parse_units("kg^4m^2*Pa/(s^2A^2*L^3)") == units_for_idk_what


class TestPrinter:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        settings.reset_default_configuration()

    def test_default_print(self):
        # printing in default format
        assert printing._default_print(2, 1) == "2 +/- 1"
        assert printing._default_print(2123, 13) == "2120 +/- 10"
        assert printing._default_print(2.1, 0.5) == "2.1 +/- 0.5"
        assert printing._default_print(2.12, 0.18) == "2.1 +/- 0.2"
        # printing with significant figure specified for error
        settings.set_sig_figs_for_error(2)
        assert printing._default_print(2, 1) == "2.0 +/- 1.0"
        assert printing._default_print(2.1, 0.5) == "2.10 +/- 0.50"
        assert printing._default_print(2.12, 0.22) == "2.12 +/- 0.22"
        assert printing._default_print(2.123, 0.123) == "2.12 +/- 0.12"
        # printing with significant figure specified for value
        settings.set_sig_figs_for_value(2)
        assert printing._default_print(2, 1) == "2.0 +/- 1.0"
        assert printing._default_print(1231, 0.5) == "1200 +/- 0"
        assert printing._default_print(123, 12) == "120 +/- 10"

    def test_scientific_print(self):
        # printing in default format
        assert printing._scientific_print(2.1, 0.5) == "2.1 +/- 0.5"
        assert printing._scientific_print(2.12, 0.18) == "2.1 +/- 0.2"
        assert printing._scientific_print(2123, 13) == "(2.12 +/- 0.01) * 10^3"
        assert printing._scientific_print(0.012312, 0.00334) == "(1.2 +/- 0.3) * 10^-2"
        assert printing._scientific_print(120000, 370) == "(1.200 +/- 0.004) * 10^5"
        # printing with significant figure specified for error
        settings.set_sig_figs_for_error(2)
        assert printing._scientific_print(2.1, 0.5) == "2.10 +/- 0.50"
        assert printing._scientific_print(2.12, 0.18) == "2.12 +/- 0.18"
        assert printing._scientific_print(2123, 13) == "(2.123 +/- 0.013) * 10^3"
        assert printing._scientific_print(0.012312, 0.00334) == "(1.23 +/- 0.33) * 10^-2"
        assert printing._scientific_print(120000, 370) == "(1.2000 +/- 0.0037) * 10^5"
        # printing with significant figure specified for value
        settings.set_sig_figs_for_value(2)
        assert printing._scientific_print(2.1, 0.5) == "2.1 +/- 0.5"
        assert printing._scientific_print(2.12, 0.18) == "2.1 +/- 0.2"
        assert printing._scientific_print(2123, 13) == "(2.1 +/- 0.0) * 10^3"
        assert printing._scientific_print(0.012312, 0.00334) == "(1.2 +/- 0.3) * 10^-2"
        assert printing._scientific_print(120000, 370) == "(1.2 +/- 0.0) * 10^5"

    def test_latex_print(self):
        # printing in default format
        assert printing._latex_print(2.1, 0.5) == r"2.1 \pm 0.5"
        assert printing._latex_print(2.12, 0.18) == r"2.1 \pm 0.2"
        assert printing._latex_print(2123, 13) == r"(2.12 \pm 0.01) * 10^3"
        assert printing._latex_print(0.012312, 0.00334) == r"(1.2 \pm 0.3) * 10^-2"
        assert printing._latex_print(120000, 370) == r"(1.200 \pm 0.004) * 10^5"
        # printing with significant figure specified for error
        settings.set_sig_figs_for_error(2)
        assert printing._latex_print(2.1, 0.5) == r"2.10 \pm 0.50"
        assert printing._latex_print(2.12, 0.18) == r"2.12 \pm 0.18"
        assert printing._latex_print(2123, 13) == r"(2.123 \pm 0.013) * 10^3"
        assert printing._latex_print(0.012312, 0.00334) == r"(1.23 \pm 0.33) * 10^-2"
        assert printing._latex_print(120000, 370) == r"(1.2000 \pm 0.0037) * 10^5"
        # printing with significant figure specified for value
        settings.set_sig_figs_for_value(2)
        assert printing._latex_print(2.1, 0.5) == r"2.1 \pm 0.5"
        assert printing._latex_print(2.12, 0.18) == r"2.1 \pm 0.2"
        assert printing._latex_print(2123, 13) == r"(2.1 \pm 0.0) * 10^3"
        assert printing._latex_print(0.012312, 0.00334) == r"(1.2 \pm 0.3) * 10^-2"
        assert printing._latex_print(120000, 370) == r"(1.2 \pm 0.0) * 10^5"
