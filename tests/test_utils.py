"""Test for utility methods"""

import pytest

import numpy as np
import qexpy as q
import qexpy.settings.literals as lit
import qexpy.utils as utils


class TestUtils:
    """Tests the utility functions in module qexpy.utils.utils"""

    def test_load_data_from_file(self):  # pylint: disable=no-self-use
        """Tests function that loads data from a file"""
        data = utils.load_data_from_file("./resources/data_for_test_load_data.csv")
        assert len(data) == 4
        for data_set in data:
            assert len(data_set) == 30

    def test_numerical_derivative(self):  # pylint: disable=no-self-use
        """Tests the numerical derivative functionality"""
        func = lambda x: x ** 2 + np.sin(x)
        assert pytest.approx(utils.numerical_derivative(func, q.pi), 2 * q.pi - 1)

    def test_calculate_covariance(self):  # pylint: disable=no-self-use
        """Tests the covariance calculation functionality"""
        arr_x = [1.2, 3.2, 2.3, 1.2, 1.9]
        arr_y = [2.2, 1.2, 2.5, 2.6, 1.0]
        assert pytest.approx(utils.calculate_covariance(arr_x, arr_y), np.cov(arr_x, arr_y))


@pytest.fixture()
def resource():
    test_units = {
        "units_for_watt": {"kg": 1, "m": 2, "s": -2},
        "units_for_henry": {"kg": 1, "m": 2, "s": -2, "A": -2},
        "units_super_complicated": {"kg": 4, "m": 2, "L": -3, "Pa": 1, "s": -2, "A": -2},
        "units_numerator_only": {"kg": 1, "m": 2},
        "units_denominator_only": {"kg": -1, "s": -2}
    }
    yield test_units


class TestUnits:
    """Tests unit parsing, constructing unit strings, and unit operations"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):  # pylint: disable=no-self-use
        """Before method that resets configurations"""
        q.get_settings().reset()

    def test_parse_units(self, resource):
        """Tests parsing unit strings into dictionaries"""

        units_for_watt = resource["units_for_watt"]
        assert utils.parse_units("kg*m^2/s^2") == units_for_watt
        assert utils.parse_units("kg^1m^2s^-2") == units_for_watt
        assert utils.parse_units("kg^1*m^2/s^2") == units_for_watt
        assert utils.parse_units("kg^1m^2/s^2") == units_for_watt

        units_for_henry = resource["units_for_henry"]
        assert utils.parse_units("kg*m^2/(s^2*A^2)") == units_for_henry
        assert utils.parse_units("kg^1m^2s^-2A^-2") == units_for_henry
        assert utils.parse_units("m^2kg/s^2A^2") == units_for_henry
        assert utils.parse_units("kg/s^2A^2*m^2") == units_for_henry

        units_super_complicated = resource["units_super_complicated"]
        assert utils.parse_units("m^2kg^4/s^2A^2L^3*Pa") == units_super_complicated
        assert utils.parse_units("kg^4m^2*Pa/(s^2A^2*L^3)") == units_super_complicated

    def test_construct_unit_string(self, resource):
        """Tests constructing unit strings from dictionaries"""

        # For exponent style printing
        assert utils.construct_unit_string(resource["units_for_watt"]) == "kg⋅m^2⋅s^-2"
        assert utils.construct_unit_string(resource["units_for_henry"]) == "kg⋅m^2⋅s^-2⋅A^-2"
        assert utils.construct_unit_string(resource["units_numerator_only"]) == "kg⋅m^2"
        assert utils.construct_unit_string(resource["units_denominator_only"]) == "kg^-1⋅s^-2"

        # For fraction style printing
        q.get_settings().unit_style = q.UnitStyle.FRACTION
        assert utils.construct_unit_string(resource["units_for_watt"]) == "kg⋅m^2/s^2"
        assert utils.construct_unit_string(resource["units_for_henry"]) == "kg⋅m^2/(s^2⋅A^2)"
        assert utils.construct_unit_string(resource["units_numerator_only"]) == "kg⋅m^2"
        assert utils.construct_unit_string(resource["units_denominator_only"]) == "1/(kg⋅s^2)"

    def test_unit_operations(self, resource):
        """Tests operating with units"""

        test_unit = resource["units_for_watt"]
        assert utils.operate_with_units(lit.ADD, test_unit, test_unit) == test_unit

        test_unit_2 = resource["units_for_henry"]
        assert utils.operate_with_units(lit.MUL, test_unit, test_unit_2) == {
            "kg": 2, "m": 4, "s": -4, "A": -2}
        assert utils.operate_with_units(lit.DIV, test_unit, test_unit_2) == {"A": 2}


class TestPrinter:
    """Tests functions that displays value-error pairs"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):  # pylint: disable=no-self-use
        """Before method that resets all configurations"""
        q.get_settings().reset()

    def test_default_print(self):  # pylint: disable=no-self-use
        """Tests the default print format"""

        # Printing in default format
        default_printer = utils.get_printer(q.PrintStyle.DEFAULT)
        assert default_printer(2, 1) == "2 +/- 1"
        assert default_printer(2123, 13) == "2120 +/- 10"
        assert default_printer(2.1, 0.5) == "2.1 +/- 0.5"
        assert default_printer(2.12, 0.18) == "2.1 +/- 0.2"

        # Printing with significant figure specified for error
        q.set_sig_figs_for_error(2)
        assert default_printer(2, 1) == "2.0 +/- 1.0"
        assert default_printer(2.1, 0.5) == "2.10 +/- 0.50"
        assert default_printer(2.12, 0.22) == "2.12 +/- 0.22"
        assert default_printer(2.123, 0.123) == "2.12 +/- 0.12"

        # Printing with significant figure specified for value
        q.set_sig_figs_for_value(2)
        assert default_printer(2, 1) == "2.0 +/- 1.0"
        assert default_printer(1231, 0.5) == "1200 +/- 0"
        assert default_printer(123, 12) == "120 +/- 10"

    def test_scientific_print(self):  # pylint: disable=no-self-use
        """Tests printing in scientific notation"""

        # Printing in default format
        scientific_printer = utils.get_printer(q.PrintStyle.SCIENTIFIC)
        assert scientific_printer(2.1, 0.5) == "2.1 +/- 0.5"
        assert scientific_printer(2.12, 0.18) == "2.1 +/- 0.2"
        assert scientific_printer(2123, 13) == "(2.12 +/- 0.01) * 10^3"
        assert scientific_printer(0.012312, 0.00334) == "(1.2 +/- 0.3) * 10^-2"
        assert scientific_printer(120000, 370) == "(1.200 +/- 0.004) * 10^5"

        # Printing with significant figure specified for error
        q.set_sig_figs_for_error(2)
        assert scientific_printer(2.1, 0.5) == "2.10 +/- 0.50"
        assert scientific_printer(2.12, 0.18) == "2.12 +/- 0.18"
        assert scientific_printer(2123, 13) == "(2.123 +/- 0.013) * 10^3"
        assert scientific_printer(0.012312, 0.00334) == "(1.23 +/- 0.33) * 10^-2"
        assert scientific_printer(120000, 370) == "(1.2000 +/- 0.0037) * 10^5"

        # Printing with significant figure specified for value
        q.set_sig_figs_for_value(2)
        assert scientific_printer(2.1, 0.5) == "2.1 +/- 0.5"
        assert scientific_printer(2.12, 0.18) == "2.1 +/- 0.2"
        assert scientific_printer(2123, 13) == "(2.1 +/- 0.0) * 10^3"
        assert scientific_printer(0.012312, 0.00334) == "(1.2 +/- 0.3) * 10^-2"
        assert scientific_printer(120000, 370) == "(1.2 +/- 0.0) * 10^5"

    def test_latex_print(self):  # pylint: disable=no-self-use
        """Test printing in latex format"""

        latex_printer = utils.get_printer(q.PrintStyle.LATEX)
        # Printing in default format
        assert latex_printer(2.1, 0.5) == r"2.1 \pm 0.5"
        assert latex_printer(2.12, 0.18) == r"2.1 \pm 0.2"
        assert latex_printer(2123, 13) == r"(2.12 \pm 0.01) * 10^3"
        assert latex_printer(0.012312, 0.00334) == r"(1.2 \pm 0.3) * 10^-2"
        assert latex_printer(120000, 370) == r"(1.200 \pm 0.004) * 10^5"

        # Printing with significant figure specified for error
        q.set_sig_figs_for_error(2)
        assert latex_printer(2.1, 0.5) == r"2.10 \pm 0.50"
        assert latex_printer(2.12, 0.18) == r"2.12 \pm 0.18"
        assert latex_printer(2123, 13) == r"(2.123 \pm 0.013) * 10^3"
        assert latex_printer(0.012312, 0.00334) == r"(1.23 \pm 0.33) * 10^-2"
        assert latex_printer(120000, 370) == r"(1.2000 \pm 0.0037) * 10^5"

        # Printing with significant figure specified for value
        q.set_sig_figs_for_value(2)
        assert latex_printer(2.1, 0.5) == r"2.1 \pm 0.5"
        assert latex_printer(2.12, 0.18) == r"2.1 \pm 0.2"
        assert latex_printer(2123, 13) == r"(2.1 \pm 0.0) * 10^3"
        assert latex_printer(0.012312, 0.00334) == r"(1.2 \pm 0.3) * 10^-2"
        assert latex_printer(120000, 370) == r"(1.2 \pm 0.0) * 10^5"
