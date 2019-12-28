"""Unit tests for the utility sub-package"""

import os
import pytest
import numpy as np

from collections import OrderedDict

import qexpy.settings.settings as sts
import qexpy.settings.literals as lit
import qexpy.utils.utils as utils
import qexpy.utils.printing as printing
import qexpy.utils.units as units

from qexpy.utils.exceptions import UndefinedOperationError


class TestDecorators:
    """Unit tests for various decorators in utils"""

    def test_check_operand_type(self):
        """test the operand type checker"""

        @utils.check_operand_type("test")
        def test_func(_):
            raise TypeError("test error")

        @utils.check_operand_type("+")
        def test_func2(_, __):
            raise TypeError("test error")

        with pytest.raises(UndefinedOperationError) as e:
            test_func('a')

        exp = "\"test\" is undefined with operands of type(s) 'str'. Expected: real numbers"
        assert str(e.value) == exp

        with pytest.raises(UndefinedOperationError) as e:
            test_func2('a', 1)
        exp = "\"+\" is undefined with operands of type(s) 'str' and 'int'. " \
              "Expected: real numbers"
        assert str(e.value) == exp

    def test_vectorize(self):
        """test the vectorize decorator"""

        @utils.vectorize
        def test_func(a):
            return a + 2

        assert test_func([1, 2, 3]) == [3, 4, 5]
        assert test_func(1) == 3
        assert all(test_func(np.array([1, 2, 3])) == [3, 4, 5])


class TestUtils:
    """Unit tests for the utils sub-module"""

    def test_validate_xrange(self):
        """tests the range validator"""

        with pytest.raises(TypeError):
            utils.validate_xrange(0)
        with pytest.raises(TypeError):
            utils.validate_xrange((0,))
        with pytest.raises(TypeError):
            utils.validate_xrange((0, '1'))
        with pytest.raises(ValueError):
            utils.validate_xrange((1, 0))
        assert utils.validate_xrange((10.5, 20.5))

    def test_numerical_derivative(self):
        """test the numerical derivative"""

        assert pytest.approx(1.9726023611141572335938, utils.numerical_derivative(
            lambda x: x ** 2 * np.sin(x), 2))

    def test_calculate_covariance(self):
        """test the covariance calculator"""

        with pytest.raises(ValueError):
            utils.calculate_covariance([1, 2, 3], [1, 2, 3, 4])

        assert pytest.approx(utils.calculate_covariance([1, 2, 3, 4], [4, 3, 2, 1]), - 5 / 3)
        assert pytest.approx(utils.calculate_covariance(
            np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])), - 5 / 3)

    def test_cov2corr(self):
        """test converting covariance matrix to correlation matrix"""

        m = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 2, 3]])
        assert utils.cov2corr(np.cov(m)) == pytest.approx(np.corrcoef(m))

    def test_load_data_from_file(self):
        """test loading an array from a data file"""

        curr_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(curr_path, "./resources/data_for_test_load_data.csv")
        data = utils.load_data_from_file(filename)
        assert len(data) == 4
        for data_set in data:
            assert len(data_set) == 30
        assert data[2, 8] == 9.95

    def test_find_mode_and_uncertainty(self):
        """test finding most probably value and uncertainty from distribution"""

        samples = np.random.normal(0, 1, 10000)
        n, bins = np.histogram(samples, bins=100)
        mode, error = utils.find_mode_and_uncertainty(n, bins, 0.68)
        assert mode == pytest.approx(0, abs=0.5)
        assert error == pytest.approx(1, abs=0.5)


class TestPrinting:
    """Unit tests for the printing sub-module"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        """Before method that resets all configurations"""
        sts.get_settings().reset()

    def test_default_print(self):
        """Tests the default print format"""

        # Printing in default format
        default_printer = printing.get_printer()
        assert default_printer(0.0, 0.0) == "0 +/- 0"
        assert default_printer(np.inf, 0.0) == "inf +/- inf"
        assert default_printer(2, 1) == "2 +/- 1"
        assert default_printer(2123, 13) == "2120 +/- 10"
        assert default_printer(2.1, 0.5) == "2.1 +/- 0.5"
        assert default_printer(2.12, 0.18) == "2.1 +/- 0.2"

        # Printing with significant figure specified for error
        sts.set_sig_figs_for_error(2)
        assert default_printer(0.0, 0.0) == "0 +/- 0"
        assert default_printer(2, 1) == "2.0 +/- 1.0"
        assert default_printer(2, 0) == "2.0 +/- 0"
        assert default_printer(2.1, 0.5) == "2.10 +/- 0.50"
        assert default_printer(2.12, 0.22) == "2.12 +/- 0.22"
        assert default_printer(2.123, 0.123) == "2.12 +/- 0.12"

        # Printing with significant figure specified for value
        sts.set_sig_figs_for_value(2)
        assert default_printer(0.0, 0.0) == "0 +/- 0"
        assert default_printer(2, 1) == "2.0 +/- 1.0"
        assert default_printer(0, 0.5) == "0.00 +/- 0.50"
        assert default_printer(1231, 0.5) == "1200 +/- 0"
        assert default_printer(123, 12) == "120 +/- 10"

    def test_scientific_print(self):
        """Tests printing in scientific notation"""

        # Printing in default format
        scientific_printer = printing.get_printer(sts.PrintStyle.SCIENTIFIC)
        assert scientific_printer(0.0, 0.0) == "0 +/- 0"
        assert scientific_printer(np.inf, 0.0) == "inf +/- inf"
        assert scientific_printer(2.1, 0.5) == "2.1 +/- 0.5"
        assert scientific_printer(2.12, 0.18) == "2.1 +/- 0.2"
        assert scientific_printer(2123, 13) == "(2.12 +/- 0.01) * 10^3"
        assert scientific_printer(0.012312, 0.00334) == "(1.2 +/- 0.3) * 10^-2"
        assert scientific_printer(120000, 370) == "(1.200 +/- 0.004) * 10^5"

        # Printing with significant figure specified for error
        sts.set_sig_figs_for_error(1)
        assert scientific_printer(100, 500) == "(1 +/- 5) * 10^2"

        sts.set_sig_figs_for_error(2)
        assert scientific_printer(0.0, 0.0) == "0 +/- 0"
        assert scientific_printer(2.1, 0.5) == "2.10 +/- 0.50"
        assert scientific_printer(2.12, 0.18) == "2.12 +/- 0.18"
        assert scientific_printer(2123, 13) == "(2.123 +/- 0.013) * 10^3"
        assert scientific_printer(0.012312, 0.00334) == "(1.23 +/- 0.33) * 10^-2"
        assert scientific_printer(120000, 370) == "(1.2000 +/- 0.0037) * 10^5"

        # Printing with significant figure specified for value
        sts.set_sig_figs_for_value(2)
        assert scientific_printer(0.0, 0.0) == "0 +/- 0"
        assert scientific_printer(2.1, 0.5) == "2.1 +/- 0.5"
        assert scientific_printer(2.12, 0.18) == "2.1 +/- 0.2"
        assert scientific_printer(2123, 13) == "(2.1 +/- 0.0) * 10^3"
        assert scientific_printer(0.012312, 0.00334) == "(1.2 +/- 0.3) * 10^-2"
        assert scientific_printer(120000, 370) == "(1.2 +/- 0.0) * 10^5"

    def test_latex_print(self):
        """Test printing in latex format"""

        latex_printer = printing.get_printer(sts.PrintStyle.LATEX)

        # Printing in default format
        assert latex_printer(2.1, 0.5) == r"2.1 \pm 0.5"
        assert latex_printer(2.12, 0.18) == r"2.1 \pm 0.2"
        assert latex_printer(2123, 13) == r"(2.12 \pm 0.01) * 10^3"
        assert latex_printer(0.012312, 0.00334) == r"(1.2 \pm 0.3) * 10^-2"
        assert latex_printer(120000, 370) == r"(1.200 \pm 0.004) * 10^5"

        # Printing with significant figure specified for error
        sts.set_sig_figs_for_error(2)
        assert latex_printer(2.1, 0.5) == r"2.10 \pm 0.50"
        assert latex_printer(2.12, 0.18) == r"2.12 \pm 0.18"
        assert latex_printer(2123, 13) == r"(2.123 \pm 0.013) * 10^3"
        assert latex_printer(0.012312, 0.00334) == r"(1.23 \pm 0.33) * 10^-2"
        assert latex_printer(120000, 370) == r"(1.2000 \pm 0.0037) * 10^5"

        # Printing with significant figure specified for value
        sts.set_sig_figs_for_value(2)
        assert latex_printer(2.1, 0.5) == r"2.1 \pm 0.5"
        assert latex_printer(2.12, 0.18) == r"2.1 \pm 0.2"
        assert latex_printer(2123, 13) == r"(2.1 \pm 0.0) * 10^3"
        assert latex_printer(0.012312, 0.00334) == r"(1.2 \pm 0.3) * 10^-2"
        assert latex_printer(120000, 370) == r"(1.2 \pm 0.0) * 10^5"


@pytest.fixture()
def resource():
    yield {
        "joule": OrderedDict([("kg", 1), ("m", 2), ("s", -2)]),
        "pascal": OrderedDict([("kg", 1), ("m", -1), ("s", -2)]),
        "coulomb": OrderedDict([("A", 1), ("s", 1)]),
        "random-denominator": OrderedDict([("A", -1), ("s", -1)]),
        "random-complicated": OrderedDict([
            ("kg", 4), ("m", 2), ("Pa", 1), ("L", -3), ("s", -2), ("A", -2)])
    }


class TestUnits:
    """Unit tests for the units sub-module"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        sts.get_settings().reset()

    def test_parse_unit_string(self, resource):
        """tests for parsing unit strings into dictionary objects"""

        joule = dict(resource["joule"])
        assert units.parse_unit_string("kg*m^2/s^2") == joule
        assert units.parse_unit_string("kg^1m^2s^-2") == joule

        pascal = dict(resource['pascal'])
        assert units.parse_unit_string("kg/(m*s^2)") == pascal
        assert units.parse_unit_string("kg/m^1s^2") == pascal
        assert units.parse_unit_string("kg^1m^-1s^-2") == pascal

        coulomb = dict(resource['coulomb'])
        assert units.parse_unit_string("A*s") == coulomb

        denominator = dict(resource['random-denominator'])
        assert units.parse_unit_string("A^-1s^-1") == denominator

        complicated = dict(resource['random-complicated'])
        assert units.parse_unit_string("kg^4m^2Pa^1L^-3s^-2A^-2") == complicated
        assert units.parse_unit_string("kg^4m^2Pa/L^3s^2A^2") == complicated
        assert units.parse_unit_string("(kg^4*m^2*Pa)/(L^3*s^2*A^2)") == complicated

        with pytest.raises(ValueError):
            units.parse_unit_string("m2kg4/A2")

    def test_construct_unit_string(self, resource):
        """tests for building a unit string from a dictionary object"""

        assert units.construct_unit_string(resource['joule']) == "kg⋅m^2⋅s^-2"
        assert units.construct_unit_string(resource['pascal']) == "kg⋅m^-1⋅s^-2"
        assert units.construct_unit_string(resource['coulomb']) == "A⋅s"
        assert units.construct_unit_string(resource['random-denominator']) == "A^-1⋅s^-1"
        assert units.construct_unit_string(
            resource['random-complicated']) == "kg^4⋅m^2⋅Pa⋅L^-3⋅s^-2⋅A^-2"

        sts.set_unit_style(sts.UnitStyle.FRACTION)
        assert units.construct_unit_string(resource['joule']) == "kg⋅m^2/s^2"
        assert units.construct_unit_string(resource['pascal']) == "kg/(m⋅s^2)"
        assert units.construct_unit_string(resource['coulomb']) == "A⋅s"
        assert units.construct_unit_string(resource['random-denominator']) == "1/(A⋅s)"
        assert units.construct_unit_string(
            resource['random-complicated']) == "kg^4⋅m^2⋅Pa/(L^3⋅s^2⋅A^2)"

    def test_unit_operations(self, resource):
        """tests for operating with unit propagation"""

        joule = resource['joule']
        assert units.operate_with_units(lit.NEG, joule) == {'kg': 1, 'm': 2, 's': -2}
        assert units.operate_with_units(lit.DIV, {}, joule) == {'kg': -1, 'm': -2, 's': 2}

        pascal = resource['pascal']
        assert units.operate_with_units(lit.ADD, pascal, pascal) == pascal
        assert units.operate_with_units(lit.ADD, {}, pascal) == pascal
        assert units.operate_with_units(lit.SUB, pascal, pascal) == pascal
        assert units.operate_with_units(lit.SUB, {}, pascal) == pascal

        with pytest.warns(UserWarning):
            assert units.operate_with_units(lit.ADD, pascal, joule) == {}

        assert units.operate_with_units(lit.MUL, pascal, joule) == {'kg': 2, 'm': 1, 's': -4}
        assert units.operate_with_units(lit.DIV, joule, pascal) == {'m': 3}
        assert units.operate_with_units(lit.SQRT, joule) == {'kg': 1 / 2, 'm': 1, 's': -1}
