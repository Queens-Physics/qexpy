"""Tests for utility methods

This file contains test cases for methods in sub-package utils, which contains regular
utility methods as well as value printing methods

"""

# noinspection PyPackageRequirements
import pytest

# noinspection PyProtectedMember
from qexpy.utils.printing import _default_print, _latex_print, _scientific_print
from qexpy.utils.utils import count_significant_figures, load_data_from_file
from qexpy.settings import set_sig_figs_for_error, set_sig_figs_for_value, reset_default_configuration


class TestUtils:

    def test_count_sig_figs(self):
        assert count_significant_figures(1.23) == 3
        assert count_significant_figures(243) == 3
        assert count_significant_figures(2.101) == 4
        assert count_significant_figures("3.100") == 4
        assert count_significant_figures("0012.1") == 3

    def test_load_data_from_file(self):
        data = load_data_from_file("./resources/sample_data.csv")
        assert len(data) == 4
        for data_set in data:
            assert len(data_set) == 30
            assert isinstance(data_set[0], float)


class TestPrinter:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        reset_default_configuration()

    def test_default_print(self):
        # printing in default format
        assert _default_print((2, 1)) == "2 +/- 1"
        assert _default_print((2123, 13)) == "2120 +/- 10"
        assert _default_print((2.1, 0.5)) == "2.1 +/- 0.5"
        assert _default_print((2.12, 0.18)) == "2.1 +/- 0.2"
        # printing with significant figure specified for error
        set_sig_figs_for_error(2)
        assert _default_print((2, 1)) == "2.0 +/- 1.0"
        assert _default_print((2.1, 0.5)) == "2.10 +/- 0.50"
        assert _default_print((2.12, 0.22)) == "2.12 +/- 0.22"
        assert _default_print((2.123, 0.123)) == "2.12 +/- 0.12"
        # printing with significant figure specified for value
        set_sig_figs_for_value(2)
        assert _default_print((2, 1)) == "2.0 +/- 1.0"
        assert _default_print((1231, 0.5)) == "1200 +/- 0"
        assert _default_print((123, 12)) == "120 +/- 10"

    def test_scientific_print(self):
        # printing in default format
        assert _scientific_print((2.1, 0.5)) == "2.1 +/- 0.5"
        assert _scientific_print((2.12, 0.18)) == "2.1 +/- 0.2"
        assert _scientific_print((2123, 13)) == "(2.12 +/- 0.01) * 10^3"
        assert _scientific_print((0.012312, 0.00334)) == "(1.2 +/- 0.3) * 10^-2"
        assert _scientific_print((120000, 370)) == "(1.200 +/- 0.004) * 10^5"
        # printing with significant figure specified for error
        set_sig_figs_for_error(2)
        assert _scientific_print((2.1, 0.5)) == "2.10 +/- 0.50"
        assert _scientific_print((2.12, 0.18)) == "2.12 +/- 0.18"
        assert _scientific_print((2123, 13)) == "(2.123 +/- 0.013) * 10^3"
        assert _scientific_print((0.012312, 0.00334)) == "(1.23 +/- 0.33) * 10^-2"
        assert _scientific_print((120000, 370)) == "(1.2000 +/- 0.0037) * 10^5"
        # printing with significant figure specified for value
        set_sig_figs_for_value(2)
        assert _scientific_print((2.1, 0.5)) == "2.1 +/- 0.5"
        assert _scientific_print((2.12, 0.18)) == "2.1 +/- 0.2"
        assert _scientific_print((2123, 13)) == "(2.1 +/- 0.0) * 10^3"
        assert _scientific_print((0.012312, 0.00334)) == "(1.2 +/- 0.3) * 10^-2"
        assert _scientific_print((120000, 370)) == "(1.2 +/- 0.0) * 10^5"

    def test_latex_print(self):
        # printing in default format
        assert _latex_print((2.1, 0.5)) == r"2.1 \pm 0.5"
        assert _latex_print((2.12, 0.18)) == r"2.1 \pm 0.2"
        assert _latex_print((2123, 13)) == r"(2.12 \pm 0.01) * 10^3"
        assert _latex_print((0.012312, 0.00334)) == r"(1.2 \pm 0.3) * 10^-2"
        assert _latex_print((120000, 370)) == r"(1.200 \pm 0.004) * 10^5"
        # printing with significant figure specified for error
        set_sig_figs_for_error(2)
        assert _latex_print((2.1, 0.5)) == r"2.10 \pm 0.50"
        assert _latex_print((2.12, 0.18)) == r"2.12 \pm 0.18"
        assert _latex_print((2123, 13)) == r"(2.123 \pm 0.013) * 10^3"
        assert _latex_print((0.012312, 0.00334)) == r"(1.23 \pm 0.33) * 10^-2"
        assert _latex_print((120000, 370)) == r"(1.2000 \pm 0.0037) * 10^5"
        # printing with significant figure specified for value
        set_sig_figs_for_value(2)
        assert _latex_print((2.1, 0.5)) == r"2.1 \pm 0.5"
        assert _latex_print((2.12, 0.18)) == r"2.1 \pm 0.2"
        assert _latex_print((2123, 13)) == r"(2.1 \pm 0.0) * 10^3"
        assert _latex_print((0.012312, 0.00334)) == r"(1.2 \pm 0.3) * 10^-2"
        assert _latex_print((120000, 370)) == r"(1.2 \pm 0.0) * 10^5"
