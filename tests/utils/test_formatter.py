"""Tests for formatting value-error strings"""

# pylint: disable=missing-function-docstring

import pytest
import numpy as np
import qexpy as q
from qexpy.utils.formatter import format_value_error


class TestFormatter:
    """Tests for formatting value-error strings"""

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, "0 +/- 0"),
            (np.inf, 0.0, "inf +/- inf"),
            (1.0, np.inf, "1 +/- inf"),
            (-2, 1, "-2 +/- 1"),
            (2123, 13, "2120 +/- 10"),
            (2.1, 0.5, "2.1 +/- 0.5"),
            (2.12, 0.18, "2.1 +/- 0.2"),
        ],
    )
    def test_default_error_1(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "default",
            "format.precision.sig_fig",
            1,
            "format.precision.mode",
            "error",
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, "0 +/- 0"),
            (-2, 1, "-2.0 +/- 1.0"),
            (2, 0, "2.0 +/- 0.0"),
            (2.1, 0.5, "2.10 +/- 0.50"),
            (2.12, 0.22, "2.12 +/- 0.22"),
            (2.123, 0.123, "2.12 +/- 0.12"),
        ],
    )
    def test_default_error_2(self, value, error, expected):
        with q.option_context(
            "format.style.value",
            "default",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "error",
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, "0 +/- 0"),
            (-2, 1, "-2.0 +/- 1.0"),
            (0, 0.5, "0.00 +/- 0.50"),
            (1231, 0.5, "1200 +/- 0"),
            (123, 12, "120 +/- 10"),
        ],
    )
    def test_default_value_2(self, value, error, expected):
        with q.option_context(
            "format.style.value",
            "default",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "value",
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, r"0 \pm 0"),
            (np.inf, 0.0, r"inf \pm inf"),
            (-2, 1, r"-2 \pm 1"),
            (2123, 13, r"2120 \pm 10"),
            (2.1, 0.5, r"2.1 \pm 0.5"),
            (2.12, 0.18, r"2.1 \pm 0.2"),
        ],
    )
    def test_default_latex_error_1(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "default",
            "format.precision.sig_fig",
            1,
            "format.precision.mode",
            "error",
            "format.style.latex",
            True,
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, r"0 \pm 0"),
            (-2, 1, r"-2.0 \pm 1.0"),
            (2, 0, r"2.0 \pm 0.0"),
            (2.1, 0.5, r"2.10 \pm 0.50"),
            (2.12, 0.22, r"2.12 \pm 0.22"),
            (2.123, 0.123, r"2.12 \pm 0.12"),
        ],
    )
    def test_default_latex_error_2(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "default",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "error",
            "format.style.latex",
            True,
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, r"0 \pm 0"),
            (-2, 1, r"-2.0 \pm 1.0"),
            (0, 0.5, r"0.00 \pm 0.50"),
            (1231, 0.5, r"1200 \pm 0"),
            (123, 12, r"120 \pm 10"),
        ],
    )
    def test_default_latex_value_2(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "default",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "value",
            "format.style.latex",
            True,
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, "0 +/- 0"),
            (np.inf, 0.0, "inf +/- inf"),
            (-2.1, 0.5, "-2.1 +/- 0.5"),
            (2.12, 0.18, "2.1 +/- 0.2"),
            (2123, 13, "(2.12 +/- 0.01) * 10^3"),
            (0.012312, 0.00334, "(1.2 +/- 0.3) * 10^-2"),
            (120000, 370, "(1.200 +/- 0.004) * 10^5"),
            (100, 500, "(1 +/- 5) * 10^2"),
        ],
    )
    def test_scientific_error_1(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "scientific",
            "format.precision.sig_fig",
            1,
            "format.precision.mode",
            "error",
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, "0 +/- 0"),
            (-2.1, 0.5, "-2.10 +/- 0.50"),
            (2.12, 0.18, "2.12 +/- 0.18"),
            (2123, 13, "(2.123 +/- 0.013) * 10^3"),
            (0.012312, 0.00334, "(1.23 +/- 0.33) * 10^-2"),
            (120000, 370, "(1.2000 +/- 0.0037) * 10^5"),
        ],
    )
    def test_scientific_error_2(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "scientific",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "error",
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, "0 +/- 0"),
            (-2.1, 0.5, "-2.1 +/- 0.5"),
            (2.12, 0.18, "2.1 +/- 0.2"),
            (2123, 13, "(2.1 +/- 0.0) * 10^3"),
            (0.012312, 0.00334, "(1.2 +/- 0.3) * 10^-2"),
            (120000, 370, "(1.2 +/- 0.0) * 10^5"),
        ],
    )
    def test_scientific_value_2(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "scientific",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "value",
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, r"0 \pm 0"),
            (np.inf, 0.0, r"inf \pm inf"),
            (-2.1, 0.5, r"-2.1 \pm 0.5"),
            (2.12, 0.18, r"2.1 \pm 0.2"),
            (2123, 13, r"(2.12 \pm 0.01) \times 10^{3}"),
            (0.012312, 0.00334, r"(1.2 \pm 0.3) \times 10^{-2}"),
            (120000, 370, r"(1.200 \pm 0.004) \times 10^{5}"),
            (100, 500, r"(1 \pm 5) \times 10^{2}"),
        ],
    )
    def test_scientific_latex_error_1(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "scientific",
            "format.precision.sig_fig",
            1,
            "format.precision.mode",
            "error",
            "format.style.latex",
            True,
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, r"0 \pm 0"),
            (-2.1, 0.5, r"-2.10 \pm 0.50"),
            (2.12, 0.18, r"2.12 \pm 0.18"),
            (2123, 13, r"(2.123 \pm 0.013) \times 10^{3}"),
            (0.012312, 0.00334, r"(1.23 \pm 0.33) \times 10^{-2}"),
            (120000, 370, r"(1.2000 \pm 0.0037) \times 10^{5}"),
        ],
    )
    def test_scientific_latex_error_2(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "scientific",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "error",
            "format.style.latex",
            True,
        ):
            assert format_value_error(value, error) == expected

    @pytest.mark.parametrize(
        "value, error, expected",
        [
            (0.0, 0.0, r"0 \pm 0"),
            (-2.1, 0.5, r"-2.1 \pm 0.5"),
            (2.12, 0.18, r"2.1 \pm 0.2"),
            (2123, 13, r"(2.1 \pm 0.0) \times 10^{3}"),
            (0.012312, 0.00334, r"(1.2 \pm 0.3) \times 10^{-2}"),
            (120000, 370, r"(1.2 \pm 0.0) \times 10^{5}"),
        ],
    )
    def test_scientific_latex_value_2(self, value, error, expected):

        with q.option_context(
            "format.style.value",
            "scientific",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "value",
            "format.style.latex",
            True,
        ):
            assert format_value_error(value, error) == expected
