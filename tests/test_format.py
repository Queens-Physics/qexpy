"""Unit tests for the internal format module."""

import numpy as np
import pytest

import qexpy as q
from qexpy.format import format_value_error


@pytest.mark.parametrize(
    "value, error, sigfigs, expected",
    [
        (1.23, 0.45, 1, "1.2 +/- 0.4"),
        (12345, 123, 2, "12340 +/- 120"),
        (1.23, 0.45, 3, "1.230 +/- 0.450"),
        (-12.34, 0.0068, 1, "-12.340 +/- 0.007"),
        (12.2345678, 0.04, 2, "12.235 +/- 0.040"),
        (0.0, 0.456, 2, "0.00 +/- 0.46"),
        (0.1234, 0.0, 2, "0.12 +/- 0.00"),
        (0.0, 0.0, 2, "0 +/- 0"),
        (np.inf, 0.12, 1, "inf +/- inf"),
        (1.23, np.inf, 2, "1.2 +/- inf"),
        (np.nan, 0.12, 2, "nan +/- nan"),
        (0.0, np.inf, 2, "0.0 +/- inf"),
    ],
)
def test_format_simple_with_error(value, error, sigfigs, expected):
    """Tests the simple format with error as the precision mode."""

    with q.set_option_context(
        "format.value",
        "simple",
        "format.precision.sigfigs",
        sigfigs,
        "format.precision.mode",
        "error",
    ):
        assert format_value_error(value, error) == expected


@pytest.mark.parametrize(
    "value, error, sigfigs, expected",
    [
        (1.23, 0.45, 2, "1.2 +/- 0.4"),
        (12345, 123, 3, "12300 +/- 100"),
        (1.23, 0.45, 4, "1.230 +/- 0.450"),
        (-12.34, 0.0068, 5, "-12.340 +/- 0.007"),
        (12.2345678, 0.04, 5, "12.235 +/- 0.040"),
        (0.0, 0.456, 2, "0.00 +/- 0.46"),
        (0.1234, 0.0, 2, "0.12 +/- 0.00"),
        (0.0, 0.0, 2, "0 +/- 0"),
        (np.inf, 0.12, 1, "inf +/- inf"),
        (1.23, np.inf, 2, "1.2 +/- inf"),
        (np.nan, 0.12, 2, "nan +/- nan"),
        (0.0, np.inf, 2, "0.0 +/- inf"),
    ],
)
def test_format_simple_with_value(value, error, sigfigs, expected):
    """Tests the simple format with value as the precision mode."""

    with q.set_option_context(
        "format.value",
        "simple",
        "format.precision.sigfigs",
        sigfigs,
        "format.precision.mode",
        "value",
    ):
        assert format_value_error(value, error) == expected


@pytest.mark.parametrize(
    "value, error, sigfigs, expected",
    [
        (1.23, 0.45, 1, "1.2 +/- 0.4"),
        (-1.23, 0.45, 1, "-1.2 +/- 0.4"),
        (12345, 230, 1, "(1.23 +/- 0.02) × 10^4"),
        (-12345, 230, 1, "(-1.23 +/- 0.02) × 10^4"),
        (0.00012345, 0.00000345, 2, "(1.234 +/- 0.034) × 10^-4"),
        (123, 23, 3, "(1.230 +/- 0.230) × 10^2"),
    ],
)
def test_format_scientific_with_error(value, error, sigfigs, expected):
    """Tests the scientific format with error as the precision mode."""

    with q.set_option_context(
        "format.value",
        "scientific",
        "format.precision.sigfigs",
        sigfigs,
        "format.precision.mode",
        "error",
    ):
        assert format_value_error(value, error) == expected


@pytest.mark.parametrize(
    "value, error, sigfigs, expected",
    [
        (1.23, 0.45, 2, "1.2 +/- 0.4"),
        (-1.23, 0.45, 2, "-1.2 +/- 0.4"),
        (1234, 230, 2, "(1.2 +/- 0.2) × 10^3"),
        (-1234, 230, 2, "(-1.2 +/- 0.2) × 10^3"),
        (0.00012345, 0.00000345, 3, "(1.23 +/- 0.03) × 10^-4"),
        (123, 23, 4, "(1.230 +/- 0.230) × 10^2"),
    ],
)
def test_format_scientific_with_value(value, error, sigfigs, expected):
    """Tests the scientific format with value as the precision mode."""

    with q.set_option_context(
        "format.value",
        "scientific",
        "format.precision.sigfigs",
        sigfigs,
        "format.precision.mode",
        "value",
    ):
        assert format_value_error(value, error) == expected
