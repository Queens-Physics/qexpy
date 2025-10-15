"""Unit tests for the internal format module."""

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
