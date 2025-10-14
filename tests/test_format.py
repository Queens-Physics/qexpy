"""Unit tests for the internal format module."""

import pytest

import qexpy as q
from qexpy.format import format_value_error


@pytest.mark.parametrize(
    "value, error, sigfigs, precision_mode, expected",
    [
        (1.23, 0.45, 1, "error", "1.2 +/- 0.4"),
        (1.23, 0.45, 3, "error", "1.230 +/- 0.450"),
        (12345, 123, 2, "error", "12340 +/- 120"),
        (-12.34, 0.0068, 1, "error", "-12.340 +/- 0.007"),
    ],
)
def test_format_simple(value, error, sigfigs, precision_mode, expected):
    """Tests constructing a string in the simple format."""

    with q.set_option_context(
        "format.value",
        "simple",
        "format.precision.sigfigs",
        sigfigs,
        "format.precision.mode",
        precision_mode,
    ):
        assert format_value_error(value, error) == expected
