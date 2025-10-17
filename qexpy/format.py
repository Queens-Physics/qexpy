"""Internal module for string formatting of numerical values and units."""

from fractions import Fraction

import numpy as np

from qexpy._config import options
from qexpy.typing import Number

_DOT_STRING = "\N{DOT OPERATOR}"
_PM = "+/-"
_TIMES = "×"


##########################
# Value/Error Formatting #
##########################


def format_value_error(value: float, error: float) -> str:
    """Format a value-error pair to a string."""

    if np.isclose(value, 0) and np.isclose(error, 0):
        return f"0 {_PM} 0"
    if np.isinf(value):
        return f"inf {_PM} inf"
    if np.isnan(value):
        return f"nan {_PM} nan"

    if not _is_valid(value) and not _is_valid(error):
        return f"{value} {_PM} {error}"

    sigfigs = options.format.precision.sigfigs
    if options.format.value == "scientific":
        return _format_value_scientific(value, error, sigfigs)

    return _format_value_simple(value, error, sigfigs)


def _format_value_simple(value: float, error: float, sigfigs: int) -> str:
    """Construct the string of a value-error pair in the simple format."""

    value, error = _round(value, error, sigfigs)
    dec = _num_decimals(value, error, sigfigs)

    value_str = f"{value:.{dec}f}"
    error_str = f"{error:.{dec}f}"
    return f"{value_str} {_PM} {error_str}"


def _format_value_scientific(value: float, error: float, sigfigs: int) -> str:
    """Format the value-error pair using the scientific notation."""

    basis = value if _is_valid(value) else error
    assert _is_valid(basis)
    order = int(np.floor(np.log10(abs(basis))))
    if order == 0:
        return _format_value_simple(value, error, sigfigs)

    value, error = _round(value, error, sigfigs)
    f_value = value / (10**order)
    f_error = error / (10**order)
    dec = _num_decimals(f_value, f_error, sigfigs)

    value_str = f"{f_value:.{dec}f}"
    error_str = f"{f_error:.{dec}f}"
    return f"({value_str} {_PM} {error_str}) {_TIMES} 10^{order}"


def _is_valid(number: float) -> bool:
    """Whether it makes sense to round the number."""
    return not (np.isinf(number) or np.isnan(number) or np.isclose(number, 0))


def _choose_round_basis(value: float, error: float) -> float:
    """Choose the number to perform rounding on."""

    if options.format.precision.mode == "value":
        return value if _is_valid(value) else error

    return error if _is_valid(error) else value


def _round(value: float, error: float, sigfigs: int) -> tuple[float, float]:
    """Round the value-error pair to the given number of sigfigs."""

    basis = _choose_round_basis(value, error)
    assert _is_valid(basis)

    order = np.floor(np.log10(abs(basis)))
    backoff = 10 ** (order - sigfigs + 1)

    value = np.round(value / backoff) * backoff
    error = np.round(error / backoff) * backoff
    return value, error


def _num_decimals(value: float, error: float, sigfigs: int) -> int:
    """Return the number of digits after the decimal point."""

    basis = _choose_round_basis(value, error)
    assert _is_valid(basis)

    order = np.floor(np.log10(abs(basis)))
    number_of_decimals = -order + sigfigs - 1
    return int(max(number_of_decimals, 0))


###################
# Unit Formatting #
###################


def format_unit_as_fraction(units: dict[str, Number]) -> str:
    """Format a unit dictionary to a string in the fraction form."""

    numerator = _DOT_STRING.join(
        f"{unit}{_exponent_to_str(exp)}" for unit, exp in units.items() if exp > 0
    )
    denominator = _DOT_STRING.join(
        f"{unit}{_exponent_to_str(-exp)}" for unit, exp in units.items() if exp < 0
    )

    if not denominator:
        return numerator

    numerator = numerator or "1"

    # If the denominator has multiplication, use brackets to avoid ambiguity
    if _DOT_STRING in denominator:
        denominator = f"({denominator})"

    # Combine the numerator and the denominator
    return f"{numerator}/{denominator}"


def format_unit_as_product(units: dict[str, Number]) -> str:
    """Format a unit dictioanry to a string in the product form."""

    return _DOT_STRING.join(
        f"{unit}{_exponent_to_str(exp)}" for unit, exp in units.items()
    )


def _number_of_decimals(value: float) -> int:
    """Return the number of decimal places for two significant figures."""

    order = int(np.floor(np.log10(abs(value) % 1)))
    number_of_decimals = -order + 2 - 1
    return number_of_decimals if number_of_decimals > 0 else 0


def _exponent_to_str(exponent: Number) -> str:
    """Return the exponent part of a unit."""

    # The exponent should be represented as a fraction if possible
    # to avoid running into the machine epsilon error.
    fraction = Fraction(exponent).limit_denominator()

    # Construct the string representation of the exponent
    if fraction == 1:
        return ""  # do not print power of 1 as it's implied
    if fraction.denominator == 1:
        return f"^{str(fraction.numerator)}"

    # When the fraction form is too complicated, keep the decimal form, and
    # keep two significant figures after the decimal point for simplicity.
    if fraction.denominator > 10:
        return f"^{exponent:.{_number_of_decimals(float(exponent))}f}"

    return f"^({str(fraction)})"
