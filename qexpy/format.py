"""Internal module for string formatting of numerical values and units."""

from fractions import Fraction

import numpy as np

from qexpy.typing import Number

_DOT_STRING = "\N{DOT OPERATOR}"


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


def unit_to_fraction_string(units: dict[str, Number]) -> str:
    """Format a unit dictionary to a string in the fraction form."""

    numerator = _DOT_STRING.join(
        f"{unit}{_exponent_to_str(exp)}"
        for unit, exp in units.items()
        if exp > 0
    )
    denomiator = _DOT_STRING.join(
        f"{unit}{_exponent_to_str(-exp)}"
        for unit, exp in units.items()
        if exp < 0
    )

    if not denomiator:
        return numerator

    numerator = numerator or "1"

    # If the denominator has multiplication, use brackets to avoid ambiguity
    if _DOT_STRING in denomiator:
        denomiator = f"({denomiator})"

    # Combine the numerator and the denominator
    return f"{numerator}/{denomiator}"


def unit_to_product_string(units: dict[str, Number]) -> str:
    """Format a unit dictioanry to a string in the product form."""

    return _DOT_STRING.join(
        f"{unit}{_exponent_to_str(exp)}" for unit, exp in units.items()
    )
