"""Function that formats string representations of experimental values"""

from __future__ import annotations

from typing import Tuple

import math as m
import qexpy as q


def format_value_error(value: float, error: float) -> str:
    """Generates the string representation a value-error pair"""

    # Fetch global configurations for value formats
    value_format = q.options.format.style.value
    mode = q.options.format.precision.mode
    sig_figs = q.options.format.precision.sig_fig
    latex = q.options.format.style.latex

    # Call the appropriate function depending on the value format
    if value_format == "scientific":
        return _format_scientific(value, error, mode, sig_figs, latex)

    return _format_default(value, error, mode, sig_figs, latex)


def _format_default(value: float, error: float, mode: str, sig_figs: int, latex: bool = False):
    """Generates the string representation in the default format"""

    sign = r"\pm" if latex else "+/-"

    if value == 0 and error == 0:
        return f"0 {sign} 0"
    if m.isinf(value):
        return f"inf {sign} inf"

    # Round the values based on significant digits
    value, error = __round_to_sig_figs(value, error, mode, sig_figs)

    # Check if the number of decimals matches the significant figures
    dec = __find_number_of_decimals(value, error, mode, sig_figs)

    # Construct the string representation
    value_str = f"{value:.{dec}f}"
    error_str = f"{error:.{dec}f}"
    return f"{value_str} {sign} {error_str}"


def _format_scientific(value: float, error: float, mode: str, sig_figs: int, latex=False) -> str:
    """Formats the value and uncertainty in scientific notation"""

    pm = r"\pm" if latex else "+/-"
    times = r"\times" if latex else "*"

    if value == 0 and error == 0:
        return f"0 {pm} 0"
    if m.isinf(value):
        return f"inf {pm} inf"

    # Find order of magnitude
    order = m.floor(m.log10(abs(value)))
    if order == 0:
        return _format_default(value, error, mode, sig_figs, latex)

    # Round the values based on significant digits
    value, error = __round_to_sig_figs(value, error, mode, sig_figs)

    # Convert to scientific notation
    f_value = value / (10**order)
    f_error = error / (10**order)

    # Check if the number of decimals matches the significant figures
    dec = __find_number_of_decimals(f_value, f_error, mode, sig_figs)

    # Construct the string to return
    order_str = f"{{{order}}}" if latex else f"{order}"
    value_str = f"{f_value:.{dec}f}"
    error_str = f"{f_error:.{dec}f}"
    return f"({value_str} {pm} {error_str}) {times} 10^{order_str}"


def __round_to_sig_figs(value: float, error: float, mode: str, sig_figs: int) -> Tuple:
    """Rounds the value and error to the required number of significant digits

    This method works by first finding the order of magnitude for the error or the value
    depending on the required method of keeping significant figures, and calculates a value
    called the backoff. For example, to round 12345678 to 3 significant figures, we find
    log10(12345678) = 7, which is the order of magnitude of the number. The formula for the
    back-off is given by:

    backoff = order_of_magnitude - significant_digits + 1.

    In this case, the back-off is 7 - 3 + 1 = 5. Then we divide 12345678 by 10^5, which gives
    123.45678, then round it to 123 before multiplying the backoff, which produces 12300

    Parameters
    ----------

    value : float
        The value of the quantity to be rounded
    error : float
        The uncertainty to be rounded
    mode : str
        Whether to enforce the number of significant figures on the value or the error
    sig_figs : int
        The number of significant figures to keep

    Returns
    -------

    the rounded results for this value-error pair

    """

    def is_valid(number):
        return not m.isinf(number) and not m.isnan(number) and number != 0

    # Check if any of the inputs are invalid for the following calculations
    if mode == "error" and not is_valid(error):
        return value, error  # do no rounding if the error is 0 or invalid
    if mode == "value" and not is_valid(value):
        return value, error  # do no rounding if the value is 0 or invalid

    # First find the back-off value for rounding
    base = abs(error) if mode == "error" else abs(value)
    order = m.floor(m.log10(base))
    backoff = 10 ** (order - sig_figs + 1)

    # Then round the value and error to the same digit
    rounded_error = round(error / backoff) * backoff
    rounded_value = round(value / backoff) * backoff

    # Return the two rounded values
    return rounded_value, rounded_error


def __find_number_of_decimals(value: float, error: float, mode: str, sig_figs: int) -> int:
    """Calculates the correct number of decimal places to show

    This method tweaks the rounded value and error to have the correct number of significant
    figures. For example, if the value of a variable is 5.001, and 3 significant figures is
    requested. After rounding, the value would become 5. If we want it to be displayed as 5.00,
    we need to find the number of digits after the decimal.

    The implementation is similar to that of the rounding algorithm defined in the method above.
    The key is to start counting significant figures from the most significant digit, by finding
    the order of magnitude of the value.

    """

    def is_valid(number):
        return not m.isinf(number) and not m.isnan(number) and number != 0

    # Check if the current number of significant figures satisfy the settings
    if is_valid(value) and (mode == "value" or not is_valid(error)):
        order = m.floor(m.log10(abs(value)))
    else:
        order = m.floor(m.log10(abs(error)))

    number_of_decimals = -order + sig_figs - 1
    return number_of_decimals if number_of_decimals > 0 else 0
