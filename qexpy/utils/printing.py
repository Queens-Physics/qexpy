import math as m
from qexpy.settings.settings import PrintStyle, get_print_style
from qexpy.settings.settings import get_sig_fig_mode, get_sig_fig_value, SigFigMode
from .utils import count_significant_figures


def _default_print(value, latex=False) -> str:
    """Prints out the value and uncertainty in its default format

    Args:
        value (tuple): the value-error pair to be printed
        latex (bool): if the value is to be printed in latex format

    Returns:
        The string representation of the value

    """
    if value[1] == 0:
        # if the uncertainty is 0, there's no need for further parsing unless there is
        # a requirement on the significant figures of the value
        if get_sig_fig_mode() == SigFigMode.VALUE:
            rounded_value, rounded_error = __round_values_to_sig_figs(value)
            return "{} +/- {}".format(rounded_value, rounded_error)
        return "{} +/- {}".format(value[0], value[1])

    # round the values based on significant digits
    rounded_value, rounded_error = __round_values_to_sig_figs(value)

    # check if the number of decimals matches the requirement of significant figures
    number_of_decimals = __find_number_of_decimals(rounded_value, rounded_error)

    # construct the string to return
    plus_minus_sign = "\pm" if latex else "+/-"
    if number_of_decimals == 0:
        return "{} {} {}".format(rounded_value, plus_minus_sign, rounded_error)
    else:
        return "{:.{num}f} {} {:.{num}f}".format(rounded_value, plus_minus_sign, rounded_error, num=number_of_decimals)


def _latex_print(value) -> str:
    """Prints out the value and uncertainty in latex format"""
    return _scientific_print(value, latex=True)


def _scientific_print(value, latex=False) -> str:
    """Prints out the value and uncertainty in scientific notation

    Args:
        value (tuple): the value-error pair to be printed
        latex (bool): if the value is to be printed in latex format

    Returns:
        The string representation of the value

    """
    # find order of magnitude
    order_of_value = m.floor(m.log10(value[0]))
    if order_of_value == 0:
        return _default_print(value, latex)

    # round the values based on significant digits
    rounded_value, rounded_error = __round_values_to_sig_figs(value)

    # convert to scientific notation
    converted_value = rounded_value / (10 ** order_of_value)
    converted_error = rounded_error / (10 ** order_of_value)

    # check if the number of decimals matches the requirement of significant figures
    number_of_decimals = __find_number_of_decimals(converted_value, converted_error)

    # construct the string to return
    plus_minus_sign = "\pm" if latex else "+/-"
    if number_of_decimals == 0:
        return "({} {} {}) * 10^{}".format(rounded_value, plus_minus_sign, rounded_error, order_of_value)
    else:
        return "({:.{num}f} {} {:.{num}f}) * 10^{}".format(converted_value, plus_minus_sign, converted_error,
                                                           order_of_value, num=number_of_decimals)


def __round_values_to_sig_figs(value) -> tuple:
    """Rounds the value and uncertainty based on sig-fig settings

    This method works by first finding the order of magnitude for the error, or
    the value, depending on the significant figure settings. It calculates a value
    called back-off, which is a helper value for rounding.

    For example, to round 12345 to 3 significant figures, log10(12345) would return
    4, which is the order of magnitude of the number. The formula for the back-off
    is order_of_magnitude - significant_digits + 1. In this case, the back-off
    would be 4 - 3 + 1 = 2. With the back-off, we first divide 12345 by 10^2, which
    results in 123.45, then round it to 123, before re-multiplying the back-off,
    which would result in 12300

    Args:
        value (tuple): the value-error pair to be rounded

    Returns:
        the rounded results for this pair

    """

    sig_fig_mode = get_sig_fig_mode()
    sig_fig_value = get_sig_fig_value()

    # first find the back-off value for rounding
    if sig_fig_mode == SigFigMode.AUTOMATIC or sig_fig_mode == SigFigMode.ERROR:
        order_of_error = m.floor(m.log10(value[1]))
        back_off = 10 ** (order_of_error - sig_fig_value + 1)
    else:
        order_of_value = m.floor(m.log10(value[0]))
        back_off = 10 ** (order_of_value - sig_fig_value + 1)

    # then round the value and error to the same digit
    if back_off < 1:
        # This weird extra condition is added because sometimes in Python, when you do
        # operations with a small floating point number, the result gets a little funky
        # For example, 1200 * 0.0001 will return 0.12000000000000001, but 12 / 10000
        # will return 0.12 as expected.
        #
        # The reason for this has to do with the way floating point numbers are stored
        # in Python, see https://docs.python.org/3.4/tutorial/floatingpoint.html for a more
        # elaborate explanation. The following hack avoids multiplication with a small
        # floating point number by replacing with with regular division
        rounded_error = round(value[1] / back_off) / (1 / back_off)
        rounded_value = round(value[0] / back_off) / (1 / back_off)
    else:
        rounded_error = round(value[1] / back_off) * back_off
        rounded_value = round(value[0] / back_off) * back_off

    # return the two rounded values as a tuple
    return rounded_value, rounded_error


def __find_number_of_decimals(value, error) -> int:
    """Finds the correct number of decimal places to show for a value-error pair"""

    sig_fig_mode = get_sig_fig_mode()
    sig_fig_value = get_sig_fig_value()

    error_number_of_decimals = __count_number_of_decimals(error)
    value_number_of_decimals = __count_number_of_decimals(value)
    raw_number_of_decimals = max(error_number_of_decimals, value_number_of_decimals)

    # check if the current number of significant figures satisfy the settings
    if sig_fig_mode == SigFigMode.AUTOMATIC or sig_fig_mode == SigFigMode.ERROR:
        current_sig_figs_of_error = count_significant_figures(error)
        if current_sig_figs_of_error < sig_fig_value:
            return raw_number_of_decimals + (sig_fig_value - current_sig_figs_of_error)
    else:
        current_sig_figs_of_value = count_significant_figures(value)
        if current_sig_figs_of_value < sig_fig_value:
            return raw_number_of_decimals + (sig_fig_value - current_sig_figs_of_value)

    return raw_number_of_decimals


def __count_number_of_decimals(number) -> int:
    string_repr_of_value = str(number)
    if "." not in string_repr_of_value:
        return 0
    decimal_part = string_repr_of_value.split(".")[1]
    return len(decimal_part)


def get_printer(print_style=get_print_style()):
    """Gets the printer function for the given print style

    This method will use the global setting for print style if none is specified

    Args:
        print_style (PrintStyle): the print style

    Returns:
        the printer function

    """
    if not isinstance(print_style, PrintStyle):
        raise ValueError("Error: the print styles supported are default, latex, and scientific.\n"
                         "These values are found under the enum settings.PrintStyle")
    if print_style == PrintStyle.DEFAULT:
        return _default_print
    elif print_style == PrintStyle.LATEX:
        return _latex_print
    else:
        return _scientific_print
