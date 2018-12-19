"""Holds all global configuration variables and Enums for user options

This file contains configurations and flags for user settings including
plot settings and error propagation settings

"""

from enum import Enum
import numbers
from typing import Union
import qexpy.settings.literals as lit


class ErrorMethod(Enum):
    """Preferred method of error propagation"""
    DERIVATIVE = "derivative"
    MONTE_CARLO = "monte-carlo"


class PrintStyle(Enum):
    """Preferred format for the string representation of values"""
    DEFAULT = "default"
    LATEX = "latex"
    SCIENTIFIC = "scientific"


class UnitStyle(Enum):
    """Preferred format for the string representation of units"""
    FRACTION = "fraction"
    EXPONENTS = "exponents"


class SigFigMode(Enum):
    """Preferred method to choose number of significant figures"""
    AUTOMATIC = "automatic"
    VALUE = "value"
    ERROR = "error"


CONFIG = {
    lit.ERROR_METHOD: ErrorMethod.DERIVATIVE,
    lit.PRINT_STYLE: PrintStyle.DEFAULT,
    lit.UNIT_STYLE: UnitStyle.EXPONENTS,
    lit.SIG_FIGS: {
        lit.SIG_FIG_MODE: SigFigMode.AUTOMATIC,
        lit.SIG_FIG_VALUE: 1
    },
    lit.MONTE_CARLO_SAMPLE_SIZE: 10000
}


def reset_default_configuration():
    """Resets all configurations to their default values"""
    CONFIG[lit.ERROR_METHOD] = ErrorMethod.DERIVATIVE
    CONFIG[lit.PRINT_STYLE] = PrintStyle.DEFAULT
    CONFIG[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.AUTOMATIC
    CONFIG[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = 1
    CONFIG[lit.UNIT_STYLE] = UnitStyle.EXPONENTS
    CONFIG[lit.MONTE_CARLO_SAMPLE_SIZE] = 10000


def set_error_method(new_method: Union[ErrorMethod, str]):
    """Sets the preferred error method for values

    This sets the method for error propagation. Keep in mind that all three error methods
    are used to calculate values behind the scene. This method allows the user to choose
    which values to print out

    Args:
        new_method (ErrorMethod): the error method to be used

    """
    if isinstance(new_method, ErrorMethod):
        CONFIG[lit.ERROR_METHOD] = new_method
    elif isinstance(new_method, str) and new_method in [lit.MONTE_CARLO_PROPAGATED, lit.DERIVATIVE_PROPAGATED]:
        CONFIG[lit.ERROR_METHOD] = ErrorMethod(new_method)
    else:
        raise ValueError("The error methods supported are derivative, min-max, and monte carlo.\n"
                         "These values are found under the enum settings.ErrorMethod")


def get_error_method() -> ErrorMethod:
    """Gets the current choice of error propagation method"""
    return CONFIG[lit.ERROR_METHOD]


def set_print_style(new_style: Union[PrintStyle, str]):
    """Sets the format of value strings

    The three available formats are default, latex, and scientific, which can all
    be found in the enum settings.PrintStyle

    Args:
        new_style (PrintStyle): the choice of format

    """
    if isinstance(new_style, PrintStyle):
        CONFIG[lit.ERROR_METHOD] = new_style
    elif isinstance(new_style, str) and new_style in ["default", "latex", "scientific"]:
        CONFIG[lit.PRINT_STYLE] = PrintStyle(new_style)
    else:
        raise ValueError("The print styles supported are default, latex, and scientific.\n"
                         "These values are found under the enum settings.PrintStyle")


def get_print_style() -> PrintStyle:
    """Gets the current print style setting"""
    return CONFIG[lit.PRINT_STYLE]


def set_unit_style(new_style: Union[UnitStyle, str]):
    """Change the format for presenting units

    The unit style can be either "fraction" or "exponents. Fraction style is the more intuitive
    way of showing units, looks like kg*m^2/s^2, whereas the exponent style shows the same unit
    as kg^1m^2s^-2

    """
    if isinstance(new_style, UnitStyle):
        CONFIG[lit.UNIT_STYLE] = new_style
    elif isinstance(new_style, str) and new_style in ["fraction", "exponents"]:
        CONFIG[lit.UNIT_STYLE] = UnitStyle(new_style)
    else:
        raise ValueError("The unit style can be either exponents or fractions. \n"
                         "The values can be found under the enum settings. UnitStyle")


def get_unit_style() -> UnitStyle:
    """Gets the current unit format settings"""
    return CONFIG[lit.UNIT_STYLE]


def set_sig_figs_for_value(new_sig_figs: numbers.Integral):
    """Sets the number of significant figures to show for all values"""
    if isinstance(new_sig_figs, numbers.Integral) and int(new_sig_figs) > 0:
        sig_figs = int(new_sig_figs)
        CONFIG[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = sig_figs
        CONFIG[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.VALUE
    else:
        raise ValueError("The number of significant figures must be a positive integer")


def set_sig_figs_for_error(new_sig_figs: numbers.Integral):
    """Sets the number of significant figures to show for uncertainties"""
    if isinstance(new_sig_figs, numbers.Integral) and int(new_sig_figs) > 0:
        sig_figs = int(new_sig_figs)
        CONFIG[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = sig_figs
        CONFIG[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.ERROR
    else:
        raise ValueError("The number of significant figures must be a positive integer")


def get_sig_fig_mode() -> SigFigMode:
    """Gets the current standard for significant figures"""
    return CONFIG[lit.SIG_FIGS][lit.SIG_FIG_MODE]


def get_sig_fig_value() -> int:
    """Gets the current setting for the number of significant figures"""
    return CONFIG[lit.SIG_FIGS][lit.SIG_FIG_VALUE]


def set_monte_carlo_sample_size(size: numbers.Integral):
    """Sets the number of samples for a Monte Carlo simulation"""
    if isinstance(size, numbers.Integral) and int(size) > 0:
        CONFIG[lit.MONTE_CARLO_SAMPLE_SIZE] = int(size)
    else:
        raise ValueError("The sample size of the Monte Carlo simulation has to be a positive integer")


def get_monte_carlo_sample_size() -> int:
    """Gets the current Monte Carlo sample size setting"""
    return CONFIG[lit.MONTE_CARLO_SAMPLE_SIZE]
