"""Holds all global configuration variables and Enums for user options

This file contains configurations and flags for user settings including
plot settings and error propagation settings

"""

from enum import Enum
import numbers
from . import literals as lit


class ErrorMethod(Enum):
    DERIVATIVE = "derivative"
    MONTE_CARLO = "monte_carlo"


class PrintStyle(Enum):
    DEFAULT = "default"
    LATEX = "latex"
    SCIENTIFIC = "scientific"


class UnitStyle(Enum):
    FRACTION = "fraction"
    EXPONENTS = "exponents"


class SigFigMode(Enum):
    AUTOMATIC = "automatic"
    VALUE = "value"
    ERROR = "error"


config = {
    "error_method": ErrorMethod.DERIVATIVE,
    "print_style": PrintStyle.DEFAULT,
    "unit_style": UnitStyle.EXPONENTS,
    "significant_figures": {
        "mode": SigFigMode.AUTOMATIC,
        "value": 1
    }
}


def reset_default_configuration():
    """Resets all configurations to their default values"""
    config[lit.ERROR_METHOD] = ErrorMethod.DERIVATIVE
    config[lit.PRINT_STYLE] = PrintStyle.DEFAULT
    config[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.AUTOMATIC
    config[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = 1
    config[lit.UNIT_STYLE] = UnitStyle.EXPONENTS


def set_error_method(new_method):
    """Sets the preferred error method for values

    This sets the method for error propagation. Keep in mind that all three error methods
    are used to calculate values behind the scene. This method allows the user to choose
    which values to print out

    Args:
        new_method (ErrorMethod): the error method to be used

    """
    if not isinstance(new_method, ErrorMethod):
        print("Error: the error methods supported are derivative, min-max, and monte carlo.\n"
              "These values are found under the enum settings.ErrorMethod")
    else:
        config[lit.ERROR_METHOD] = new_method


def get_error_method() -> ErrorMethod:
    return config[lit.ERROR_METHOD]


def set_print_style(new_style):
    """Sets the format of value strings

    The three available formats are default, latex, and scientific, which can all
    be found in the enum settings.PrintStyle

    Args:
        new_style (PrintStyle): the choice of format

    """
    if not isinstance(new_style, PrintStyle):
        raise ValueError("The print styles supported are default, latex, and scientific.\n"
                         "These values are found under the enum settings.PrintStyle")
    else:
        config[lit.ERROR_METHOD] = new_style


def get_print_style() -> PrintStyle:
    return config[lit.PRINT_STYLE]


def set_unit_style(new_style):
    if not isinstance(new_style, UnitStyle):
        raise ValueError("The unit style can be either exponents or fractions. \n"
                         "The values can be found under the enum settings. UnitStyle")
    else:
        config[lit.UNIT_STYLE] = new_style


def get_unit_style() -> UnitStyle:
    return config[lit.UNIT_STYLE]


def get_sig_fig_mode() -> SigFigMode:
    return config[lit.SIG_FIGS][lit.SIG_FIG_MODE]


def get_sig_fig_value() -> int:
    return config[lit.SIG_FIGS][lit.SIG_FIG_VALUE]


def set_sig_figs_for_value(new_sig_figs):
    try:
        sig_figs = int(new_sig_figs)
        if sig_figs <= 0:
            raise ValueError  # the number of significant figures has to be greater than 0
        config[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = sig_figs
        config[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.VALUE
    except (ValueError, TypeError):
        raise ValueError("The number of significant figures has to be an integer greater than 0")


def set_sig_figs_for_error(new_sig_figs):
    try:
        sig_figs = int(new_sig_figs)
        if sig_figs <= 0:
            raise ValueError  # the number of significant figures has to be greater than 0
        config[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = sig_figs
        config[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.ERROR
    except (ValueError, TypeError):
        raise ValueError("The number of significant figures has to be an integer")
