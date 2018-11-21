"""Holds all global configuration variables and Enums for user options

This file contains configurations and flags for user settings including
plot settings and error propagation settings

"""

from enum import Enum
from .literals import ERROR_METHOD, SIG_FIGS, SIG_FIG_MODE, SIG_FIG_VALUE


class ErrorMethod(Enum):
    DERIVATIVE = "derivative"
    MIN_MAX = "min_max"
    MONTE_CARLO = "monte_carlo"


class SigFigMode(Enum):
    AUTOMATIC = "automatic"
    VALUE = "value"
    ERROR = "error"


config = {
    "error_method": ErrorMethod.DERIVATIVE,
    "significant_figures": {
        "mode": SigFigMode.AUTOMATIC,
        "value": 1
    }
}


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
        config[ERROR_METHOD] = new_method


def get_error_method() -> ErrorMethod:
    return config[ERROR_METHOD]


def get_sig_fig_mode() -> SigFigMode:
    return config[SIG_FIGS][SIG_FIG_MODE]


def get_sig_fig_value() -> int:
    return config[SIG_FIGS][SIG_FIG_VALUE]


def set_sig_figs_for_value(new_sig_figs):
    try:
        sig_figs = int(new_sig_figs)
        if sig_figs <= 0:
            raise ValueError  # the number of significant figures has to be greater than 0
        config[SIG_FIGS][SIG_FIG_VALUE] = sig_figs
        config[SIG_FIGS][SIG_FIG_MODE] = SigFigMode.VALUE
    except (ValueError, TypeError):
        print("Error: the number of significant figures has to be an integer greater than 0")


def set_sig_figs_for_error(new_sig_figs):
    try:
        sig_figs = int(new_sig_figs)
        if sig_figs <= 0:
            raise ValueError  # the number of significant figures has to be greater than 0
        config[SIG_FIGS][SIG_FIG_VALUE] = sig_figs
        config[SIG_FIGS][SIG_FIG_MODE] = SigFigMode.ERROR
    except (ValueError, TypeError):
        print("Error: the number of significant figures has to be an integer")
