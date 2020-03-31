"""Holds all global configurations and Enum types for common options"""

import functools
from enum import Enum
from typing import Union

from . import literals as lit


class ErrorMethod(Enum):
    """Preferred method of error propagation"""
    DERIVATIVE = lit.DERIVATIVE
    MONTE_CARLO = lit.MONTE_CARLO
    AUTO = lit.AUTO


class PrintStyle(Enum):
    """Preferred format for the string representation of values"""
    DEFAULT = lit.DEFAULT
    LATEX = lit.LATEX
    SCIENTIFIC = lit.SCIENTIFIC


class UnitStyle(Enum):
    """Preferred format for the string representation of units"""
    FRACTION = lit.FRACTION
    EXPONENTS = lit.EXPONENTS


class SigFigMode(Enum):
    """Preferred method to choose number of significant figures"""
    AUTOMATIC = lit.AUTO
    VALUE = lit.SET_TO_VALUE
    ERROR = lit.SET_TO_ERROR


class Settings:
    """The settings object, implemented as a singleton"""

    __instance = None

    @staticmethod
    def get_instance():
        """Gets the Settings singleton instance"""
        if not Settings.__instance:
            Settings.__instance = Settings()
        return Settings.__instance

    def __init__(self):
        self.__config = {
            lit.ERROR_METHOD: ErrorMethod.DERIVATIVE,
            lit.PRINT_STYLE: PrintStyle.DEFAULT,
            lit.UNIT_STYLE: UnitStyle.EXPONENTS,
            lit.SIG_FIGS: {
                lit.SIG_FIG_MODE: SigFigMode.AUTOMATIC,
                lit.SIG_FIG_VALUE: 1
            },
            lit.MONTE_CARLO_SAMPLE_SIZE: 100000,
            lit.PLOT_DIMENSIONS: (6.4, 4.8)
        }

    @property
    def error_method(self) -> ErrorMethod:
        """ErrorMethod: The preferred error method for derived values

        There are three possible error methods, keep in mind that all three methods are used
        to calculate the values behind the scene. The options are found under q.ErrorMethod

        """
        return self.__config[lit.ERROR_METHOD]

    @error_method.setter
    def error_method(self, new_method: Union[ErrorMethod, str]):
        if isinstance(new_method, ErrorMethod):
            self.__config[lit.ERROR_METHOD] = new_method
        elif new_method in [lit.MONTE_CARLO, lit.DERIVATIVE]:
            self.__config[lit.ERROR_METHOD] = ErrorMethod(new_method)
        else:
            raise ValueError("Invalid error method!")

    @property
    def print_style(self) -> PrintStyle:
        """PrintStyle: The preferred format to display a value with an uncertainty

        The three available formats are default, latex, and scientific. The options are found
        under q.PrintStyle

        """
        return self.__config[lit.PRINT_STYLE]

    @print_style.setter
    def print_style(self, style: Union[PrintStyle, str]):
        if isinstance(style, PrintStyle):
            self.__config[lit.PRINT_STYLE] = style
        elif isinstance(style, str) and style in [lit.DEFAULT, lit.LATEX, lit.SCIENTIFIC]:
            self.__config[lit.PRINT_STYLE] = PrintStyle(style)
        else:
            raise ValueError("Invalid print style!")

    @property
    def unit_style(self) -> UnitStyle:
        """UnitStyle: The preferred format to display a unit string

        The supported unit styles are "fraction" and "exponents. Fraction style is the more
        intuitive way of showing units, looks like kg*m^2/s^2, whereas the exponent style
        shows the same unit as kg^1m^2s^-2, which is more accurate and less ambiguous.

        """
        return self.__config[lit.UNIT_STYLE]

    @unit_style.setter
    def unit_style(self, style: Union[UnitStyle, str]):
        if isinstance(style, UnitStyle):
            self.__config[lit.UNIT_STYLE] = style
        elif isinstance(style, str) and style in [lit.FRACTION, lit.EXPONENTS]:
            self.__config[lit.UNIT_STYLE] = UnitStyle(style)
        else:
            raise ValueError("Invalid unit style!")

    @property
    def sig_fig_mode(self) -> SigFigMode:
        """SigFigMode: The standard for choosing number of significant figures

        Supported modes are VALUE and ERROR. When the mode is VALUE, the center value of the
        quantity will be displayed with the specified number of significant figures, and the
        uncertainty will be displayed to match the number of decimal places of the value, and
        vice versa for the ERROR mode.

        """
        return self.__config[lit.SIG_FIGS][lit.SIG_FIG_MODE]

    @property
    def sig_fig_value(self) -> int:
        """int: The default number of significant figures"""
        return self.__config[lit.SIG_FIGS][lit.SIG_FIG_VALUE]

    @sig_fig_value.setter
    def sig_fig_value(self, new_value: int):
        if isinstance(new_value, int) and new_value > 0:
            self.__config[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = new_value
        else:
            raise ValueError("The number of significant figures must be a positive integer")

    def set_sig_figs_for_value(self, new_sig_figs: int):
        """Sets the number of significant figures to show for all values"""
        self.sig_fig_value = new_sig_figs
        self.__config[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.VALUE

    def set_sig_figs_for_error(self, new_sig_figs: int):
        """Sets the number of significant figures to show for uncertainties"""
        self.sig_fig_value = new_sig_figs
        self.__config[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.ERROR

    @property
    def monte_carlo_sample_size(self) -> int:
        """int: The default sample size used in Monte Carlo error propagation"""
        return self.__config[lit.MONTE_CARLO_SAMPLE_SIZE]

    @monte_carlo_sample_size.setter
    def monte_carlo_sample_size(self, size: int):
        if isinstance(size, int) and size > 0:
            self.__config[lit.MONTE_CARLO_SAMPLE_SIZE] = size
        else:
            raise ValueError("The sample size has to be a positive integer")

    @property
    def plot_dimensions(self) -> (float, float):
        """The default dimensions of a plot in inches"""
        return self.__config[lit.PLOT_DIMENSIONS]

    @plot_dimensions.setter
    def plot_dimensions(self, new_dimensions: (float, float)):
        if not isinstance(new_dimensions, tuple) or len(new_dimensions) != 2:
            raise ValueError("The plot dimensions must be a tuple with two entries")
        if any(not isinstance(num, (int, float)) or num <= 0 for num in new_dimensions):
            raise ValueError("The dimensions of the plot must be numeric")
        self.__config[lit.PLOT_DIMENSIONS] = new_dimensions

    def reset(self):
        """Resets all configurations to their default values"""
        self.__config[lit.ERROR_METHOD] = ErrorMethod.DERIVATIVE
        self.__config[lit.PRINT_STYLE] = PrintStyle.DEFAULT
        self.__config[lit.SIG_FIGS][lit.SIG_FIG_MODE] = SigFigMode.AUTOMATIC
        self.__config[lit.SIG_FIGS][lit.SIG_FIG_VALUE] = 1
        self.__config[lit.UNIT_STYLE] = UnitStyle.EXPONENTS
        self.__config[lit.MONTE_CARLO_SAMPLE_SIZE] = 10000
        self.__config[lit.PLOT_DIMENSIONS] = (6.4, 4.8)


def get_settings() -> Settings:
    """Gets the settings singleton instance"""
    return Settings.get_instance()


def reset_default_configuration():
    """Resets all configurations to their default values"""
    get_settings().reset()


def set_error_method(new_method: Union[ErrorMethod, str]):
    """Sets the preferred error propagation method for values"""
    get_settings().error_method = new_method


def set_print_style(new_style: Union[PrintStyle, str]):
    """Sets the format to display the value strings for ExperimentalValues"""
    get_settings().print_style = new_style


def set_unit_style(new_style: Union[UnitStyle, str]):
    """Change the format for presenting units"""
    get_settings().unit_style = new_style


def set_sig_figs_for_value(new_sig_figs: int):
    """Sets the number of significant figures to show for all values"""
    get_settings().set_sig_figs_for_value(new_sig_figs)


def set_sig_figs_for_error(new_sig_figs: int):
    """Sets the number of significant figures to show for uncertainties"""
    get_settings().set_sig_figs_for_error(new_sig_figs)


def set_monte_carlo_sample_size(size: int):
    """Sets the number of samples for a Monte Carlo simulation"""
    get_settings().monte_carlo_sample_size = size


def set_plot_dimensions(new_dimensions: (float, float)):
    """Sets the default dimensions of a plot"""
    get_settings().plot_dimensions = new_dimensions


def use_mc_sample_size(size: int):
    """Wrapper decorator that temporarily sets the monte carlo sample size"""

    def set_monte_carlo_sample_size_wrapper(func):
        """Inner wrapper decorator"""

        @functools.wraps(func)
        def inner_wrapper(*args):
            # preserve the original sample size and set the sample size to new value
            temp_size = get_settings().monte_carlo_sample_size
            set_monte_carlo_sample_size(size)

            # run the function
            result = func(*args)

            # restores the original sample size
            set_monte_carlo_sample_size(temp_size)

            # return function output
            return result

        return inner_wrapper

    return set_monte_carlo_sample_size_wrapper
