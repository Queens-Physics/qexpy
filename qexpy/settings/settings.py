"""Holds all global configuration variables and Enums for user options

This file contains configurations and flags for user settings including
plot settings and error propagation settings

"""

from enum import Enum
from typing import Union
from . import literals as lit


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
            lit.MONTE_CARLO_SAMPLE_SIZE: 10000,
            lit.PLOT_DIMENSIONS: (6.4, 4.8)
        }

    @property
    def error_method(self) -> ErrorMethod:
        """ErrorMethod: the preferred error method for derived values

        There are three possible error methods, keep in mind that all three methods are used
        to calculate the values behind the scene.

        """
        return self.__config[lit.ERROR_METHOD]

    @error_method.setter
    def error_method(self, new_method: Union[ErrorMethod, str]):
        if isinstance(new_method, ErrorMethod):
            self.__config[lit.ERROR_METHOD] = new_method
        elif new_method in [lit.MONTE_CARLO_PROPAGATED, lit.DERIVATIVE_PROPAGATED]:
            self.__config[lit.ERROR_METHOD] = ErrorMethod(new_method)
        else:
            raise ValueError("The error methods supported are derivative, min-max, and monte carlo.\n"
                             "These options can be found under the enum q.ErrorMethod")

    @property
    def print_style(self) -> PrintStyle:
        """PrintStyle: format of the value strings for ExperimentalValues

        The three available formats are default, latex, and scientific.

        """
        return self.__config[lit.PRINT_STYLE]

    @print_style.setter
    def print_style(self, new_style: Union[PrintStyle, str]):
        if isinstance(new_style, PrintStyle):
            self.__config[lit.PRINT_STYLE] = new_style
        elif isinstance(new_style, str) and new_style in ["default", "latex", "scientific"]:
            self.__config[lit.PRINT_STYLE] = PrintStyle(new_style)
        else:
            raise ValueError("The print styles supported are default, latex, and scientific.\n"
                             "These values are found under the enum q.PrintStyle")

    @property
    def unit_style(self) -> UnitStyle:
        """UnitStyle: The format used to present units

        The unit style can be either "fraction" or "exponents. Fraction style is the more intuitive
        way of showing units, looks like kg*m^2/s^2, whereas the exponent style shows the same unit
        as kg^1m^2s^-2

        """
        return self.__config[lit.UNIT_STYLE]

    @unit_style.setter
    def unit_style(self, new_style: Union[UnitStyle, str]):
        if isinstance(new_style, UnitStyle):
            self.__config[lit.UNIT_STYLE] = new_style
        elif isinstance(new_style, str) and new_style in ["fraction", "exponents"]:
            self.__config[lit.UNIT_STYLE] = UnitStyle(new_style)
        else:
            raise ValueError("The unit style can be either exponents or fractions. \n"
                             "The values can be found under the enum q.UnitStyle")

    @property
    def sig_fig_mode(self) -> SigFigMode:
        """SigFigMode: The standard for choosing number of significant figures

        The significant figure mode can be either VALUE or ERROR. When the mode is VALUE, the
        value of the quantity will be displayed with the specified number of significant figures,
        and the uncertainty will be displayed to match the number of decimal places of the value,
        and vice versa.

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
            raise ValueError("The sample size of the Monte Carlo simulation has to be a positive integer")

    @property
    def plot_dimensions(self) -> (float, float):
        """The default dimensions of a plot in inches"""
        return self.__config[lit.PLOT_DIMENSIONS]

    @plot_dimensions.setter
    def plot_dimensions(self, new_dimensions: (float, float)):
        if not isinstance(new_dimensions, tuple) and len(new_dimensions) != 2:
            raise ValueError("The plot dimensions must be a tuple with two entries")
        if any(not isinstance(num, (int, float)) for num in new_dimensions):
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
