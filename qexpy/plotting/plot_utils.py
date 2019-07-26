"""This file contains helper methods for the plotting module"""

import re
from qexpy.utils.exceptions import InvalidArgumentTypeError, IllegalArgumentError


def _validate_xrange(new_range: tuple, allow_empty=False):
    """Helper function to validate the xrange specified"""
    if not isinstance(new_range, tuple) and not isinstance(new_range, list):
        raise InvalidArgumentTypeError("xrange", new_range, "tuple or list of length 2")
    if allow_empty and new_range == ():
        return
    if len(new_range) != 2 or new_range[0] > new_range[1]:
        raise IllegalArgumentError(
            "Error: the xrange has to be a tuple of length 2 where the second number is larger than the first.")


def _validate_fmt(new_fmt: str):
    """Helper function to validate the format string of a plot object"""
    if not isinstance(new_fmt, str):
        raise InvalidArgumentTypeError("fmt", new_fmt, "string")
    if not re.fullmatch(r"(\.|,|o|v|^|<|>|1|2|3|4|s|p|\*|h|H|\+|x|D|d|\||_)?(-|(--)|(-.)|(:))?([bgrcmykw])?", new_fmt):
        raise IllegalArgumentError(
            "The format string is invalid. Please refer to documentations for a list of valid format strings.")
