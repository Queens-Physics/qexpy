"""Miscellaneous utility methods"""

import re
from typing import Union, Callable
from numbers import Real
import numpy as np
from qexpy.utils.exceptions import InvalidArgumentTypeError

# Global variable to keep track of whether the output_notebook command was run
_MPL_OUTPUT_NOTEBOOK_CALLED = False

# helper constant for type checking
ARRAY_TYPES = list, np.ndarray


def count_significant_figures(number: Union[str, Real]) -> int:
    """Counts the number of significant figures for a number

    The input can be either a number or the string representation of a number

    """

    try:
        # first remove the decimal point
        str_repr_of_value = str(number).replace(".", "")
        # then strip the leading 0s
        str_repr_of_value = re.sub(r"^0*", "", str_repr_of_value)
        return len(str_repr_of_value)
    except (ValueError, TypeError):
        raise InvalidArgumentTypeError("count_significant_figures()", got=number,
                                       expected="a number or the string representation of a number")


def load_data_from_file(path: str, delimiter=',') -> np.ndarray:
    """Reads data from a file

    Retrieves data from a file, separated with the given delimiter.

    Args:
        path (str): the file path
        delimiter (str): the delimiter with which the lines are split

    Returns:
        A 2-dimensional numpy array where each column in the file becomes an array of numbers

    """
    import csv
    with open(path, newline='') as openfile:
        reader = csv.reader(openfile, delimiter=delimiter)
        number_pattern = r"\-?[0-9]+(\.[0-9]+)?"
        # find all rows with only valid entries (entries that matches the number format)
        valid_rows = filter(lambda row: all(re.match(number_pattern, entry) for entry in row), reader)
        # for each row convert the entries to numbers
        data = map(lambda row: list(float(entry) for entry in row), valid_rows)
        # TODO: add code to check if all rows are of the same length (or else the transpose won't work properly)
        # transpose to arrays that correspond to the columns
        ret = np.transpose(np.array(list(data), dtype=float))
    return ret


def calculate_covariance(arr_x: ARRAY_TYPES, arr_y: ARRAY_TYPES):
    """Calculates the covariance of two arrays"""
    return 1 / (len(arr_x) - 1) * sum(((x - arr_x.mean()) * (y - arr_y.mean()) for x, y in zip(arr_x, arr_y)))


@np.vectorize
def numerical_derivative(function: Callable, x0: Real, dx=1e-10):
    """Calculates the numerical derivative of a function with respect to x at x0"""
    return (function(x0 + dx) - function(x0 - dx)) / (2 * dx)


def _in_notebook() -> bool:
    """Simple function to check if module is loaded in a notebook"""
    return hasattr(__builtins__, '__IPYTHON__')


def _mpl_output_notebook():
    from IPython import get_ipython
    get_ipython()
    # ipython.magic('matplotlib inline')
    global _MPL_OUTPUT_NOTEBOOK_CALLED  # disable=global-statement
    _MPL_OUTPUT_NOTEBOOK_CALLED = True
