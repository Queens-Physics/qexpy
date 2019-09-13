"""Miscellaneous utility functions"""

import functools

import numpy as np

from typing import Callable
from numbers import Real
from .exceptions import UndefinedOperationError

import qexpy.settings as sts


def check_operand_type(operation):
    """wrapper decorator for undefined operation error reporting"""

    def check_operand_type_wrapper(func):

        @functools.wraps(func)
        def operation_wrapper(*args):
            try:
                return func(*args)
            except TypeError:
                raise UndefinedOperationError(operation, got=args, expected="real numbers")

        return operation_wrapper

    return check_operand_type_wrapper


def vectorize(func):
    """vectorize a function if inputs are arrays"""

    @functools.wraps(func)
    def wrapper_vectorize(*args):
        if any(isinstance(arg, (list, np.ndarray)) for arg in args):
            return np.vectorize(func)(*args)
        return func(*args)

    return wrapper_vectorize


def use_mc_sample_size(size: int):
    """Wrapper decorator that temporarily sets the monte carlo sample size"""

    def set_monte_carlo_sample_size_wrapper(func):
        """Inner wrapper decorator"""

        @functools.wraps(func)
        def inner_wrapper(*args):
            # preserve the original sample size and set the sample size to new value
            temp_size = sts.get_settings().monte_carlo_sample_size
            sts.set_monte_carlo_sample_size(size)

            # run the function
            result = func(*args)

            # restores the original sample size
            sts.set_monte_carlo_sample_size(temp_size)

            # return function output
            return result

        return inner_wrapper

    return set_monte_carlo_sample_size_wrapper


def validate_xrange(xrange):
    """validates that an xrange is legal"""

    if not isinstance(xrange, (tuple, list)) or len(xrange) != 2:
        raise TypeError("The \"xrange\" should be a list or tuple of length 2")

    if xrange[0] > xrange[1]:
        raise ValueError("The low bound of xrange is higher than the high bound")

    return True


@vectorize
def numerical_derivative(function: Callable, x0: Real, dx=1e-10):
    """Calculates the numerical derivative of a function with respect to x at x0"""
    return (function(x0 + dx) - function(x0 - dx)) / (2 * dx)


def calculate_covariance(arr_x, arr_y):
    """Calculates the covariance of two arrays"""
    if len(arr_x) != len(arr_y):
        raise ValueError("Cannot calculate covariance for arrays of different lengths.")
    return 1 / (len(arr_x) - 1) * sum(
        ((x - np.mean(arr_x)) * (y - np.mean(arr_y)) for x, y in zip(arr_x, arr_y)))


def cov2corr(pcov: np.ndarray) -> np.ndarray:
    """Calculate a correlation matrix from a covariance matrix"""
    std = np.sqrt(np.diag(pcov))
    return pcov / np.outer(std, std)


def load_data_from_file(filepath: str, delimiter=",") -> np.ndarray:
    """Reads arrays of data from a file

    The file should be structured like a csv file. The delimiter can be replaced with other
    characters, but the default is comma. The function returns an array of arrays, one for
    each column in the table of numbers.

    Args:
        filepath (str): The name of the file to read from
        delimiter (str): The delimiter that separates each row

    Returns:
        A 2-dimensional np.ndarray where each array is a column in the file

    """
    import csv
    with open(filepath, newline='') as openfile:
        reader = csv.reader(openfile, delimiter=delimiter)
        # read file into array of rows
        rows_of_data = list([float(entry) for entry in row] for row in reader)
        # transpose data into array of columns
        result = np.transpose(np.array(rows_of_data, dtype=float))
    return result
