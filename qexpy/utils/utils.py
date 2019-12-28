"""Miscellaneous utility functions"""

import functools
import csv

import numpy as np

from typing import Callable
from numbers import Real
from .exceptions import UndefinedOperationError


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
        if any(isinstance(arg, np.ndarray) for arg in args):
            return np.vectorize(func)(*args)
        if any(isinstance(arg, list) for arg in args):
            return np.vectorize(func)(*args).tolist()
        return func(*args)

    return wrapper_vectorize


def validate_xrange(xrange):
    """validates that an xrange is legal"""

    if not isinstance(xrange, (tuple, list)) or len(xrange) != 2:
        raise TypeError("The \"xrange\" should be a list or tuple of length 2")

    if any(not isinstance(value, Real) for value in xrange):
        raise TypeError("The \"xrange\" must be real numbers")

    if xrange[0] > xrange[1]:
        raise ValueError("The low bound of xrange is higher than the high bound")

    return True


@vectorize
def numerical_derivative(function: Callable, x0: Real, dx=1e-5):
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


def find_mode_and_uncertainty(n, bins, confidence) -> (float, float):
    """Find the mode and uncertainty with a confidence of a histogram distribution"""
    number_of_samples = sum(n)
    max_idx = n.argmax()
    value = (bins[max_idx] + bins[max_idx + 1]) / 2
    count = n[max_idx]
    low_idx, high_idx = max_idx, max_idx
    while count < confidence * number_of_samples:
        low_idx -= 1
        high_idx += 1
        count += n[low_idx] + n[high_idx]
    error = (bins[high_idx] + bins[high_idx + 1]) / 2 - value
    return value, error


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
    with open(filepath, newline='') as openfile:
        reader = csv.reader(openfile, delimiter=delimiter)
        # read file into array of rows
        rows_of_data = list([float(entry) for entry in row] for row in reader)
        # transpose data into array of columns
        result = np.transpose(np.array(rows_of_data, dtype=float))
    return result
