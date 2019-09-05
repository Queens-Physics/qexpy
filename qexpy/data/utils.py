"""Utility methods for the data module"""

import numpy as np

from numbers import Real

from . import data as dt, datasets as dts  # pylint: disable=cyclic-import

ARRAY_TYPES = np.ndarray, list


def wrap_in_experimental_value(operand) -> "dt.ExperimentalValue":
    """Wraps a variable in an ExperimentalValue object

    Wraps single numbers in a Constant, number pairs in a MeasuredValue. If the argument
    is already an ExperimentalValue instance, return directly. If the

    """

    if isinstance(operand, Real):
        return dt.Constant(operand)
    if isinstance(operand, dt.ExperimentalValue):
        return operand
    if isinstance(operand, tuple) and len(operand) == 2:
        return dt.MeasuredValue(operand[0], operand[1])
    raise TypeError("Cannot parse a {} into an ExperimentalValue".format(type(operand)))


def wrap_in_measurement(value, **kwargs) -> "dt.ExperimentalValue":
    """Wraps a value in a Measurement object"""

    if isinstance(value, Real):
        return dt.MeasuredValue(value, 0, **kwargs)
    if isinstance(value, tuple) and len(value) == 2:
        return dt.MeasuredValue(*value, **kwargs)
    if isinstance(value, dt.ExperimentalValue):
        return value

    raise ValueError(
        "Elements of a MeasurementArray must be convertible to an ExperimentalValue")


def wrap_in_value_array(operand, **kwargs) -> np.ndarray:
    """Wraps input in an ExperimentalValueArray"""

    # wrap array times in numpy arrays
    if isinstance(operand, dts.ExperimentalValueArray):
        return operand
    if isinstance(operand, ARRAY_TYPES):
        return np.asarray([wrap_in_measurement(value, **kwargs) for value in operand])

    # wrap single value times in array
    return np.asarray([wrap_in_measurement(operand, **kwargs)])
