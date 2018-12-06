"""Module containing data structures for experimental values

This module defines ExperimentalValue and all its sub-classes. Experimental measurements
and their corresponding uncertainties can be recorded as an instance of this class. Any
operations done with these instances will have the properly propagated errors and units.

"""

import numpy as np
import math as m
import numbers
import warnings
import uuid
import collections
from typing import Dict

import qexpy.utils.utils as utils
from . import operations as op
import qexpy.settings.literals as lit
import qexpy.utils.units as units
import qexpy.utils.printing as printing
import qexpy.settings.settings as settings

# a simple structure for a value-error pair
ValueWithError = collections.namedtuple("ValueWithError", "value, error")


class ExperimentalValue:
    """Root class for objects with a value and an uncertainty

    This class is practically an abstract class, not to be instantiated directly. An instance
    of this class can store multiple value-error pairs. These pairs are stored in the _values
    attribute, a dictionary object where the key of the entries indicate the source of this
    value-error pair. "recorded" indicates that this value and its uncertainty are directly
    recorded by the user. Another two possible keys are "derivative" and "monte-carlo", which
    represents the two error methods by which the uncertainties can be propagated.

    This class also contains functionality for automatic unit tracking. Each instance of this
    class has a _units attribute, a dictionary object which stores each separate independent
    unit this value has, and their exponents. For example, the unit for Newton (kg*m^2/s^2) is
    stored as {"kg": 1, "m": 2, "s": -2}. These units will be tracked and propagated during
    any operation done with objects of this class.

    Attributes:
        _values (dict): the values and uncertainties on the values
        _units (dict): the units and their exponents
        _name (str): the name of this instance
        _id (UUID): the unique ID of an instance

    TODO:
        Implement smart unit tracking.
        Find better way of tracking relationships between values

    """

    _register = {}  # module level database for instantiated MeasuredValues and DerivedValues
    _correlations = {}  # database for correlation between values
    _formula_register = {}  # module level database for calculated values

    def __init__(self, unit="", name=""):
        """Default constructor, not to be called directly"""

        # initialize values
        self._values = {}  # type: Dict[str, ValueWithError]
        self._units = units.parse_units(unit) if unit else {}
        self._name = name
        self._id = uuid.uuid4()

    def __str__(self):
        string = ""
        # print name of the quantity
        if self.name:
            string += self.name + " = "
        # print the value and error
        string += self.__print_value()
        if self._units:
            string += " [" + self.unit + "]"
        return string

    @property
    def value(self) -> float:
        """Default getter for value of the quantity

        This is practically an abstract method, to be overridden by its child classes

        """
        return 0

    @property
    def error(self) -> float:
        """Default getter for uncertainty of the quantity

        This is practically an abstract method, to be overridden by its child classes

        """
        return 0

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        if isinstance(new_name, str):
            self._name = new_name
        else:
            raise ValueError("The name of a value must be a string")

    @property
    def unit(self) -> str:
        if not self._units:
            return ""
        return units.construct_unit_string(self._units)

    @unit.setter
    def unit(self, new_unit: str):
        if isinstance(new_unit, str):
            self._units = units.parse_units(new_unit)
        else:
            raise ValueError("You can only set the unit of a value using its string representation")

    def __eq__(self, other):
        return self.value == _wrap_operand(other).value

    def __gt__(self, other):
        return self.value > _wrap_operand(other).value

    def __ge__(self, other):
        return self.value >= _wrap_operand(other).value

    def __lt__(self, other):
        return self.value < _wrap_operand(other).value

    def __le__(self, other):
        return self.value <= _wrap_operand(other).value

    def __neg__(self):
        return DerivedValue({
            lit.OPERATOR: lit.NEG,
            lit.OPERANDS: [self]
        })

    def __add__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.ADD,
            lit.OPERANDS: [self, _wrap_operand(other)]
        })

    def __radd__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.ADD,
            lit.OPERANDS: [_wrap_operand(other), self]
        })

    def __sub__(self, other):
        if op.construct_formula_string(self) == op.construct_formula_string(other):
            # For values subtracting themselves, the result should be exactly 0
            return Constant(0)
        return DerivedValue({
            lit.OPERATOR: lit.SUB,
            lit.OPERANDS: [self, _wrap_operand(other)]
        })

    def __rsub__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.SUB,
            lit.OPERANDS: [_wrap_operand(other), self]
        })

    def __mul__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.MUL,
            lit.OPERANDS: [self, _wrap_operand(other)]
        })

    def __rmul__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.MUL,
            lit.OPERANDS: [_wrap_operand(other), self]
        })

    def __truediv__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.DIV,
            lit.OPERANDS: [self, _wrap_operand(other)]
        })

    def __rtruediv__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.DIV,
            lit.OPERANDS: [_wrap_operand(other), self]
        })

    def scale(self, factor, update_units=True):
        """Scale the value and error of this instance by a factor

        This method can be used to scale up or down a value, usually due to the change of
        unit (for example, change from using [m] to using [mm]. The uncertainty of the value
        will be updated accordingly. By default, the units will be updated if applicable.
        This feature can be disabled by passing update_units=False

        """
        # TODO: this is cool , implement this

    def derivative(self, other: "ExperimentalValue"):
        """Finds the derivative with respect to another ExperimentalValue

        This method is usually called from a DerivedValue object. When you take the derivative
        of a measured value with respect to anything other than itself, the return value should
        be 0. The derivative of any quantity with respect to itself is always 1

        Args:
            other (ExperimentalValue): the other ExperimentalValue

        """
        if isinstance(other, ExperimentalValue):
            return 1 if self._id == other._id else 0
        else:
            raise ValueError("You can only find the derivative of a value with respect to another "
                             "qexpy defined value.")

    def get_units(self):
        """Gets the raw dictionary object for units"""
        from copy import deepcopy
        # use deep copy because a dictionary object is mutable
        return deepcopy(self._units)

    def __print_value(self) -> str:
        """Helper method that prints the value-error pair in proper format"""
        if self.value == float('inf'):
            return "inf"
        return printing.get_printer()(self.value, self.error)


class MeasuredValue(ExperimentalValue):
    """Class for user recorded experimental values with an uncertainty

    This class is not to be instantiated directly, use the Measurement method instead.

    """

    def __init__(self, value=0.0, error=0.0, unit="", name=""):
        super().__init__(unit, name)
        self._values[lit.RECORDED] = ValueWithError(value, error)
        ExperimentalValue._register[self._id] = self

    @property
    def value(self) -> float:
        """Gets the value for this measurement"""
        return self._values[lit.RECORDED].value

    @value.setter
    def value(self, new_value):
        """Modifies the value of a measurement"""
        if isinstance(new_value, numbers.Real):
            self._values[lit.RECORDED] = ValueWithError(new_value, self._values[lit.RECORDED].error)
        else:
            raise ValueError("You can only set the value of a measurement to a number")
        if hasattr(self, "_raw_data"):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the value of a repeated measurement. Doing so has "
                          "caused you to lose the original list of raw measurement data")
            self.__class__ = MeasuredValue  # casting it to base class

    @property
    def error(self) -> float:
        """Gets the uncertainty on the measurement"""
        return self._values[lit.RECORDED].error

    @error.setter
    def error(self, new_error):
        """Modifies the value of a measurement

        Args:
            new_error (float): The new uncertainty

        """
        if isinstance(new_error, numbers.Real) and new_error > 0:
            self._values[lit.RECORDED] = ValueWithError(self._values[lit.RECORDED].value, new_error)
        else:
            raise ValueError("You can only set the error of a measurement to a positive number")
        if hasattr(self, "_raw_data"):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the uncertainty of a repeated measurement.")

    @property
    def relative_error(self) -> float:
        """Gets the relative error (error/mean) of a MeasuredValue object."""
        return self.error / self.value if self.value != 0 else 0.

    @relative_error.setter
    def relative_error(self, relative_error):
        """Sets the relative error (error/mean) of a MeasuredValue object.

        Args:
            relative_error (float): The new uncertainty

        """
        if isinstance(relative_error, numbers.Real) and relative_error > 0:
            new_error = self.value * float(relative_error)
            self._values[lit.RECORDED] = ValueWithError(self.value, new_error)
        else:
            raise ValueError("The relative uncertainty of a measurement has to be a positive number")
        if hasattr(self, "_raw_data"):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the uncertainty of a repeated measurement.")


class RepeatedlyMeasuredValue(MeasuredValue):
    """Class to store the result of repeated measurements on a single quantity

    This class is instantiated when the user takes multiple measurements on a single quantity.
    It is just like any regular MeasuredValue object, with the original array of measurements
    preserved as "raw_data".

    The mean and error on the mean is used as the default value and error of this measurement.
    The user also has an option of using the standard deviation as the error on this measurement.

    The user can also manually set the uncertainty of this object. However, if the user choose
    to manually override the value of this measurement. The original raw data will be lost, and
    the instance will be casted to its parent class MeasuredValue

    Attributes:
        _std (float): the standard derivative of set of measurements
        _error_on_mean (float): the error on the mean of the set of measurements
        _raw_data (np.ndarray): the original list of raw measurements

    """

    def __init__(self, measurement_array, unit, name):
        super().__init__(unit=unit, name=name)
        measurements = np.array(measurement_array)
        self._std = measurements.std(ddof=1)
        self._error_on_mean = self._std / m.sqrt(measurements.size)
        self._values[lit.RECORDED] = ValueWithError(measurements.mean(), self._error_on_mean)
        self._raw_data = measurements

    @property
    def raw_data(self) -> np.ndarray:
        """Gets the raw data that was used to generate this measurement"""
        from copy import deepcopy
        # returns a copy of the list so that the original data is not tempered
        return deepcopy(self._raw_data)

    @property
    def std(self) -> float:
        return self._std

    @property
    def error_on_mean(self) -> float:
        return self._error_on_mean

    def use_std_for_uncertainty(self):
        value = self._values[lit.RECORDED]
        self._values[lit.RECORDED] = ValueWithError(value.value, self._std)

    def use_error_on_mean_for_uncertainty(self):
        value = self._values[lit.RECORDED]
        self._values[lit.RECORDED] = ValueWithError(value.value, self._error_on_mean)

    def show_histogram(self):
        """Plots the raw measurement data in a histogram

        For the result of repeated measurements of a single quantity, the raw measurement
        data is preserved. With this method, you can visualize these values in a histogram.
        with lines corresponding to the mean and the range covered by one standard deviation

        """


# noinspection PyPep8Naming
def Measurement(*args, **kwargs):
    """Records a measurement with uncertainties

    This method is used to create a MeasuredValue object from a single measurement or an array
    of repeated measurements on a single quantity (if you want them averaged). This method is
    named upper case because it is a wrapper for constructors, and should look like a constructor
    from the outside

    When two values are passed to this method, the first argument will be recognized as the value,
    the second as the uncertainty. If the second value is not provided, the uncertainty is by
    default set to 0. If a list of values is passed to this method, the mean and standard deviation
    of the value will be calculated and preserved.

    TODO:
        Enable users to add uncertainties to an array of measurements for a single quantity

    Usage:
        Measurement(12, 1) -> 12 +/- 1
        Measurement(12) -> 12 +/- 0
        Measurement([5.6, 4.8, 6.1, 4.9, 5.1]) -> 5.3000 +/- 0.5431

    You can also specify the name and unit of the value as keyword arguments.

    For example:
        Measurement(12, 1, name="length", unit="m") -> length = 12 +/- 1 [m]

    """

    unit = kwargs["unit"] if "unit" in kwargs else ""
    name = kwargs["name"] if "name" in kwargs else ""

    if len(args) == 1 and isinstance(args[0], utils.ARRAY_TYPES):
        return RepeatedlyMeasuredValue(args[0], unit, name)
    elif len(args) == 1 and isinstance(args[0], numbers.Real):
        return MeasuredValue(float(args[0]), 0.0, unit, name)
    elif len(args) == 2 and isinstance(args[0], numbers.Real) and isinstance(args[1], numbers.Real):
        return MeasuredValue(float(args[0]), float(args[1]), unit, name)
    else:
        raise ValueError("Input must be either a single array of values, or the central value "
                         "and its uncertainty in one measurement")


class DerivedValue(ExperimentalValue):
    """Values derived from operations with other experimental values

    This class is not to be instantiated directly. It will be created when operations are done
    on other ExperimentalValue objects. The error of the DerivedValue object will be propagated
    from the original ExperimentalValue objects.

    The DerivedValue object is preserves information on how it was derived. This is stored in the
    "_formula" attribute, which is a dictionary object with the operator and the operands specified.
    If the operands are also DerivedValues, their formula can also be accessed, which in some way
    works like a syntax tree

    Attributes:
        _error_method (ErrorMethod): the error method used for this value
        _is_error_method_specified (bool): true if the user specified an error method for
            this value, false if default was used
        _formula (dict): the formula of this object

    """

    def __init__(self, formula, error_method=None):
        """The default constructor for the result of an operation

        The operation is stored as a syntax tree in the "_formula" attribute.

        Args:
            formula (dict): the formula organized as a tree

        """
        super().__init__()
        ExperimentalValue._register[self._id] = self

        # set default error method for this value
        if error_method:
            self._is_error_method_specified = True
            self._error_method = error_method
        else:
            # use global default if not specified
            self._is_error_method_specified = False
            self._error_method = settings.get_error_method()
        self._formula = formula

        # record formula in register
        formula_string = op.construct_formula_string(formula)
        if formula_string in ExperimentalValue._formula_register:
            from copy import deepcopy
            existing_result = ExperimentalValue._formula_register[formula_string]  # type: ExperimentalValue
            self._values = deepcopy(existing_result._values)
            self._units = deepcopy(existing_result._units)
            return  # for values that is already calculated, return the result
        else:
            ExperimentalValue._formula_register[formula_string] = self

        # propagate results
        self._values = op.execute(formula[lit.OPERATOR], formula[lit.OPERANDS])
        self._units = op.propagate_units(formula[lit.OPERATOR], formula[lit.OPERANDS])

    @property
    def value(self) -> float:
        """Gets the value for this calculated quantity"""
        return self.__get_value_error_pair().value

    @value.setter
    def value(self, new_value):
        """Modifies the value of this quantity"""
        if isinstance(new_value, numbers.Real):
            self._values = {lit.RECORDED: ValueWithError(new_value, self.error)}  # reset the values of this object
            # TODO: don't forget to reset the relations between this instance and other values
            warnings.warn("You are trying to modify the value of a calculated quantity. Doing so has caused "
                          "this value to be regarded as a Measurement and all other information lost")
            self.__class__ = MeasuredValue  # casting it to MeasuredValue
        else:
            raise ValueError("You can only set the value of a ExperimentalValue to a number")

    @property
    def error(self) -> float:
        """Gets the uncertainty on the calculated quantity"""
        return self.__get_value_error_pair().error

    @error.setter
    def error(self, new_error):
        """Modifies the uncertainty of this quantity

        This is not recommended. Doing so will cause this object to be casted to MeasuredValue

        Args:
            new_error (float): The new uncertainty

        """
        if isinstance(new_error, numbers.Real) and new_error > 0:
            self._values = {lit.RECORDED: ValueWithError(self.value, new_error)}  # reset the values of this object
            # TODO: don't remember to reset the relations between this instance and other values
            warnings.warn("You are trying to modify the value of a calculated quantity. Doing so has caused "
                          "this value to be regarded as a Measurement and all other information lost")
            self.__class__ = MeasuredValue  # casting it to MeasuredValue
        else:
            raise ValueError("You can only set the error of a ExperimentalValue to a positive number")

    @property
    def relative_error(self) -> float:
        """Gets the relative error (error/mean) of a DerivedValue object."""
        return self.error / self.value if self.value != 0 else 0.

    @relative_error.setter
    def relative_error(self, relative_error):
        """Sets the relative error (error/mean) of a DerivedValue object.

        This is not recommended. Doing so will cause this object to be casted to MeasuredValue

        Args:
            relative_error (float): The new relative uncertainty

        """
        if isinstance(relative_error, numbers.Real) and relative_error > 0:
            new_error = self.value * float(relative_error)
            self._values = {lit.RECORDED: ValueWithError(self.value, new_error)}  # reset the values of this object
            # TODO: don't remember to reset the relations between this instance and other values
            warnings.warn("You are trying to modify the value of a calculated quantity. Doing so has caused "
                          "this value to be regarded as a Measurement and all other information lost")
            self.__class__ = MeasuredValue  # casting it to MeasuredValue
        else:
            raise ValueError("The relative uncertainty of a ExperimentalValue has to be a positive number")

    def derivative(self, other):
        """Finds the derivative with respect to another ExperimentalValue

        Args:
            other (ExperimentalValue): the other value

        """

        if self._id == other._id:
            return 1  # the derivative of anything with respect to itself is 1
        root_operator = self._formula[lit.OPERATOR]
        raw_operands = self._formula[lit.OPERANDS]
        operands = list(map(_wrap_operand, raw_operands))
        return op.differentiator(root_operator)(other, *operands)

    def __get_value_error_pair(self) -> ValueWithError:
        """Gets the value-error pair for the current specified method"""
        if self._is_error_method_specified:
            error_method = self._error_method
        else:
            error_method = settings.get_error_method()
        if error_method == settings.ErrorMethod.MONTE_CARLO and lit.MONTE_CARLO_PROPAGATED not in self._values:
            # The Monte Carlo method is slower than the derivative method, so it's not calculated by default
            # unless specifically requested by the user.
            self._values[lit.MONTE_CARLO_PROPAGATED] = op.get_monte_carlo_propagated_value(self)
        return self._values[error_method.value]


class Constant(ExperimentalValue):
    """A value with no uncertainty

    This is created when a constant (int, float, etc.) is used in operation with another
    ExperimentalValue. This class is instantiated before calculating operations to ensure
    objects can be combined.

    """

    def __init__(self, value, unit="", name=""):
        super().__init__(unit=unit, name=name)
        if isinstance(value, numbers.Real):
            self._values[lit.RECORDED] = ValueWithError(value, 0)
        else:
            raise ValueError("The value must be a number")

    @property
    def value(self):
        return self._values[lit.RECORDED].value

    @property
    def error(self):
        return 0

    def derivative(self, other):
        return 0  # the derivative of a constant with respect to anything is 0


class MeasurementArray(np.ndarray):
    """An array of measurements

    This class is used to hold a series of measurements. It can be used for data analysis,
    fitting, and plotting.

    """


def set_covariance(a, b, cov=None):
    """Sets the covariance of two MeasuredValue objects

    Users can manually set the covariance of two MeasuredValue objects. If the two objects are
    both RepeatedMeasurement with raw_data of the same length, the user can omit the input
    value, and the covariance will be calculated automatically. If a value is specified, the
    calculated value will be overridden with the user specified value

    Once the covariance is set or calculated, it will be stored in a module level register for
    easy reference during error propagation. By default the error propagation process assumes
    that the covariance between any two values is 0, unless specified or calculated

    Args:
        a (MeasuredValue): an instances of MeasuredValue
        b (MeasuredValue: an instances of MeasuredValue
        cov (float): the new covariance value

    """
    if not isinstance(a, MeasuredValue) or not isinstance(b, MeasuredValue):
        raise ValueError("You can only set the covariance between two Measurements")
    if a.error == 0 or b.error == 0:
        raise ArithmeticError("Constants or values with no standard deviation don't correlate with other values")
    if isinstance(a, RepeatedlyMeasuredValue) and isinstance(b, RepeatedlyMeasuredValue):
        covariance = cov
        if len(a.raw_data) == len(b.raw_data) and cov is None:
            covariance = utils.calculate_covariance(a.raw_data, b.raw_data)
        elif len(a.raw_data) != len(b.raw_data) and cov is None:
            raise ValueError("The two arrays of repeated measurements are not of the same length")
        correlation = covariance / (a.std * b.std)
    elif cov is None:
        raise ValueError("The covariance value is not provided")
    else:
        covariance = cov
        correlation = cov / (a.error * b.error)

    # check that the result makes sense
    if correlation > 1 or correlation < -1:
        raise ValueError("The covariance provided: {} is non-physical".format(cov))

    # set relations in the module level register
    __set_covariance(a, b, covariance)
    __set_correlation(a, b, correlation)


def get_covariance(a, b) -> float:
    """Finds the covariance of two ExperimentalValues"""
    if isinstance(a, uuid.UUID) and isinstance(b, uuid.UUID):
        a = ExperimentalValue._register[a]
        b = ExperimentalValue._register[b]
    if a.error == 0 or b.error == 0:
        return 0  # constants don't correlate with anyone
    if isinstance(a, MeasuredValue) and isinstance(b, MeasuredValue):
        return __get_covariance(a, b)
    else:
        return 0  # TODO: implement covariance propagation for DerivedValues


def set_correlation(a, b, cor=None):
    """Sets the correlation factor between two MeasuredValue objects

    See Also:
        set_covariance

    """
    # check that the input makes sense
    if cor and (cor > 1 or cor < -1):
        raise ValueError("The correlation factor provided: {} is non-physical.".format(cor))

    if not isinstance(a, MeasuredValue) or not isinstance(b, MeasuredValue):
        raise ValueError("You can only set the correlation factor between two Measurements")
    if a.error == 0 or b.error == 0:
        raise ArithmeticError("Constants or values with no standard deviation don't correlate with other values")
    if isinstance(a, RepeatedlyMeasuredValue) and isinstance(b, RepeatedlyMeasuredValue):
        if len(a.raw_data) == len(b.raw_data) and cor is None:
            covariance = utils.calculate_covariance(a.raw_data, b.raw_data)
            correlation = covariance / (a.std * b.std)
        elif len(a.raw_data) != len(b.raw_data) and cor is None:
            raise ValueError("The two arrays of repeated measurements are not of the same length")
        else:
            correlation = cor
            covariance = cor * (a.std * b.std)
    elif cor is None:
        raise ValueError("The covariance value is not provided")
    else:
        covariance = cor * (a.error * b.error)
        correlation = cor

    # set relations in the module level register
    __set_covariance(a, b, covariance)
    __set_correlation(a, b, correlation)


def get_correlation(a, b) -> float:
    """Finds the correlation factor of two ExperimentalValues"""
    if isinstance(a, uuid.UUID) and isinstance(b, uuid.UUID):
        a = ExperimentalValue._register[a]
        b = ExperimentalValue._register[b]
    if a.error == 0 or b.error == 0:
        return 0  # constants don't correlate with anyone
    if isinstance(a, MeasuredValue) and isinstance(b, MeasuredValue):
        return __get_correlation(a, b)
    else:
        return 0  # TODO: implement covariance propagation for DerivedValues


def _wrap_operand(operand):
    """Wraps the operand of an operation in an ExperimentalValue instance

    If the operand is a number, construct a Constant object with this value. If the operand
    is an ExperimentalValue object, return the object directly.

    """
    if isinstance(operand, numbers.Real):
        return Constant(operand)
    elif isinstance(operand, ExperimentalValue):
        return operand
    else:
        raise ValueError("Operand of type {} for this operation is invalid!".format(type(operand)))


def __set_covariance(a, b, cov):
    """sets a covariance between two MeasuredValue objects

    Args:
        a (MeasuredValue): the first MeasuredValue object
        b (MeasuredValue): the second MeasuredValue object
        cov (float): the covariance factor

    """
    id_string = "_".join(sorted(map(str, [a._id, b._id])))
    if id_string in ExperimentalValue._correlations:
        ExperimentalValue._correlations[id_string][lit.COVARIANCE] = cov
    else:
        ExperimentalValue._correlations[id_string] = {
            lit.VALUES: (a, b),
            lit.COVARIANCE: cov
        }


def __set_correlation(a, b, cor):
    """sets a correlation factor between two MeasuredValue objects

    Args:
        a (MeasuredValue): the first MeasuredValue object
        b (MeasuredValue): the second MeasuredValue object
        cor (float): the correlation factor

    """
    id_string = "_".join(sorted(map(str, [a._id, b._id])))
    if id_string in ExperimentalValue._correlations:
        ExperimentalValue._correlations[id_string][lit.CORRELATION] = cor
    else:
        ExperimentalValue._correlations[id_string] = {
            lit.VALUES: (a, b),
            lit.CORRELATION: cor
        }


def __get_covariance(a, b) -> float:
    """Gets the covariance factor between two values

    Args:
        a (ExperimentalValue): the first ExperimentalValue object
        b (ExperimentalValue): the second ExperimentalValue object

    """
    if a._id == b._id:
        # the covariance between a value and itself is the variance
        return a.std ** 2 if isinstance(a, RepeatedlyMeasuredValue) else a.error ** 2
    id_string = "_".join(sorted(map(str, [a._id, b._id])))
    if id_string in ExperimentalValue._correlations:
        return ExperimentalValue._correlations[id_string][lit.COVARIANCE]
    else:
        return 0


def __get_correlation(a, b) -> float:
    """Gets the correlation factor between two values

    Args:
        a (ExperimentalValue): the first ExperimentalValue object
        b (ExperimentalValue): the second ExperimentalValue object

    """
    if a._id == b._id:
        return 1  # values have unit correlation with themselves
    id_string = "_".join(sorted(map(str, [a._id, b._id])))
    if id_string in ExperimentalValue._correlations:
        return ExperimentalValue._correlations[id_string][lit.CORRELATION]
    else:
        return 0
