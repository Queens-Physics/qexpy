"""Module containing data structures for experimental values

This module defines ExperimentalValue and all its sub-classes. Experimental measurements
and their corresponding uncertainties can be recorded as an instance of this class. Any
operations done with these instances will have the properly propagated errors and units.

"""

import warnings
import uuid
import collections
from numbers import Real
from typing import Dict, Union, List
import math as m
import numpy as np

import qexpy.utils.utils as utils
import qexpy.data.operations as op
import qexpy.settings.literals as lit
import qexpy.utils.units as units
import qexpy.utils.printing as printing
import qexpy.settings.settings as settings

# a simple structure for a value-error pair
ValueWithError = collections.namedtuple("ValueWithError", "value, error")

# a data structure to store an expression tree of a formula
Formula = collections.namedtuple("Formula", "operator, operands")

# a data structure to store the correlation between two values
Correlation = collections.namedtuple("Correlation", "values, correlation, covariance")


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
        _id (uuid.UUID): the unique ID of an instance

    TODO:
        Implement smart unit tracking.

    """

    # module level database for instantiated MeasuredValues and DerivedValues
    _register = {}  # type: Dict[uuid.UUID, ExperimentalValue]

    # stores the correlation between values
    _correlations = {}  # type: Dict[str, Correlation]

    def __init__(self, unit="", name=""):
        """Default constructor, not to be called directly"""

        # initialize values
        self._values = {}  # type: Dict[str, ValueWithError]
        self._units = units.parse_units(unit) if unit else {}  # type: Dict[str, int]
        self._name = name  # type: str
        self._id = uuid.uuid4()  # type: uuid.UUID

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
        """Default getter for the value name"""
        return self._name

    @name.setter
    def name(self, new_name: str):
        if isinstance(new_name, str):
            self._name = new_name
        else:
            raise ValueError("The name of a value must be a string")

    @property
    def unit(self) -> str:
        """Default getter for the string representation of the units"""
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
        return self.value == wrap_operand(other).value

    def __gt__(self, other):
        return self.value > wrap_operand(other).value

    def __ge__(self, other):
        return self.value >= wrap_operand(other).value

    def __lt__(self, other):
        return self.value < wrap_operand(other).value

    def __le__(self, other):
        return self.value <= wrap_operand(other).value

    def __neg__(self):
        return DerivedValue(Formula(lit.NEG, [self]))

    def __pow__(self, power):
        return DerivedValue(Formula(lit.POW, [self, wrap_operand(power)]))

    def __rpow__(self, other):
        return DerivedValue(Formula(lit.POW, [wrap_operand(other), self]))

    def __add__(self, other):
        return DerivedValue(Formula(lit.ADD, [self, wrap_operand(other)]))

    def __radd__(self, other):
        return DerivedValue(Formula(lit.ADD, [wrap_operand(other), self]))

    def __sub__(self, other):
        return DerivedValue(Formula(lit.SUB, [self, wrap_operand(other)]))

    def __rsub__(self, other):
        return DerivedValue(Formula(lit.SUB, [wrap_operand(other), self]))

    def __mul__(self, other):
        return DerivedValue(Formula(lit.MUL, [self, wrap_operand(other)]))

    def __rmul__(self, other):
        return DerivedValue(Formula(lit.MUL, [wrap_operand(other), self]))

    def __truediv__(self, other):
        return DerivedValue(Formula(lit.DIV, [self, wrap_operand(other)]))

    def __rtruediv__(self, other):
        return DerivedValue(Formula(lit.DIV, [wrap_operand(other), self]))

    def scale(self, factor, update_units=True):
        """Scale the value and error of this instance by a factor

        This method can be used to scale up or down a value, usually due to the change of
        unit (for example, change from using [m] to using [mm]. The uncertainty of the value
        will be updated accordingly. By default, the units will be updated if applicable.
        This feature can be disabled by passing update_units=False

        """

    def derivative(self, other: "ExperimentalValue") -> float:
        """Finds the derivative with respect to another ExperimentalValue

        This method is usually called from a DerivedValue object. When you take the derivative
        of a measured value with respect to anything other than itself, the return value should
        be 0. The derivative of any quantity with respect to itself is always 1

        Args:
            other (ExperimentalValue): the target of the differentiation

        """
        if not isinstance(other, ExperimentalValue):
            raise ValueError("You can only find the derivative of a variable with respect to another "
                             "qexpy defined variable.")
        return 1 if self._id == other._id else 0

    def get_units(self) -> Dict[str, int]:
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

    The MeasuredValue class represents a single measurement, or more precisely, a single data point
    in an experiment. For backwards compatibility, this class is given an alias "Measurement"

    Usage:
        Measurement(12, 1) -> 12 +/- 1
        Measurement(12) -> 12 +/- 0

    You can also specify the name and unit of the value as keyword arguments.

    For example:
        Measurement(12, 1, name="length", unit="m") -> length = 12 +/- 1 [m]

    Usually during an experiment, the user may wish to take multiple measurements on the same quantity,
    and use their average as the final value. This is also supported:

    For example:
        Measurement([5.6, 4.8, 6.1, 4.9, 5.1]) -> 5.3000 +/- 0.5431

    The user does not need to pass in the uncertainties on each measurement. By default, the mean and
    error on the mean is used as the value-error pair of this variable. For more details, see comments
    on the RepeatedlyMeasuredValue class.

    """

    def __new__(cls, data: Union[Real, List[Real]], error: Union[Real, List[Real]] = None, unit="", name=""):
        """Default constructor for a MeasuredValue

        This method is called when the user instantiate a MeasuredValue object by calling its alias
        Measurement. The user can create a MeasuredValue from a single measurement or an array of
        repeated measurements on a single quantity (to be averaged).

        When two values are passed to this method, the first argument will be recognized as the value,
        the second as the uncertainty. If the second value is not provided, the uncertainty is by default
        set to 0. If a list of values is passed to this method, the mean and standard deviation of the
        value will be calculated and preserved.

        """
        if isinstance(data, utils.ARRAY_TYPES) and error is None:
            instance = super().__new__(RepeatedlyMeasuredValue)
            instance.__init__(data, unit=unit, name=name)
        elif isinstance(data, utils.ARRAY_TYPES) and isinstance(error, (Real, utils.ARRAY_TYPES)):
            instance = super().__new__(RepeatedlyMeasuredValue)
            instance.__init__(data, error=error, unit=unit, name=name)
        elif isinstance(data, Real) and error is None:
            instance = super().__new__(cls)
            instance.__init__(float(data), 0.0, unit=unit, name=name)
        elif isinstance(data, Real) and isinstance(error, Real):
            instance = super().__new__(cls)
            instance.__init__(float(data), float(error), unit=unit, name=name)
        else:
            raise ValueError("Invalid Arguments! Measurements can be recorded either using a center value "
                             "with its uncertainty or an array of repeated measurements on the same quantity.")
        return instance

    def __init__(self, value=0.0, error=0.0, unit="", name=""):
        """Default initializer for a regular measurement"""
        super().__init__(unit, name)
        self._values[lit.RECORDED] = ValueWithError(value, error)
        ExperimentalValue._register[self._id] = self

    @property
    def value(self) -> float:
        """Gets the value for this measurement"""
        return self._values[lit.RECORDED].value

    @value.setter
    def value(self, new_value: Real):
        """Modifies the value of a measurement"""
        if isinstance(new_value, Real):
            self._values[lit.RECORDED] = ValueWithError(new_value, self._values[lit.RECORDED].error)
        else:
            raise ValueError("You can only set the value of a measurement to a number")
        if isinstance(self, RepeatedlyMeasuredValue):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the value of a repeated measurement. Doing so has "
                          "caused you to lose the original list of raw measurement data")
            self.__class__ = MeasuredValue  # casting it to base class

    @property
    def error(self) -> float:
        """Gets the uncertainty on the measurement"""
        return self._values[lit.RECORDED].error

    @error.setter
    def error(self, new_error: Real):
        """Modifies the value of a measurement"""
        if isinstance(new_error, Real) and float(new_error) > 0:
            self._values[lit.RECORDED] = ValueWithError(self._values[lit.RECORDED].value, new_error)
        else:
            raise ValueError("You can only set the error of a measurement to a positive number")
        if isinstance(self, RepeatedlyMeasuredValue):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the uncertainty of a repeated measurement.")

    @property
    def relative_error(self) -> float:
        """Gets the relative error (error/mean) of a MeasuredValue object."""
        return self.error / self.value if self.value != 0 else 0.

    @relative_error.setter
    def relative_error(self, relative_error: Real):
        """Sets the relative error (error/mean) of a MeasuredValue object.

        Args:
            relative_error (Real): The new uncertainty

        """
        if isinstance(relative_error, Real) and float(relative_error) > 0:
            new_error = self.value * float(relative_error)
            self._values[lit.RECORDED] = ValueWithError(self.value, new_error)
        else:
            raise ValueError("The relative uncertainty of a measurement has to be a positive number")
        if isinstance(self, RepeatedlyMeasuredValue):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the uncertainty of a repeated measurement.")


class RepeatedlyMeasuredValue(MeasuredValue):
    """Class to store the result of repeated measurements on a single quantity

    This class is instantiated when the user takes multiple measurements on a single quantity.
    It is just like any regular MeasuredValue object, with the original array of measurements
    preserved as "raw_data".

    The mean and error on the mean is used as the default value and error of this measurement.
    The user also has an option of using the standard deviation as the error on this measurement.
    If the corresponding uncertainties for the list of raw data is also passed in, the user can
    choose the error weighted mean and propagated error as the value-error pair.

    The user can also manually set the uncertainty of this object. However, if the user choose
    to manually override the value of this measurement. The original raw data will be lost, and
    the instance will be casted to its parent class MeasuredValue

    Attributes:
        _raw_data (MeasuredValueArray): the original list of raw measurements
        _mean (float): the raw mean of the raw measurements
        _std (float): the standard derivative of set of measurements
        _error_on_mean (float): the error on the mean of the set of measurements

    """

    def __init__(self, measurement_array: List[Real], error: Union[Real, List[Real]] = None, unit="", name=""):
        """default constructor for a repeatedly measured value"""

        # check validity of inputs
        if isinstance(error, utils.ARRAY_TYPES) and len(error) != len(measurement_array):
            raise ValueError("The error array and raw measurements are not of the same length")

        # call parent constructor
        super().__init__(unit=unit, name=name)

        # initialize raw data and its corresponding uncertainties
        self._raw_data = MeasuredValueArray(measurement_array, error, unit, name)

        # calculate its statistical properties
        self._mean = self._raw_data.mean().value
        self._std = self._raw_data.std(ddof=1)
        self._error_on_mean = self._raw_data.error_on_mean()

        # sets the value and error pair adopted for this object
        self._values[lit.RECORDED] = ValueWithError(self._mean, self._error_on_mean)

    @property
    def raw_data(self) -> np.ndarray:
        """Gets the raw data that was used to generate this measurement"""
        return self._raw_data.values

    @property
    def std(self) -> float:
        """Getter for the standard deviation"""
        return self._std

    @property
    def error_on_mean(self) -> float:
        """Getter for the error on the mean"""
        return self._error_on_mean

    @property
    def mean(self) -> float:
        """default getter for the mean of raw measurements"""
        return self._mean

    @property
    def error_weighted_mean(self) -> float:
        """default getter for the error weighted mean if errors are specified"""
        return self._raw_data.error_weighted_mean()

    @property
    def propagated_error(self) -> float:
        """getter for the error propagated with errors passed in if present"""
        return self._raw_data.propagated_error()

    def use_std_for_uncertainty(self):
        """Sets the uncertainty of this value to the standard deviation"""
        value = self._values[lit.RECORDED]
        self._values[lit.RECORDED] = ValueWithError(value.value, self._std)

    def use_error_on_mean_for_uncertainty(self):
        """Sets the uncertainty of this value to the error on the mean

        This is default behaviour, because the reason for taking multiple measurements is to
        reduce the uncertainty, not to find out what the uncertainty of a single measurement is.

        """
        value = self._values[lit.RECORDED]
        self._values[lit.RECORDED] = ValueWithError(value.value, self._error_on_mean)

    def use_error_weighted_mean_as_value(self):
        """Sets the value of this object to the error weighted mean"""
        value = self._values[lit.RECORDED]
        error_weighted_mean = self.error_weighted_mean
        if error_weighted_mean != 0:
            self._values[lit.RECORDED] = ValueWithError(error_weighted_mean, value.error)
        else:
            warnings.warn("This measurement was not taken with individual uncertainties. The error "
                          "weighted mean is not applicable")

    def use_propagated_error_for_uncertainty(self):
        """Sets the uncertainty of this object to the weight propagated error"""
        value = self._values[lit.RECORDED]
        propagated_error = self.propagated_error
        if propagated_error != 0:
            self._values[lit.RECORDED] = ValueWithError(value.value, propagated_error)
        else:
            warnings.warn("This measurement was not taken with individual uncertainties. The propagated "
                          "error is not applicable")

    def show_histogram(self):
        """Plots the raw measurement data in a histogram

        For the result of repeated measurements of a single quantity, the raw measurement
        data is preserved. With this method, you can visualize these values in a histogram.
        with lines corresponding to the mean and the range covered by one standard deviation

        """


class DerivedValue(ExperimentalValue):
    """Values derived from operations with other experimental values

    This class is not to be instantiated directly. It will be created when operations are done
    with other ExperimentalValue objects. The error of the DerivedValue object will be propagated
    from the original ExperimentalValue objects.

    The DerivedValue object is preserves information on how it was derived. This is stored in the
    "_formula" attribute, which is a named tuple "Formula" with the operator and the operands specified.
    If the operands are also DerivedValues, their "formula" can also be accessed, which works like
    an expression tree

    Each "Formula" has an "operator" and an "operands". The "operator" is the name of the operation, such
    as "ADD" or "MUL". The "operands" is a list of operands on which the operation is executed. An
    "operand" can be either a value, or an ExperimentalValue object

    Attributes:
        _error_method (ErrorMethod): the error method used for this value
        _is_error_method_specified (bool): true if the user specified an error method for this value,
            false if default was used
        _formula (Formula): the formula of this object

    """

    def __init__(self, formula: Formula, error_method: settings.ErrorMethod = None):
        """The default constructor for the result of an operation

        The value and uncertainty of this variable is not calculated in the constructor. It will only
        be calculated when it is requested by the user, either by explicitly calling the accessors
        or by printing the string representation of the value.

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

        # store the formula in the instance
        self._formula = formula
        self._units = op.propagate_units(formula)

    @property
    def value(self) -> float:
        """Gets the value for this calculated quantity"""
        return self.__get_value_error_pair().value

    @value.setter
    def value(self, new_value: Real):
        """Modifies the value of this quantity"""
        if isinstance(new_value, Real):
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
    def error(self, new_error: Real):
        """Modifies the uncertainty of this quantity

        This is not recommended. Doing so will cause this object to be casted to MeasuredValue

        """
        if isinstance(new_error, Real) and float(new_error) > 0:
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
    def relative_error(self, relative_error: Real):
        """Sets the relative error (error/mean) of a DerivedValue object.

        This is not recommended. Doing so will cause this object to be casted to MeasuredValue

        """
        if isinstance(relative_error, Real) and float(relative_error) > 0:
            new_error = self.value * float(relative_error)
            self._values = {lit.RECORDED: ValueWithError(self.value, new_error)}  # reset the values of this object
            # TODO: don't remember to reset the relations between this instance and other values
            warnings.warn("You are trying to modify the value of a calculated quantity. Doing so has caused "
                          "this value to be regarded as a Measurement and all other information lost")
            self.__class__ = MeasuredValue  # casting it to MeasuredValue
        else:
            raise ValueError("The relative uncertainty of a ExperimentalValue has to be a positive number")

    def derivative(self, other: ExperimentalValue) -> float:
        """Finds the derivative with respect to another ExperimentalValue"""

        if self._id == other._id:
            return 1  # the derivative of anything with respect to itself is 1
        return op.differentiate(self._formula, other)

    def __get_value_error_pair(self) -> ValueWithError:
        """Gets the value-error pair for the current specified method"""

        if self._is_error_method_specified:
            error_method = self._error_method
        else:
            error_method = settings.get_error_method()

        # calculate the values if not present
        if error_method == settings.ErrorMethod.DERIVATIVE and lit.DERIVATIVE_PROPAGATED not in self._values:
            self._values[lit.DERIVATIVE_PROPAGATED] = op.get_derivative_propagated_value_and_error(self._formula)
        elif error_method == settings.ErrorMethod.MONTE_CARLO and lit.MONTE_CARLO_PROPAGATED not in self._values:
            self._values[lit.MONTE_CARLO_PROPAGATED] = op.get_monte_carlo_propagated_value_and_error(self._formula)

        return self._values[error_method.value]


class Constant(ExperimentalValue):
    """A value with no uncertainty

    This is created when a constant (int, float, etc.) is used in operation with another
    ExperimentalValue. This class is instantiated before calculating operations to ensure
    objects can be combined.

    """

    def __init__(self, value, unit="", name=""):
        super().__init__(unit=unit, name=name)
        if isinstance(value, Real):
            self._values[lit.RECORDED] = ValueWithError(value, 0)
        else:
            raise ValueError("The value must be a number")

    @property
    def value(self):
        return self._values[lit.RECORDED].value

    @property
    def error(self):
        return 0

    def derivative(self, other) -> 0:
        return 0  # the derivative of a constant with respect to anything is 0


class MeasuredValueArray(np.ndarray):
    """An array of measurements

    This class is used to hold a series of measurements. It can be used for data analysis,
    fitting, and plotting.

    """

    def __new__(cls, data: List[Real], error: Union[List[Real], Real] = None, unit="", name=""):
        """Default constructor for a MeasuredValueArray

        This is used instead of __init__ for object initialization. This is the convention for
        subclassing numpy arrays. The MeasuredValueArray class is not to be called directly. The
        constructor wrapper MeasurementArray is recommended as it's easier to use

        """

        # initialize raw data and its corresponding uncertainties
        measurements = np.array(data)
        if error is None:
            error_array = np.asarray(list(0 for _ in measurements))
        elif isinstance(error, Real):
            error_array = np.asarray(list(float(error) for _ in measurements))
        elif isinstance(error, utils.ARRAY_TYPES) and len(error) == len(data):
            error_array = np.asarray(error)
        else:
            raise ValueError("The error of a MeasurementArray can only be a single value, or an "
                             "array of values that are of the same length as the measurement array.")

        measured_values = list(MeasuredValue(val, err, unit, name) for val, err in zip(measurements, error_array))
        obj = np.asarray(measured_values).view(MeasuredValueArray)
        return obj

    @property
    def name(self) -> str:
        """Default getter for the value name"""
        return self[0].name

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise ValueError("The name of a value must be a string")
        for data in self:
            data.name = new_name

    @property
    def unit(self) -> str:
        """Default getter for the string representation of the units"""
        return self[0].unit

    @unit.setter
    def unit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise ValueError("You can only set the unit of a value using its string representation")
        new_unit = units.parse_units(new_unit)
        for data in self:
            data._units = new_unit

    @property
    def values(self) -> np.ndarray:
        """Gets an array of values"""
        return np.asarray(list(data.value for data in self))

    @property
    def errors(self) -> np.ndarray:
        """Gets the error array"""
        return np.asarray(list(data.error for data in self))

    def mean(self, **kwargs) -> MeasuredValue:
        """Gets the mean of the array"""
        result = np.mean(self.values)
        error = self.error_on_mean()
        name = "Mean of {}".format(self.name) if self.name else ""
        return MeasuredValue(float(result), error, self.unit, name)

    def std(self, ddof=1, **kwargs) -> float:
        """Gets the standard deviation of this array"""
        return float(np.std(self.values, ddof=ddof))

    def sum(self, **kwargs) -> MeasuredValue:
        """Gets the sum of the array"""
        result = np.sum(self.values)
        error = np.sqrt(np.sum(self.errors ** 2))
        return MeasuredValue(float(result), float(error), self.unit, self.name)

    def error_on_mean(self) -> float:
        """Gets the error on the mean of this array"""
        return self.std() / m.sqrt(self.size)

    def error_weighted_mean(self) -> float:
        """Gets the error weighted mean of this array"""
        if any(err == 0 for err in self.errors):
            warnings.warn("One or more of the errors are 0, the error weighted mean cannot be calculated.")
            return 0
        weights = np.asarray(list(1 / (err ** 2) for err in self.errors))
        return float(np.sum(weights * self.values) / np.sum(weights))

    def propagated_error(self) -> float:
        """Gets the propagated error from the error weighted mean calculation"""
        if any(err == 0 for err in self.errors):
            warnings.warn("One or more of the errors are 0, the propagated error cannot be calculated.")
            return 0
        weights = np.asarray(list(1 / (err ** 2) for err in self.errors))
        return 1 / np.sqrt(np.sum(weights))


def get_variable_by_id(variable_id: uuid.UUID) -> ExperimentalValue:
    """Internal method used to retrieve an ExperimentalValue instance with its ID"""
    return ExperimentalValue._register[variable_id]


def set_covariance(var1: Union[uuid.UUID, MeasuredValue], var2: Union[uuid.UUID, MeasuredValue], cov: Real = None):
    """Sets the covariance of two MeasuredValue objects

    Users can manually set the covariance of two MeasuredValue objects. If the two objects are
    both RepeatedMeasurement with raw_data of the same length, the user can omit the input
    value, and the covariance will be calculated automatically. If a value is specified, the
    calculated value will be overridden with the user specified value

    Once the covariance is set or calculated, it will be stored in a module level register for
    easy reference during error propagation. By default the error propagation process assumes
    that the covariance between any two values is 0, unless specified or calculated

    """
    if isinstance(var1, uuid.UUID) and isinstance(var2, uuid.UUID):
        var1 = ExperimentalValue._register[var1]
        var2 = ExperimentalValue._register[var2]
    if not isinstance(var1, MeasuredValue) or not isinstance(var2, MeasuredValue):
        raise ValueError("You can only set the covariance between two Measurements")
    if var1.error == 0 or var2.error == 0:
        raise ArithmeticError("Constants or values with no standard deviation don't correlate with other values")
    if isinstance(var1, RepeatedlyMeasuredValue) and isinstance(var2, RepeatedlyMeasuredValue):
        covariance = cov
        if len(var1.raw_data) == len(var2.raw_data) and cov is None:
            covariance = utils.calculate_covariance(var1.raw_data, var2.raw_data)
        elif len(var1.raw_data) != len(var2.raw_data) and cov is None:
            raise ValueError("The two arrays of repeated measurements are not of the same length")
        correlation = covariance / (var1.std * var2.std)
    elif cov is None:
        raise ValueError("The covariance value is not provided")
    else:
        covariance = cov
        correlation = cov / (var1.error * var2.error)

    # check that the result makes sense
    if correlation > 1 or correlation < -1:
        raise ValueError("The covariance provided: {} is non-physical".format(cov))

    # set relations in the module level register
    id_string = "_".join(sorted(map(str, [var1._id, var2._id])))
    ExperimentalValue._correlations[id_string] = Correlation((var1, var2), correlation, covariance)


def get_covariance(var1: Union[uuid.UUID, MeasuredValue], var2: Union[uuid.UUID, MeasuredValue]) -> float:
    """Finds the covariance of two ExperimentalValues"""
    if isinstance(var1, uuid.UUID) and isinstance(var2, uuid.UUID):
        var1 = ExperimentalValue._register[var1]
        var2 = ExperimentalValue._register[var2]
    if var1.error == 0 or var2.error == 0:
        return 0  # constants don't correlate with anyone
    if isinstance(var1, MeasuredValue) and isinstance(var2, MeasuredValue):
        return __get_covariance(var1, var2)
    return 0  # TODO: implement covariance propagation for DerivedValues


def set_correlation(var1: Union[uuid.UUID, MeasuredValue], var2: Union[uuid.UUID, MeasuredValue], cor: Real = None):
    """Sets the correlation factor between two MeasuredValue objects

    See Also:
        set_covariance

    """
    if isinstance(var1, uuid.UUID) and isinstance(var2, uuid.UUID):
        var1 = ExperimentalValue._register[var1]
        var2 = ExperimentalValue._register[var2]
    # check that the input makes sense
    if cor and (float(cor) > 1 or float(cor) < -1):
        raise ValueError("The correlation factor provided: {} is non-physical.".format(cor))

    if not isinstance(var1, MeasuredValue) or not isinstance(var2, MeasuredValue):
        raise ValueError("You can only set the correlation factor between two Measurements")
    if var1.error == 0 or var2.error == 0:
        raise ArithmeticError("Constants or values with no standard deviation don't correlate with other values")
    if isinstance(var1, RepeatedlyMeasuredValue) and isinstance(var2, RepeatedlyMeasuredValue):
        if len(var1.raw_data) == len(var2.raw_data) and cor is None:
            covariance = utils.calculate_covariance(var1.raw_data, var2.raw_data)
            correlation = covariance / (var1.std * var2.std)
        elif len(var1.raw_data) != len(var2.raw_data) and cor is None:
            raise ValueError("The two arrays of repeated measurements are not of the same length")
        else:
            correlation = cor
            covariance = cor * (var1.std * var2.std)
    elif cor is None:
        raise ValueError("The covariance value is not provided")
    else:
        covariance = cor * (var1.error * var2.error)
        correlation = cor

    # set relations in the module level register
    id_string = "_".join(sorted(map(str, [var1._id, var2._id])))
    ExperimentalValue._correlations[id_string] = Correlation((var1, var2), correlation, covariance)


def get_correlation(var1: Union[uuid.UUID, MeasuredValue], var2: Union[uuid.UUID, MeasuredValue]) -> float:
    """Finds the correlation factor of two ExperimentalValues"""
    if isinstance(var1, uuid.UUID) and isinstance(var2, uuid.UUID):
        var1 = ExperimentalValue._register[var1]
        var2 = ExperimentalValue._register[var2]
    if var1.error == 0 or var2.error == 0:
        return 0  # constants don't correlate with anyone
    if isinstance(var1, MeasuredValue) and isinstance(var2, MeasuredValue):
        return __get_correlation(var1, var2)

    # TODO: implement covariance propagation for DerivedValues
    return 0


def wrap_operand(operand: Union[Real, ExperimentalValue]) -> ExperimentalValue:
    """Wraps the operand of an operation as an ExperimentalValue instance

    If the operand is a number, construct a Constant object with this value. If the operand
    is an ExperimentalValue object, return the object directly.

    """
    if isinstance(operand, Real):
        return Constant(operand)
    if isinstance(operand, ExperimentalValue):
        return operand
    raise ValueError("Operand of type {} for this operation is invalid!".format(type(operand)))


def _get_formula(operand: Union[Real, ExperimentalValue]) -> Union[Formula, MeasuredValue, Constant]:
    """Takes the input and returns the formula representation

    If the operand is a number, construct a Constant object with this value, if the operand is a
    Constant or MeasuredValue, return the object directly. If the operand is a derived value, return
    the formula of that object

    """
    if isinstance(operand, Real):
        return Constant(operand)
    if isinstance(operand, (Constant, MeasuredValue)):
        return operand
    if isinstance(operand, DerivedValue):
        return operand._formula
    raise ValueError("Operand of type {} for this operation is invalid!".format(type(operand)))


def __get_covariance(var1: Union[uuid.UUID, MeasuredValue], var2: Union[uuid.UUID, MeasuredValue]) -> float:
    """Gets the covariance factor between two values"""
    if isinstance(var1, uuid.UUID) and isinstance(var2, uuid.UUID):
        var1 = ExperimentalValue._register[var1]
        var2 = ExperimentalValue._register[var2]
    if var1._id == var2._id:
        # the covariance between a value and itself is the variance
        return var1.std ** 2 if isinstance(var1, RepeatedlyMeasuredValue) else var1.error ** 2
    id_string = "_".join(sorted(map(str, [var1._id, var2._id])))
    if id_string in ExperimentalValue._correlations:
        return ExperimentalValue._correlations[id_string].covariance

    return 0


def __get_correlation(var1: Union[uuid.UUID, MeasuredValue], var2: Union[uuid.UUID, MeasuredValue]) -> float:
    """Gets the correlation factor between two values"""
    if isinstance(var1, uuid.UUID) and isinstance(var2, uuid.UUID):
        var1 = ExperimentalValue._register[var1]
        var2 = ExperimentalValue._register[var2]
    if var1._id == var2._id:
        return 1  # values have unit correlation with themselves
    id_string = "_".join(sorted(map(str, [var1._id, var2._id])))
    if id_string in ExperimentalValue._correlations:
        return ExperimentalValue._correlations[id_string].correlation

    return 0
