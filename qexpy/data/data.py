"""Module containing data structures for experimental values

This module defines ExperimentalValue and all its sub-classes. Experimental measurements
and their corresponding uncertainties can be recorded as a instances of this class. Any
operations done with these instances will have the properly propagated errors and units.

"""

import warnings
import uuid
import collections
import functools
from numbers import Real
from typing import Dict, Union, List, Tuple
import math as m
import numpy as np

import qexpy.utils.utils as utils
import qexpy.data.operations as op
import qexpy.settings.literals as lit
import qexpy.utils.units as units
import qexpy.utils.printing as printing
import qexpy.settings.settings as settings
from qexpy.utils.exceptions import InvalidArgumentTypeError, UndefinedOperationError

# a simple structure for a value-error pair
ValueWithError = collections.namedtuple("ValueWithError", "value, error")

# a data structure to store an expression tree of a formula
Formula = collections.namedtuple("Formula", "operator, operands")

# a data structure to store the correlation between two values
Correlation = collections.namedtuple("Correlation", "values, correlation, covariance")


def check_operand_type(operation):
    """wrapper decorator for undefined operation error reporting"""

    def check_operand_type_wrapper(func):
        @functools.wraps(func)
        def operation_wrapper(*args):
            try:
                return func(*args)
            except TypeError:
                raise UndefinedOperationError(operation, got=args, expected="real numbers or QExPy defined values")

        return operation_wrapper

    return check_operand_type_wrapper


class ExperimentalValue:
    """Base class for objects with a value and an uncertainty

    The ExperimentalValue is a wrapper class for any individual value involved in an experiment
    and subsequent data analysis. Each ExperimentalValue has a center value and an uncertainty,
    and optionally, a name and a unit. When working with QExPy, any measurements recorded with
    QExPy and the result of any data analysis done with these measurements will all be instances
    of this class.

    Examples:
        >>> import qexpy as q
        >>> a = q.Measurement(302, 5)

        The ExperimentalValue object has the following basic properties
        >>> a.value
        302
        >>> a.error
        5
        >>> a.relative_error
        0.016556291390728478

        These properties can be changed
        >>> a.value = 303
        >>> a.value
        303
        >>> a.relative_error = 0.05
        >>> a.error
        15.15

        You can specify the name or the units of a value
        >>> a.name = "force"
        >>> a.unit = "kg*m^2/s^2"

        The string representation of the value will include the name and units if specified
        >>> print(a)
        force = 300 +/- 20 [kg⋅m^2⋅s^-2]

        You can also specify how you want the values or the units to be printed
        >>> q.set_print_style("scientific")
        >>> q.set_unit_style("fraction")
        >>> q.set_sig_figs_for_error(2)
        >>> print(a)
        force = (3.03 +/- 0.15) * 10^2 [kg⋅m^2/s^2]

    """

    # module level register that stores references to all instantiated MeasuredValue and
    # DerivedValue objects during a session.
    _register: Dict[uuid.UUID, "ExperimentalValue"] = {}

    # stores the correlation between values, where the key is the unique IDs of the two
    # instances combined.
    _correlations: Dict[str, Correlation] = {}

    def __init__(self, unit="", name=""):
        """Constructor for ExperimentalValue

        The ExperimentalValue is practically an abstract class. It should not be instantiated directly

        """

        # Stores the value/error pairs corresponding to their source. User recorded values are stored
        # with the key "recorded", derived values are stored with the key "derivative" or "monte-carlo",
        # depending which error propagation method the user specifies.
        self._values = {}  # type: Dict[str, ValueWithError]

        # Stores each unit string and their corresponding exponents.
        self._units = units.parse_units(unit) if unit else {}  # type: Dict[str, int]

        # The name of this quantity if given
        self._name = name  # type: str

        # Each instance is given a unique ID for easy reference
        self._id = uuid.uuid4()  # type: uuid.UUID

    def __str__(self):
        """The string representation of this quantity

        The default string representation for an ExperimentalValue is "name = value +/- error [unit]",
        but if the name or unit is not specified, they won't be shown

        """
        name_string = "{} = ".format(self.name) if self.name else ""
        unit_string = " [{}]".format(self.unit) if self.unit else ""
        return "{}{}{}".format(name_string, self.print_value(), unit_string)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.print_value())

    @property
    def value(self) -> float:
        """The center value of this quantity"""
        return 0

    @property
    def error(self) -> float:
        """The uncertainty of this quantity"""
        return 0

    @property
    def relative_error(self) -> float:
        """The ratio of the uncertainty to its center value"""
        return 0

    @property
    def name(self) -> str:
        """The name of this quantity"""
        return self._name

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise InvalidArgumentTypeError("name", got=new_name, expected="string")
        self._name = new_name

    @property
    def unit(self) -> str:
        """The unit of this quantity"""
        if not self._units:
            return ""
        return units.construct_unit_string(self._units)

    @unit.setter
    def unit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise InvalidArgumentTypeError("unit", got=new_unit, expected="string")
        self._units = units.parse_units(new_unit) if new_unit else {}

    @check_operand_type("==")
    def __eq__(self, other):
        return self.value == wrap_operand(other).value

    @check_operand_type(">")
    def __gt__(self, other):
        return self.value > wrap_operand(other).value

    @check_operand_type(">=")
    def __ge__(self, other):
        return self.value >= wrap_operand(other).value

    @check_operand_type("<")
    def __lt__(self, other):
        return self.value < wrap_operand(other).value

    @check_operand_type("<=")
    def __le__(self, other):
        return self.value <= wrap_operand(other).value

    def __neg__(self):
        return DerivedValue(Formula(lit.NEG, [self]))

    @check_operand_type("pow")
    def __pow__(self, power):
        return DerivedValue(Formula(lit.POW, [self, wrap_operand(power)]))

    @check_operand_type("pow")
    def __rpow__(self, other):
        return DerivedValue(Formula(lit.POW, [wrap_operand(other), self]))

    @check_operand_type("+")
    def __add__(self, other):
        return DerivedValue(Formula(lit.ADD, [self, wrap_operand(other)]))

    @check_operand_type("+")
    def __radd__(self, other):
        return DerivedValue(Formula(lit.ADD, [wrap_operand(other), self]))

    @check_operand_type("-")
    def __sub__(self, other):
        return DerivedValue(Formula(lit.SUB, [self, wrap_operand(other)]))

    @check_operand_type("-")
    def __rsub__(self, other):
        return DerivedValue(Formula(lit.SUB, [wrap_operand(other), self]))

    @check_operand_type("*")
    def __mul__(self, other):
        return DerivedValue(Formula(lit.MUL, [self, wrap_operand(other)]))

    @check_operand_type("*")
    def __rmul__(self, other):
        return DerivedValue(Formula(lit.MUL, [wrap_operand(other), self]))

    @check_operand_type("/")
    def __truediv__(self, other):
        return DerivedValue(Formula(lit.DIV, [self, wrap_operand(other)]))

    @check_operand_type("/")
    def __rtruediv__(self, other):
        return DerivedValue(Formula(lit.DIV, [wrap_operand(other), self]))

    def scale(self, factor, update_units=True):
        """Scale the value and error of this instance by a factor

        This method can be used to scale up or down a value, usually called when the user wish to
        change the unit of this value (e.g. from [m] to [km]). Using this method, the uncertainty
        of this quantity will be undated accordingly. By default, the unit will also be scaled if
        applicable, but this can be disabled by passing "update_units=False" as an argument

        This method is not implemented yet. Right now this is just a placeholder.

        """

    def derivative(self, other: "ExperimentalValue") -> float:
        """Calculates the derivative of this quantity with respect to another

        The derivative of any value with respect to itself is 1, and for unrelated values, the
        derivative is always 0. This method is typically called from a DerivedValue, to find out
        its derivative with respect to one of the measurements it's derived from.

        Args:
            other (ExperimentalValue): the target for finding the derivative

        """
        if not isinstance(other, ExperimentalValue):
            raise InvalidArgumentTypeError("derivative()", got=other, expected="ExperimentalValue")
        return 1 if self._id == other._id else 0

    def print_value(self) -> str:
        """Helper method that prints the value-error pair in proper format"""
        res = printing.get_printer()(self.value, self.error)
        return res


class MeasuredValue(ExperimentalValue):
    """Class for any user recorded values with uncertainties, alias: Measurement

    A MeasuredValue (Measurement) object represents a single measured quantity with a center value
    and an uncertainty. For backwards compatibility, this class is given an alias "Measurement".
    The user is encouraged to call "Measurement" instead of "MeasuredValue" to record a measurement.

    Args:
        data: the value of this measurement. It's either a real number or an array of real numbers that
            represents a series of repeated measurements taken on this single quantity
        error: the uncertainty of this measurement. It's either a real number or an array of real numbers
            if the data is also an array of repeated measurements.
        unit: the string representation of the unit, e.g. "kg*m^2/s^2"
        name: the name of this quantity, e.g. "length"

    Examples:
        >>> import qexpy as q

        You can create a measurement with a value and an uncertainty
        >>> a = q.Measurement(12, 1)
        >>> print(a)
        12 +/- 1

        If the uncertainty is not specified, it's by default set to 0
        >>> b = q.Measurement(12)
        >>> print(b)
        12 +/- 0

        You can also specify the name and unit of this quantity
        >>> c = q.Measurement(302, 5, unit="kg*m^2/s^2", name="force")
        >>> print(c)
        force = 302 +/- 5 [kg⋅m^2⋅s^-2]

        For repeated measurements, simply pass an array into the constructor as data, and the average will
        be taken as the center value for this measurement, with the error on the mean as the uncertainty.
        >>> q.reset_default_configuration()
        >>> q.set_sig_figs_for_error(4)
        >>> d = q.Measurement([5.6, 4.8, 6.1, 4.9, 5.1])
        >>> print(d)
        5.3000 +/- 0.5431

        More examples for recording repeated measurements are documented under RepeatedlyMeasuredValue

    See Also:
        :py:class:`.RepeatedlyMeasuredValue`

    """

    def __new__(cls, data: Union[Real, List[Real]], error: Union[Real, List[Real]] = None, unit="", name=""):
        if isinstance(data, Real):
            instance = super().__new__(cls)
        elif isinstance(data, utils.ARRAY_TYPES):
            instance = super().__new__(RepeatedlyMeasuredValue)
        else:
            raise InvalidArgumentTypeError("creating a Measurement", got=data, expected="real number or array")
        return instance

    def __init__(self, data, error=None, unit="", name=""):
        if error is not None and not isinstance(error, Real):
            raise InvalidArgumentTypeError("setting uncertainties", got=error, expected="real number or array")
        super().__init__(unit, name)
        self._values[lit.RECORDED] = ValueWithError(data, float(error) if error else 0.0)
        ExperimentalValue._register[self._id] = self

    @property
    def value(self) -> float:
        """The value of this measurement"""
        return self._values[lit.RECORDED].value

    @value.setter
    def value(self, new_value: Real):
        if not isinstance(new_value, Real):
            raise InvalidArgumentTypeError("value", got=new_value, expected="real number")
        self._values[lit.RECORDED] = ValueWithError(new_value, self._values[lit.RECORDED].error)

    @property
    def error(self) -> float:
        """The uncertainty of this measurement"""
        return self._values[lit.RECORDED].error

    @error.setter
    def error(self, new_error: Real):
        if not isinstance(new_error, Real):
            raise InvalidArgumentTypeError("error", got=new_error, expected="positive real number")
        if new_error < 0:
            raise ValueError("Invalid argument for error: {}, expecting: positive real number".format(new_error))
        self._values[lit.RECORDED] = ValueWithError(self._values[lit.RECORDED].value, new_error)

    @property
    def relative_error(self) -> float:
        """The relative error (uncertainty/center value) of this measurement"""
        return self.error / self.value if self.value != 0 else 0.

    @relative_error.setter
    def relative_error(self, relative_error: Real):
        if not isinstance(relative_error, Real):
            raise InvalidArgumentTypeError("error", got=relative_error, expected="positive real number")
        if relative_error < 0:
            raise ValueError("Invalid argument for error: {}, expecting: positive real number".format(relative_error))
        new_error = self.value * float(relative_error)
        self._values[lit.RECORDED] = ValueWithError(self.value, new_error)


class RepeatedlyMeasuredValue(MeasuredValue):
    """A MeasuredValue recorded with an array of repeated measurements

    There is no designated constructor for this class. When the user calls "Measurement" with an
    array of real numbers, this class is automatically instantiated.

    By default, the mean of the array of measurements and the standard error (error on the mean)
    is used as the value and uncertainty for this measurement when accessed.

    References:
        https://en.wikipedia.org/wiki/Standard_error
        https://en.wikipedia.org/wiki/Standard_deviation
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    See Also:
        :py:class:`.MeasuredValue`
        :py:class:`.ExperimentalValueArray`

    Examples:
        >>> import qexpy as q

        The most common way of recording a value with repeated measurements is to only give the
        center values for the measurements
        >>> a = q.Measurement([9, 10, 11])
        >>> print(a)
        10.0 +/- 0.6

        There are other statistical properties of the array of measurements
        >>> a.std
        1
        >>> a.error_on_mean
        0.5773502691896258

        You can choose to use the standard deviation as the uncertainty
        >>> a.use_std_for_uncertainty()
        >>> a.error
        1

        You can also specify individual uncertainties for the measurements
        >>> a = q.Measurement([10, 11], [0.1, 1])
        >>> print(a)
        10.5 +/- 0.5
        >>> a.error_weighted_mean
        10.00990099009901
        >>> a.propagated_mean
        0.09950371902099892

        You can choose which statistical properties to be used as the value/error as usual
        >>> a.use_error_weighted_mean_as_value()
        >>> a.use_propagated_error_for_uncertainty()
        >>> q.set_sig_figs_for_error(4)
        >>> print(a)
        10.00990 +/- 0.09950

    """

    def __init__(self, measurement_array: List[Real], error: Union[Real, List[Real]] = None, unit="", name=""):

        # check validity of inputs
        if isinstance(error, utils.ARRAY_TYPES) and len(error) != len(measurement_array):
            raise ValueError(
                "Cannot map {} uncertainties to {} measurements".format(len(error), len(measurement_array)))

        # Initialize raw data and its corresponding uncertainties. Internally, the array of measurements
        # is stored as a ExperimentalValueArray, however, the principal is that ExperimentalValueArray should be
        # used only to represent an array of measurements of different quantities.
        self._raw_data = ExperimentalValueArray(measurement_array, error, unit, name)

        # calculate its statistical properties
        self._mean = self._raw_data.mean().value
        self._std = self._raw_data.std(ddof=1)
        self._error_on_mean = self._raw_data.error_on_mean()

        # call parent constructor with mean and error on mean as value and uncertainty
        super().__init__(self._mean, self._error_on_mean, unit=unit, name=name)

    @property
    def value(self) -> float:
        """The value of this measurement"""
        return self._values[lit.RECORDED].value

    @value.setter
    def value(self, new_value: Real):
        if not isinstance(new_value, Real):
            raise InvalidArgumentTypeError("value", got=new_value, expected="real number")
        self._values[lit.RECORDED] = ValueWithError(new_value, self._values[lit.RECORDED].error)
        warnings.warn("\nYou are trying to set the mean of an array of repeated measurements, as a result, "
                      "\nthe original array of data is overridden.")
        self.__class__ = MeasuredValue

    @property
    def raw_data(self) -> np.ndarray:
        """Gets the raw data that was used to generate this measurement"""
        return self._raw_data.values if all(x.error == 0 for x in self._raw_data) else self._raw_data

    @property
    def std(self) -> float:
        """the standard deviation of the raw data"""
        return self._std

    @property
    def error_on_mean(self) -> float:
        """the error on the mean or the standard error"""
        return self._error_on_mean

    @property
    def mean(self) -> float:
        """the mean of raw measurements"""
        return self._mean

    @property
    def error_weighted_mean(self) -> float:
        """error weighted mean if individual errors are specified"""
        return self._raw_data.error_weighted_mean()

    @property
    def propagated_error(self) -> float:
        """error propagated with errors passed in if present"""
        return self._raw_data.propagated_error()

    def use_std_for_uncertainty(self):
        """Sets the uncertainty of this value to the standard deviation"""
        value = self._values[lit.RECORDED]
        self._values[lit.RECORDED] = ValueWithError(value.value, self._std)

    def use_error_on_mean_for_uncertainty(self):
        """Sets the uncertainty of this value to the error on the mean

        This is default behaviour, because the reason for taking multiple measurements is usually to
        minimize the uncertainty, not to find out what the uncertainty of a single measurement is.

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
            warnings.warn("\nThis measurement was not taken with individual uncertainties. The error "
                          "\nweighted mean is not applicable")

    def use_propagated_error_for_uncertainty(self):
        """Sets the uncertainty of this object to the weight propagated error"""
        value = self._values[lit.RECORDED]
        propagated_error = self.propagated_error
        if propagated_error != 0:
            self._values[lit.RECORDED] = ValueWithError(value.value, propagated_error)
        else:
            warnings.warn("\nThis measurement was not taken with individual uncertainties. The propagated "
                          "\nerror is not applicable")

    def show_histogram(self):
        """Plots the raw measurement data in a histogram"""


class DerivedValue(ExperimentalValue):
    """Values derived from other ExperimentalValue instances

    The user does not need to instantiate this class. It is automatically created When the user performs
    an operation using other ExperimentalValue instances, and it comes with the properly propagated
    uncertainties. The two available methods for error propagation are the regular derivative method, and
    the Monte Carlo simulation method.

    Internally, a DerivedValue object preserves information on how it is derived, so the user is able to
    make use of that information. For example, the user can find the derivative of a DerivedValue with
    respect to another ExperimentalValue that this value is derived from.

    Examples:
        >>> import qexpy as q

        First let's create some standard measurements
        >>> a = q.Measurement(5, 0.2)
        >>> b = q.Measurement(4, 0.1)
        >>> c = q.Measurement(6.3, 0.5)
        >>> d = q.Measurement(7.2, 0.5)

        Now we can perform operations on them
        >>> result = q.sqrt(c) * d - b / q.exp(a)
        >>> result
        DerivedValue(18 +/- 1)
        >>> result.value
        18.04490478513969
        >>> result.error
        1.4454463754287323

        By default, the standard derivative method is used, but we can change it ourselves
        >>> q.set_error_method("monte-carlo")
        >>> result.value
        18.03203135268583
        >>> result.error
        1.4116412532654283

    """

    def __init__(self, formula: Formula, error_method: settings.ErrorMethod = None):

        # set default error method for this value
        if error_method:
            self._is_error_method_specified = True
            self._error_method = error_method
        else:
            # use global default if not specified
            self._is_error_method_specified = False
            self._error_method = settings.get_error_method()

        # The _formula attribute is the core of a DerivedValue object. It is the formula by which this
        # value is derived. It is stored as a namedtuple with two fields: "operator" and "operands"
        # The "operator" is a string representing the operation performed (e.g. "mul", "add", ...),
        # and "operands" is a list of ExperimentalValue objects on which the operation was executed.
        # If the operand is another DerivedValue, its formula can also be accessed. This structure
        # practically works as an expression tree, allowing the value and uncertainty to be calculated
        # on the fly, instead of set in stone during instantiation.
        self._formula = formula

        super().__init__()
        ExperimentalValue._register[self._id] = self

        # propagate units if applicable
        self._units = op.propagate_units(formula)

    @property
    def value(self) -> float:
        """The value of this calculated quantity"""
        return self.__get_value_error_pair().value

    @value.setter
    def value(self, new_value: Real):
        if not isinstance(new_value, Real):
            raise InvalidArgumentTypeError("value", got=new_value, expected="real number")
        self._values = {lit.RECORDED: ValueWithError(new_value, self.error)}  # reset the values of this object
        warnings.warn("\nYou are trying to override the calculated value of a derived quantity. As a result, "
                      "\nthis value is casted to a regular Measurement")
        self.__class__ = MeasuredValue  # casting it to MeasuredValue

    @property
    def error(self) -> float:
        """The propagated error on the calculated quantity"""
        return self.__get_value_error_pair().error

    @error.setter
    def error(self, new_error: Real):
        if not isinstance(new_error, Real):
            raise InvalidArgumentTypeError("error", got=new_error, expected="positive real number")
        if new_error < 0:
            raise ValueError("Invalid argument for error: {}, expecting: positive real number".format(new_error))
        self._values = {lit.RECORDED: ValueWithError(self.value, new_error)}  # reset the values of this object
        warnings.warn("\nYou are trying to override the propagated error of a calculated quantity. As a result, "
                      "\nthis value is casted to a regular Measurement")
        self.__class__ = MeasuredValue  # casting it to MeasuredValue

    @property
    def relative_error(self) -> float:
        """Gets the relative error (error/mean) of a DerivedValue object."""
        return self.error / self.value if self.value != 0 else 0.

    @relative_error.setter
    def relative_error(self, relative_error: Real):
        if not isinstance(relative_error, Real):
            raise InvalidArgumentTypeError("error", got=relative_error, expected="positive real number")
        if relative_error < 0:
            raise ValueError("Invalid argument for error: {}, expecting: positive real number".format(relative_error))
        new_error = self.value * float(relative_error)
        self._values = {lit.RECORDED: ValueWithError(self.value, new_error)}  # reset the values of this object
        warnings.warn("You are trying to override the propagated error of a calculated quantity. As a result, "
                      "this value is casted to a regular Measurement")
        self.__class__ = MeasuredValue  # casting it to MeasuredValue

    def derivative(self, other: ExperimentalValue) -> float:
        """Finds the derivative with respect to another ExperimentalValue"""

        if self._id == other._id:
            return 1  # the derivative of anything with respect to itself is 1
        return op.differentiate(self._formula, other)

    def __get_value_error_pair(self) -> ValueWithError:
        """Gets the value-error pair for the current specified error method"""

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
    """A value with no uncertainty"""

    def __init__(self, value, unit="", name=""):
        if not isinstance(value, Real):
            raise ValueError("Cannot create Constant with value {} of type {}".format(value, type(value)))
        super().__init__(unit=unit, name=name)
        self._values[lit.RECORDED] = ValueWithError(value, 0)

    @property
    def value(self):
        return self._values[lit.RECORDED].value

    @property
    def error(self):
        return 0

    def derivative(self, other) -> 0:
        return 0  # the derivative of a constant with respect to anything is 0


class ExperimentalValueArray(np.ndarray):
    """An array of experimental values. alias: MeasurementArray

    A ExperimentalValueArray (MeasurementArray) represents a series of ExperimentalValue objects, It can
    be used for data analysis, fitting, and plotting. This is implemented as a subclass of numpy.ndarray.

    For backwards compatibility, this class is given the alias "MeasurementArray". Users are encouraged
    to call "MeasurementArray" instead of "ExperimentalValueArray" to record arrays of measurements.

    Args:
        data: a list of measurements
        error: the uncertainty of the measurements. It can either be an array of uncertainties,
            corresponding to each individual measurement, or a single uncertainty that applies to
            all measurements in this series.
        unit: the unit of this series of measurements
        name: the name of this series of measurements

    Examples:
        >>> import qexpy as q

        We can instantiate an array of measurements with two lists
        >>> a = q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5])
        >>> a
        ExperimentalValueArray([MeasuredValue(1.0 +/- 0.1),
                    MeasuredValue(2.0 +/- 0.2),
                    MeasuredValue(3.0 +/- 0.3),
                    MeasuredValue(4.0 +/- 0.4),
                    MeasuredValue(5.0 +/- 0.5)], dtype=object)

        We can also create an array of measurements with a single uncertainty. As usual, if the error
        is not specified, they will be set to 0 by default
        >>> a = q.MeasurementArray([1, 2, 3, 4, 5], 0.5, unit="m", name="length")
        >>> a
        ExperimentalValueArray([MeasuredValue(1.0 +/- 0.5),
                    MeasuredValue(2.0 +/- 0.5),
                    MeasuredValue(3.0 +/- 0.5),
                    MeasuredValue(4.0 +/- 0.5),
                    MeasuredValue(5.0 +/- 0.5)], dtype=object)

        We can access the different statistical properties of this array
        >>> print(np.sum(a))
        15 +/- 1 [m]
        >>> print(a.mean())
        3.0 +/- 0.7 [m]
        >>> a.std()
        1.5811388300841898

        Manipulation of a MeasurementArray is also very easy. We can append or insert into the array
        values in multiple formats
        >>> a = a.append((7, 0.2))  # a measurement can be inserted as a tuple
        >>> print(a[5])
        length = 7.0 +/- 0.2 [m]
        >>> a = a.insert(0, 8)  # if error is not specified, it is set to 0 by default
        >>> print(a[0])
        length = 8 +/- 0 [m]

        The same operations also works with array-like objects, in which case they are concatenated
        >>> a = a.append([(10, 0.1), (11, 0.3)])
        >>> a
        ExperimentalValueArray([MeasuredValue(8.0 +/- 0),
                    MeasuredValue(1.0 +/- 0.5),
                    MeasuredValue(2.0 +/- 0.5),
                    MeasuredValue(3.0 +/- 0.5),
                    MeasuredValue(4.0 +/- 0.5),
                    MeasuredValue(5.0 +/- 0.5),
                    MeasuredValue(7.0 +/- 0.2),
                    MeasuredValue(10.0 +/- 0.1),
                    MeasuredValue(11.0 +/- 0.3)], dtype=object)

        The ExperimentalValueArray object is vectorized just like numpy.ndarray. You can perform basic
        arithmetic operations as well as functions with them and get back ExperimentalValueArray objects
        >>> a = q.MeasurementArray([0, 1, 2], 0.5)
        >>> a + 2
        ExperimentalValueArray([DerivedValue(2.0 +/- 0.5),
                        DerivedValue(3.0 +/- 0.5),
                        DerivedValue(4.0 +/- 0.5)], dtype=object)
        >>> q.sin(a)
        ExperimentalValueArray([DerivedValue(0.0 +/- 0.5),
                        DerivedValue(0.8 +/- 0.3),
                        DerivedValue(0.9 +/- 0.2)], dtype=object)


    See Also:
        numpy.ndarray

    """

    def __new__(cls, data: List[Real], error: Union[List[Real], Real] = None, unit="", name=""):
        """Default constructor for a ExperimentalValueArray

        __new__ is used instead of __init__ for object initialization. This is the convention for
        subclassing numpy arrays.

        """

        # initialize raw data and its corresponding uncertainties
        if not isinstance(data, utils.ARRAY_TYPES):
            raise InvalidArgumentTypeError("creating MeasurementArray",
                                           got=[data], expected="a list or real numbers")
        measurements = np.asarray(data)

        if error is None:
            error_array = [0] * len(measurements)
        elif isinstance(error, Real):
            error_array = [float(error)] * len(measurements)
        elif isinstance(error, utils.ARRAY_TYPES) and len(error) == len(data):
            error_array = np.asarray(error)
        else:
            raise InvalidArgumentTypeError("uncertainties of a MeasurementArray",
                                           got=error, expected="real number or a list of real numbers")

        measured_values = list(MeasuredValue(val, err, unit, name) for val, err in zip(measurements, error_array))
        obj = np.asarray(measured_values, dtype=ExperimentalValue).view(ExperimentalValueArray)
        return obj

    def __str__(self):
        string = ", ".join(variable.print_value() for variable in self)
        name = self.name + " = " if self.name else ""
        unit = " (" + self.unit + ")" if self.unit else ""
        return "{}[ {} ]{}".format(name, string, unit)

    def __setitem__(self, key, value):
        value = self.__wrap_measurement(value)
        super().__setitem__(key, value)

    @property
    def name(self) -> str:
        """name of this array of values"""
        return self[0].name

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise InvalidArgumentTypeError("name", got=new_name, expected="string")
        for data in self:
            data.name = new_name

    @property
    def unit(self) -> str:
        """The unit of this array of values"""
        return self[0].unit

    @unit.setter
    def unit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise InvalidArgumentTypeError("unit", got=new_unit, expected="string")
        new_unit = units.parse_units(new_unit)
        for data in self:
            data._units = new_unit

    @property
    def values(self) -> np.ndarray:
        """An array of center values"""
        return np.asarray(list(data.value for data in self))

    @property
    def errors(self) -> np.ndarray:
        """An array of the uncertainties"""
        return np.asarray(list(data.error for data in self))

    def append(self, value: Union[Real, List, Tuple, ExperimentalValue]) -> "ExperimentalValueArray":
        """adds a value to the end of this array and returns the new array"""
        if isinstance(value, ExperimentalValueArray):
            pass  # don't do anything if the new value is already a ExperimentalValueArray
        elif isinstance(value, utils.ARRAY_TYPES):
            value = list(self.__wrap_measurement(value) for value in value)
        else:
            value = self.__wrap_measurement(value)
        # append new value and cast to ExperimentalValueArray
        return np.append(self, value).view(ExperimentalValueArray)

    def insert(self, position: int, value: Union[Real, List, Tuple, ExperimentalValue]) -> "ExperimentalValueArray":
        """adds a value to a position in this array and returns the new array"""
        if isinstance(value, ExperimentalValueArray):
            pass  # don't do anything if the new value is already a ExperimentalValueArray
        elif isinstance(value, utils.ARRAY_TYPES):
            value = list(self.__wrap_measurement(value) for value in value)
        else:
            value = self.__wrap_measurement(value)
        return np.insert(self, position, value).view(ExperimentalValueArray)

    def delete(self, position: int) -> "ExperimentalValueArray":
        """deletes the value on the requested position and returns the new array"""
        return np.delete(self, position).view(ExperimentalValueArray)

    def mean(self, **kwargs) -> MeasuredValue:  # pylint: disable=unused-argument
        """The mean of the array"""
        result = np.mean(self.values)
        error = self.error_on_mean()
        name = "Mean of {}".format(self.name) if self.name else ""
        return MeasuredValue(float(result), error, self.unit, name)

    def std(self, ddof=1, **kwargs) -> float:  # pylint: disable=unused-argument
        """The standard deviation of this array"""
        return float(np.std(self.values, ddof=ddof))

    def sum(self, **kwargs) -> MeasuredValue:  # pylint: disable=unused-argument
        """The sum of the array"""
        result = np.sum(self.values)
        error = np.sqrt(np.sum(self.errors ** 2))
        return MeasuredValue(float(result), float(error), self.unit, self.name)

    def error_on_mean(self) -> float:
        """The error on the mean of this array"""
        return self.std() / m.sqrt(self.size)

    def error_weighted_mean(self) -> float:
        """The error weighted mean of this array"""
        if any(err == 0 for err in self.errors):
            warnings.warn("One or more of the errors are 0, the error weighted mean cannot be calculated.")
            return 0
        weights = np.asarray(list(1 / (err ** 2) for err in self.errors))
        return float(np.sum(weights * self.values) / np.sum(weights))

    def propagated_error(self) -> float:
        """The propagated error from the error weighted mean calculation"""
        if any(err == 0 for err in self.errors):
            warnings.warn("One or more of the errors are 0, the propagated error cannot be calculated.")
            return 0
        weights = np.asarray(list(1 / (err ** 2) for err in self.errors))
        return 1 / np.sqrt(np.sum(weights))

    def __wrap_measurement(self, value: Union[Real, Tuple, ExperimentalValue]) -> ExperimentalValue:
        """wraps a value in a Measurement object"""
        if isinstance(value, Real):
            value = MeasuredValue(float(value), 0, unit=self.unit, name=self.name)
        elif isinstance(value, tuple) and len(value) == 2:
            value = MeasuredValue(float(value[0]), float(value[1]), unit=self.unit, name=self.name)
        elif not isinstance(value, ExperimentalValue):
            # TODO: also check for units and names to see if they are compatible
            raise ValueError("Elements of MeasurementArray must be or are convertible to qexpy defined values")
        return value


def get_variable_by_id(variable_id: uuid.UUID) -> ExperimentalValue:
    """Internal method used to retrieve an ExperimentalValue instance with its ID"""
    return ExperimentalValue._register[variable_id]


def get_covariance(var1: ExperimentalValue, var2: ExperimentalValue) -> float:
    """Finds the covariances between two ExperimentalValues

    As of now, the covariance between DerivedValue objects is always 0, covariance propagation
    is still under development.

    TODO:
        implement covariance propagation for DerivedValues

    """
    if any(not isinstance(var, ExperimentalValue) for var in [var1, var2]):
        raise UndefinedOperationError("get_covariance()", [var1, var2], "ExperimentalValue instances")
    if var1.error == 0 or var2.error == 0:
        return 0  # constants don't correlate with anyone
    if isinstance(var1, MeasuredValue) and isinstance(var2, MeasuredValue):
        return __get_covariance(var1, var2)
    return 0


def set_covariance(var1: MeasuredValue, var2: MeasuredValue, cov: Real = None):
    """Sets the covariance between two measurements

    The covariance between two variables is by default 0. The user can set the covariance between two
    Measurements to any value, and it will be taken into account during error propagation. When two
    Measurement objects are recorded as arrays of repeated measurements of the same length, the user
    can leave the covariance term blank, and have QExPy calculate the covariance between them.

    Examples:
        >>> import qexpy as q
        >>> a = q.Measurement(5, 0.5)
        >>> b = q.Measurement(6, 0.3)

        The user can manually set the covariance between two values
        >>> q.set_covariance(a, b, 0.135)
        >>> q.get_covariance(a, b)
        0.135

        The correlation factor is calculated behind the scene as well
        >>> q.get_correlation(a, b)
        0.9

        The user can ask QExPy to calculate the covariance if applicable
        >>> a = q.Measurement([1, 1.2, 1.3, 1.4])
        >>> b = q.Measurement([2, 2.1, 3, 2.3])
        >>> q.set_covariance(a, b)  # this will declare that a and b are indeed correlated
        >>> q.get_covariance(a, b)
        0.0416667

    See Also:
        :py:method:`.set_correlation`

    """

    if not isinstance(var1, MeasuredValue) or not isinstance(var2, MeasuredValue):
        raise UndefinedOperationError("set_covariance()", [var1, var2], "Measurement objects")
    if var1.error == 0 or var2.error == 0:
        raise ArithmeticError("Constants or values with no standard deviation don't correlate with other values")

    if isinstance(var1, RepeatedlyMeasuredValue) and isinstance(var2, RepeatedlyMeasuredValue) and cov is None:
        covariance, correlation = __calculate_covariance_and_correlation(var1, var2)
    elif cov is None:
        raise ValueError("The covariance value is not provided and it cannot be calculated for the inputs given")
    else:
        covariance = cov
        var1_std = var1.std if isinstance(var1, RepeatedlyMeasuredValue) else var1.error
        var2_std = var2.std if isinstance(var2, RepeatedlyMeasuredValue) else var2.error
        correlation = cov / (var1_std * var2_std)

    # check that the result makes sense
    if correlation > 1 or correlation < -1:
        raise ValueError("The covariance provided: {} is non-physical".format(cov))

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


def set_correlation(var1: MeasuredValue, var2: MeasuredValue, cor: Real = None):
    """Sets the correlation factor between two MeasuredValue objects

    The correlation factor is a value between -1 and 1. This method can be used the same way
    as set_covariance.

    See Also:
        :py:method:`.set_covariance`

    """

    if not isinstance(var1, MeasuredValue) or not isinstance(var2, MeasuredValue):
        raise UndefinedOperationError("set_correlation()", [var1, var2], "Measurement objects")
    if cor and (float(cor) > 1 or float(cor) < -1):
        raise ValueError("The correlation factor provided: {} is non-physical.".format(cor))
    if var1.error == 0 or var2.error == 0:
        raise ArithmeticError("Constants or values with no standard deviation don't correlate with other values")

    if isinstance(var1, RepeatedlyMeasuredValue) and isinstance(var2, RepeatedlyMeasuredValue) and cor is None:
        covariance, correlation = __calculate_covariance_and_correlation(var1, var2)
    elif cor is None:
        raise ValueError("The covariance value is not provided")
    else:
        correlation = cor
        var1_std = var1.std if isinstance(var1, RepeatedlyMeasuredValue) else var1.error
        var2_std = var2.std if isinstance(var2, RepeatedlyMeasuredValue) else var2.error
        covariance = cor * (var1_std * var2_std)

    # set relations in the module level register
    id_string = "_".join(sorted(map(str, [var1._id, var2._id])))
    ExperimentalValue._correlations[id_string] = Correlation((var1, var2), correlation, covariance)


def wrap_operand(operand: Union[Real, ExperimentalValue]) -> ExperimentalValue:
    """Wraps the operand of an operation as an ExperimentalValue instance

    If the operand is a number, construct a Constant object with this value. If the operand
    is an ExperimentalValue object, return the object directly.

    """
    if isinstance(operand, Real):
        return Constant(operand)
    if isinstance(operand, ExperimentalValue):
        return operand
    if isinstance(operand, tuple) and len(operand) == 2:
        return MeasuredValue(operand[0], operand[1])
    raise TypeError


def __get_covariance(var1: MeasuredValue, var2: MeasuredValue) -> float:
    """Gets the covariance factor between two values"""
    if var1._id == var2._id:
        # the covariance between a value and itself is the variance
        return var1.std ** 2 if isinstance(var1, RepeatedlyMeasuredValue) else var1.error ** 2
    id_string = "_".join(sorted([str(var1._id), str(var2._id)]))
    if id_string in ExperimentalValue._correlations:
        return ExperimentalValue._correlations[id_string].covariance
    return 0


def __get_correlation(var1: MeasuredValue, var2: MeasuredValue) -> float:
    """Gets the correlation factor between two values"""
    if var1._id == var2._id:
        return 1  # values have unit correlation with themselves
    id_string = "_".join(sorted([str(var1._id), str(var2._id)]))
    if id_string in ExperimentalValue._correlations:
        return ExperimentalValue._correlations[id_string].correlation
    return 0


def __calculate_covariance_and_correlation(var1: RepeatedlyMeasuredValue, var2: RepeatedlyMeasuredValue) -> tuple:
    if len(var1.raw_data) != len(var2.raw_data):
        raise ValueError("Cannot calculate covariance between arrays of length {} and {}."
                         "".format(len(var1.raw_data), len(var2.raw_data)))
    covariance = utils.calculate_covariance(var1.raw_data, var2.raw_data)
    correlation = covariance / (var1.std * var2.std)
    return covariance, correlation
