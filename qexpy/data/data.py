"""Module containing the core data structures for experimental values

This module defines ExperimentalValue and all its sub-classes. They serve as a container for
quantities recorded in an experiment, or calculated in subsequent data analysis, with error
propagation and other features (such as unit propagation) built-in.

"""

import uuid
import warnings
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Union
from numbers import Real
from collections import namedtuple

from qexpy.utils import IllegalArgumentError, UndefinedActionError
from qexpy.settings import ErrorMethod

import qexpy.utils as utils
import qexpy.settings as sts
import qexpy.settings.literals as lit

from . import operations as op
from . import utils as dut

ARRAY_TYPES = list, np.ndarray

# A simple data structure to store a value-uncertainty pair
ValueWithError = namedtuple("ValueWithError", "value, error")

# A sub-tree in an expression tree representing a formula. The "operator" is the root node of
# the sub-tree, and the "operands" is a list of branches. The leaf nodes of a complete tree
# are individual ExperimentalValue instances
Formula = namedtuple("Formula", "operator, operands")

# A data structure to store the correlation between two values.
Correlation = namedtuple("Correlation", "correlation, covariance")


class ExperimentalValue(ABC):
    """Base class for quantities with a value and an uncertainty

    The ExperimentalValue is a container for an individual quantity involved in an experiment
    and subsequent data analysis. Each quantity has a value and an uncertainty (error), and
    optionally, a name and a unit. ExperimentalValue instances can be used in calculations
    just like any other numerical variable in Python. The result of such calculations will be
    wrapped in ExperimentalValue instances, with the properly propagated uncertainties.

    Examples:
        >>> import qexpy as q

        >>> a = q.Measurement(302, 5) # The standard way to initialize an ExperimentalValue

        >>> # Access the basic properties
        >>> a.value
        302
        >>> a.error
        5
        >>> a.relative_error  # This is defined as error/value
        0.016556291390728478

        >>> # These properties can be changed
        >>> a.value = 303
        >>> a.value
        303
        >>> a.relative_error = 0.05
        >>> a.error  # The error and relative_error are connected
        15.15

        >>> # You can specify the name or the units of a value
        >>> a.name = "force"
        >>> a.unit = "kg*m^2/s^2"

        >>> # The string representation of the value will include the name and units
        >>> print(a)
        force = 300 +/- 20 [kg⋅m^2⋅s^-2]

        >>> # You can also specify how you want the values or the units to be printed
        >>> q.set_print_style(q.PrintStyle.SCIENTIFIC)
        >>> q.set_unit_style(q.UnitStyle.FRACTION)
        >>> q.set_sig_figs_for_error(2)
        >>> print(a)
        force = (3.03 +/- 0.15) * 10^2 [kg⋅m^2/s^2]

    """

    # Static register that stores references to all instantiated values in a session.
    _register = {}  # type: Dict[uuid.UUID, "ExperimentalValue"]

    # Static database that stores all correlations between measurements. The key of this
    # database is the UUIDs of the two measurements concatenated in natual order.
    _correlations = {}  # type: Dict[str, Correlation]

    def __init__(self, unit: str = "", name: str = "", save=True):
        """Constructor for ExperimentalValue"""

        # Stores each unit string and their powers.
        if unit is not None and not isinstance(unit, str):
            raise TypeError("The unit provided is not a string!")
        self._unit = utils.parse_unit_string(unit) if unit else {}  # type: Dict[str, int]

        # The name of this quantity if given
        if name is not None and not isinstance(name, str):
            raise TypeError("The name provided is not a string!")
        self._name = name  # type: str

        # Each instance is given a unique ID for easy reference
        self._id = uuid.uuid4()  # type: uuid.UUID

        if save:  # save this value in the register
            self._register[self._id] = self

    def __str__(self):
        name_string = "{} = ".format(self.name) if self.name else ""
        unit_string = " [{}]".format(self.unit) if self.unit else ""
        return "{}{}{}".format(name_string, self.print_value_error(), unit_string)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.print_value_error())

    @property
    @abstractmethod
    def value(self):
        """float: The center value of this quantity"""
        raise NotImplementedError

    @property
    @abstractmethod
    def error(self):
        """float: The uncertainty of this quantity"""
        raise NotImplementedError

    @property
    @abstractmethod
    def relative_error(self):
        """float: The ratio of the uncertainty to its center value"""
        raise NotImplementedError

    @property
    def std(self):
        """float: The standard deviation of this quantity"""
        return self.error  # usually the standard deviation is the error

    @property
    def name(self):
        """str: The name of this quantity"""
        return self._name

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise TypeError(
                "Cannot set name of value to \"{}\"".format(type(new_name).__name__))
        self._name = new_name

    @property
    def unit(self):
        """str: The unit of this quantity"""
        return utils.construct_unit_string(self._unit) if self._unit else ""

    @unit.setter
    def unit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise TypeError(
                "Cannot set unit of value to \"{}\"".format(type(new_unit).__name__))
        self._unit = utils.parse_unit_string(new_unit) if new_unit else {}

    @utils.check_operand_type("==")
    def __eq__(self, other):
        return self.value == dut.wrap_in_experimental_value(other).value

    def __neg__(self):
        return DerivedValue(Formula(lit.NEG, [self]))

    @utils.check_operand_type(">")
    def __gt__(self, other):
        return self.value > dut.wrap_in_experimental_value(other).value

    @utils.check_operand_type(">=")
    def __ge__(self, other):
        return self.value >= dut.wrap_in_experimental_value(other).value

    @utils.check_operand_type("<")
    def __lt__(self, other):
        return self.value < dut.wrap_in_experimental_value(other).value

    @utils.check_operand_type("<=")
    def __le__(self, other):
        return self.value <= dut.wrap_in_experimental_value(other).value

    @utils.check_operand_type("pow")
    def __pow__(self, power):
        if isinstance(power, ARRAY_TYPES):
            return power.__rpow__(self)
        return DerivedValue(Formula(lit.POW, [self, dut.wrap_in_experimental_value(power)]))

    @utils.check_operand_type("pow")
    def __rpow__(self, other):
        return DerivedValue(Formula(lit.POW, [
            dut.wrap_in_experimental_value(other), self]))

    @utils.check_operand_type("+")
    def __add__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return other.__radd__(self)
        return DerivedValue(Formula(lit.ADD, [self, dut.wrap_in_experimental_value(other)]))

    @utils.check_operand_type("+")
    def __radd__(self, other):
        return DerivedValue(Formula(lit.ADD, [
            dut.wrap_in_experimental_value(other), self]))

    @utils.check_operand_type("-")
    def __sub__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return other.__rsub__(self)
        return DerivedValue(Formula(lit.SUB, [self, dut.wrap_in_experimental_value(other)]))

    @utils.check_operand_type("-")
    def __rsub__(self, other):
        return DerivedValue(Formula(lit.SUB, [
            dut.wrap_in_experimental_value(other), self]))

    @utils.check_operand_type("*")
    def __mul__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return other.__rmul__(self)
        return DerivedValue(Formula(lit.MUL, [self, dut.wrap_in_experimental_value(other)]))

    @utils.check_operand_type("*")
    def __rmul__(self, other):
        return DerivedValue(Formula(lit.MUL, [
            dut.wrap_in_experimental_value(other), self]))

    @utils.check_operand_type("/")
    def __truediv__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return other.__rtruediv__(self)
        return DerivedValue(Formula(lit.DIV, [self, dut.wrap_in_experimental_value(other)]))

    @utils.check_operand_type("/")
    def __rtruediv__(self, other):
        return DerivedValue(Formula(lit.DIV, [
            dut.wrap_in_experimental_value(other), self]))

    @abstractmethod
    def derivative(self, other: "ExperimentalValue") -> float:
        """Calculates the derivative of this quantity with respect to another

        The derivative of any value with respect to itself is 1, and for unrelated values,
        the derivative is always 0. This method is typically called from a DerivedValue,
        to find out its derivative with respect to one of the measurements it's derived from.

        Args:
            other (ExperimentalValue): the target for finding the derivative

        """
        raise NotImplementedError

    # pylint: disable=no-self-use,unused-argument
    def get_covariance(self, other: "ExperimentalValue") -> float:
        """Gets the covariance between this value and another value"""
        return 0  # default covariance is 0

    def set_covariance(self, other: "ExperimentalValue", cov: float = None):
        """Sets the covariance between this value and another value

        The covariance between two variables is by default 0. Users can set the covariance
        between two measurements to any value, and it will be taken into account during error
        propagation. When two measurements are recorded as arrays of repeated measurements of
        the same length, users can leave the covariance term empty, and let QExPy calculate
        the covariance between them. You should only do this when these two quantities are
        measured at the same time, and can be related physically.

        Examples:
            >>> import qexpy as q
            >>> a = q.Measurement(5, 0.5)
            >>> b = q.Measurement(6, 0.3)

            >>> # The user can manually set the covariance between two values
            >>> a.set_covariance(b, 0.135)
            >>> a.get_covariance(b)
            0.135

            >>> # The correlation factor is calculated behind the scene as well
            >>> a.get_correlation(b)
            0.9

            >>> # The user can ask QExPy to calculate the covariance if applicable
            >>> a = q.Measurement([1, 1.2, 1.3, 1.4])
            >>> b = q.Measurement([2, 2.1, 3, 2.3])
            >>> a.set_covariance(b)  # this will declare that a and b are indeed correlated
            >>> a.get_covariance(b)
            0.0416667

        """
        raise UndefinedActionError("Cannot set covariance between non-measurements.")

    # pylint: disable=no-self-use,unused-argument
    def get_correlation(self, other: "ExperimentalValue") -> float:
        """Gets the correlation between this value and another value"""
        return 0  # default correlation is 0

    def set_correlation(self, other: "ExperimentalValue", corr: float = None):
        """Sets the correlation between this value and another value

        The correlation factor is a value between -1 and 1. This method can be used the same
        way as set_covariance.

        See Also:
            :py:func:`ExperimentalValue.set_covariance`

        """
        raise UndefinedActionError("Cannot set correlation between non-measurements.")

    def print_value_error(self) -> str:
        """Helper method that prints the value-error pair in proper format"""
        return utils.get_printer()(self.value, self.error)

    @staticmethod
    def get(variable_id: uuid.UUID) -> "ExperimentalValue":
        """Retrieves a value from the register using its UUID"""
        return ExperimentalValue._register[
            variable_id] if variable_id in ExperimentalValue._register else None


class Constant(ExperimentalValue):
    """A value with no uncertainty"""

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs, save=False)
        self._value_error = ValueWithError(value, 0)

    @property
    def value(self) -> float:
        return self._value_error.value

    @property
    def error(self) -> 0:
        return 0  # pragma: no cover

    @property
    def relative_error(self) -> 0:
        return 0  # pragma: no cover

    def derivative(self, other: "ExperimentalValue") -> 0:
        return 0  # the derivative of a constant with respect to anything is 0


class MeasuredValue(ExperimentalValue):
    """Container for user-recorded values with uncertainties

    The MeasuredValue represents a single measurement recorded in an experiment. This class
    is given an alias "Measurement" for backward compatibility and for a more intuitive user
    interface. On the top level of this package, this class is imported as "Measurement".

    Args:
        data (Real|List): The center value of the measurement
        error (Real|List): The uncertainty on the value

    Keyword Args:
        unit (str): The unit of this value
        name (str): The name of this value

    """

    def __new__(cls, data, error=None, **kwargs):  # pylint: disable=unused-argument
        if isinstance(data, Real):
            instance = super().__new__(cls)
        elif isinstance(data, ARRAY_TYPES):
            instance = super().__new__(RepeatedlyMeasuredValue)
        else:
            raise IllegalArgumentError("Invalid data type to record a measurement!")
        return instance

    def __init__(self, data, error=None, **kwargs):
        if error is not None and not isinstance(error, Real):
            raise IllegalArgumentError("Invalid data type to record an uncertainty!")
        unit = kwargs.get("unit", "")
        name = kwargs.get("name", "")
        save = kwargs.get("save", True)
        super().__init__(unit, name, save=save)
        self._value, self._error = float(data), float(error) if error else 0.0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Real):
        if not isinstance(value, Real):
            raise TypeError("Cannot assign a {} to the value!".format(type(value).__name__))
        self._value = value

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error: Real):
        if not isinstance(error, Real):
            raise TypeError("Cannot assign a {} to the error!".format(type(error).__name__))
        if error < 0:
            raise ValueError("The error must be a positive real number!")
        self._error = error

    @property
    def relative_error(self):
        return self.error / self.value if self.value != 0 else 0.

    @relative_error.setter
    def relative_error(self, relative_error: Real):
        if not isinstance(relative_error, Real):
            raise TypeError(
                "Cannot assign a {} to the error!".format(type(relative_error).__name__))
        if relative_error < 0:
            raise ValueError("The error must be a positive real number!")
        new_error = self.value * float(relative_error)
        self._error = new_error

    def derivative(self, other: "ExperimentalValue") -> float:
        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError(
                "You can only find derivative with respect to another ExperimentalValue")
        # Derivative of a measurement with respect to anything other than itself is 0
        return 1 if self._id == other._id else 0

    def get_covariance(self, other: "ExperimentalValue") -> float:
        """Gets the covariance of this value with another value"""

        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError("Cannot find covariance for non-QExPy defined values")
        if not isinstance(other, MeasuredValue):
            return 0  # only covariance between measurements is supported.

        if self.std == 0 or other.std == 0:
            return 0  # constants don't correlate with anyone
        if self._id == other._id:
            # The covariance between a measurement and itself is the variance
            return self.std ** 2

        id_string = "_".join(sorted([str(self._id), str(other._id)]))
        if id_string in ExperimentalValue._correlations:
            return ExperimentalValue._correlations[id_string].covariance

        return 0

    def set_covariance(self, other: "ExperimentalValue", cov: float = None):
        """Sets the covariance of this value with another value"""

        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError("Cannot set covariance for non-QExPy defined values")
        if not isinstance(other, MeasuredValue):
            raise IllegalArgumentError("Only covariance between measurements is supported.")

        if self.std == 0 or other.std == 0:
            raise ArithmeticError("Cannot set covariance for values with 0 errors")
        if cov is None:
            raise IllegalArgumentError(
                "The covariance is not provided, and cannot be calculated!")

        corr = cov / (self.std * other.std)
        # check that the result makes sense
        if corr > 1 or corr < -1:
            raise ValueError("The covariance: {} is non-physical".format(cov))

        # register the correlation between these measurements
        id_string = "_".join(sorted([str(self._id), str(other._id)]))
        correlation_record = Correlation(corr, cov)
        ExperimentalValue._correlations[id_string] = correlation_record

    def get_correlation(self, other: "ExperimentalValue") -> float:
        """Gets the correlation factor of this value with another value"""

        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError("Can't find correlation for non-QExPy defined values")
        if not isinstance(other, MeasuredValue):
            return 0  # only covariance between measurements is supported.

        if self.std == 0 or other.std == 0:
            return 0  # constants don't correlate with anyone
        if self._id == other._id:
            return 1  # values have unit correlation with themselves

        id_string = "_".join(sorted([str(self._id), str(other._id)]))
        if id_string in ExperimentalValue._correlations:
            return ExperimentalValue._correlations[id_string].correlation
        return 0

    def set_correlation(self, other: "ExperimentalValue", corr: float = None):
        """Sets the correlation factor of this value with another value"""

        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError("Cannot set correlation for non-QExPy defined values")
        if not isinstance(other, MeasuredValue):
            raise IllegalArgumentError("Only covariance between measurements is supported.")

        if self.std == 0 or other.std == 0:
            raise ArithmeticError("Cannot set correlation for values with 0 errors")
        if corr is None:
            raise IllegalArgumentError(
                "The correlation factor is not provided, and cannot be calculated!")

        # check that the result makes sense
        if corr > 1 or corr < -1:
            raise ValueError("The correlation factor: {} is non-physical".format(corr))
        cov = corr * (self.std * other.std)

        # register the correlation between these measurements
        id_string = "_".join(sorted([str(self._id), str(other._id)]))
        correlation_record = Correlation(corr, cov)
        ExperimentalValue._correlations[id_string] = correlation_record


class RepeatedlyMeasuredValue(MeasuredValue):
    """Container for a MeasuredValue recorded as an array of repeated measurements

    This class is instantiated if an array of values is used to record a Measurement of a
    single quantity with repeated takes. By default, the mean of the array is used as the
    value of this quantity, and the standard error (error on the mean) is the uncertainty.
    The reason for this choice is because the reason for taking multiple measurements is
    usually to minimize the uncertainty on the quantity, not to find out the uncertainty on
    a single measurement (which is what standard deviation is).

    Examples:
        >>> import qexpy as q

        >>> # The most common way of recording a value with repeated measurements is to only
        >>> # give the center values for the measurements
        >>> a = q.Measurement([9, 10, 11])
        >>> print(a)
        10.0 +/- 0.6

        >>> # There are other statistical properties of the array of measurements
        >>> a.std
        1
        >>> a.error_on_mean
        0.5773502691896258

        >>> # You can choose to use the standard deviation as the uncertainty
        >>> a.use_std_for_uncertainty()
        >>> a.error
        1

        >>> # You can also specify individual uncertainties for the measurements
        >>> a = q.Measurement([10, 11], [0.1, 1])
        >>> print(a)
        10.5 +/- 0.5
        >>> a.error_weighted_mean
        10.00990099009901
        >>> a.propagated_error
        0.09950371902099892

        >>> # You can choose which statistical properties to be used as the value/error
        >>> a.use_error_weighted_mean_as_value()
        >>> a.use_propagated_error_for_uncertainty()
        >>> q.set_sig_figs_for_error(4)
        >>> print(a)
        10.00990 +/- 0.09950

        """

    def __init__(self, data: List, error: Union[List, Real] = None, **kwargs):
        """Constructor of a RepeatedlyMeasuredValue"""

        # Check validity of inputs
        if isinstance(error, ARRAY_TYPES) and len(error) != len(data):
            raise ValueError("The lengths of uncertainties and data do not match")

        # pylint: disable=cyclic-import
        from .datasets import ExperimentalValueArray

        # Initialize raw data and its uncertainties. Internally, the raw data is implemented
        # as an ExperimentalValueArray. However, in principle, the ExperimentalValueArray
        # should only be used for an array of measurements of different quantities.
        self._raw_data = ExperimentalValueArray(data, error, save=False, **kwargs)

        # Calculate its statistical properties
        self._mean = self._raw_data.mean().value
        self._std = self._raw_data.std(ddof=1)
        self._error_on_mean = self._raw_data.error_on_mean()

        # Call parent constructor with mean and error on mean as value and uncertainty
        super().__init__(self._mean, self._error_on_mean, **kwargs)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: Real):
        if not isinstance(new_value, Real):
            raise TypeError(
                "Cannot assign a {} to the value!".format(type(new_value).__name__))
        warnings.warn(
            "You are trying to override the value calculated from an array of repeated "
            "measurements. This value is now considered a single Measurement.")
        self.__class__ = MeasuredValue
        self._value = new_value

    @property
    def raw_data(self):
        """np.ndarray: The raw data that was used to generate this measurement"""
        return self._raw_data.values if all(
            x.error == 0 for x in self._raw_data) else self._raw_data

    @property
    def std(self):
        """float: The standard deviation of the raw data"""
        return self._std

    @property
    def error_on_mean(self):
        """float: The error on the mean or the standard error"""
        return self._error_on_mean

    @property
    def mean(self):
        """float: The mean of raw measurements"""
        return self._mean

    @property
    def error_weighted_mean(self):
        """float: Error weighted mean if individual errors are specified"""
        return self._raw_data.error_weighted_mean()

    @property
    def propagated_error(self):
        """float: Error propagated with errors passed in if present"""
        return self._raw_data.propagated_error()

    def use_std_for_uncertainty(self):
        """Sets the uncertainty of this value to the standard deviation"""
        self._error = self._std

    def use_error_on_mean_for_uncertainty(self):
        """Sets the uncertainty of this value to the error on the mean"""
        self._error = self._error_on_mean

    def use_error_weighted_mean_as_value(self):
        """Sets the value of this object to the error weighted mean"""
        error_weighted_mean = self.error_weighted_mean
        if not np.isnan(error_weighted_mean):
            self._value = error_weighted_mean
        else:  # pragma: no cover
            warnings.warn("The error weighted mean is not valid")

    def use_propagated_error_for_uncertainty(self):
        """Sets the uncertainty of this object to the weight propagated error"""
        propagated_error = self.propagated_error
        if not np.isnan(propagated_error):
            self._error = propagated_error
        else:  # pragma: no cover
            warnings.warn("The propagated error is not valid")

    def set_covariance(self, other: "ExperimentalValue", cov: float = None):
        """Sets the covariance of this value with another value"""

        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError("Cannot set covariance for non-QExPy defined values")
        if not isinstance(other, MeasuredValue):
            raise IllegalArgumentError("Only covariance between measurements is supported.")

        if cov is None and isinstance(other, RepeatedlyMeasuredValue):
            try:
                cov = utils.calculate_covariance(self.raw_data, other.raw_data)
            except ValueError:
                cov = None

        super().set_covariance(other, cov)

    def set_correlation(self, other: "ExperimentalValue", corr: float = None):
        """Sets the correlation factor of this value with another value"""

        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError("Cannot set correlation for non-QExPy defined values")
        if not isinstance(other, MeasuredValue):
            raise IllegalArgumentError("Only covariance between measurements is supported.")

        if corr is None and isinstance(other, RepeatedlyMeasuredValue):
            try:
                cov = utils.calculate_covariance(self.raw_data, other.raw_data)
                corr = cov / (self.std * other.std)
            except ValueError:
                corr = None

        super().set_correlation(other, corr)

    def show_histogram(self, **kwargs) -> tuple:  # pragma: no cover
        """Plots the raw measurement data in a histogram

        See Also:
            This works the same as the :py:func:`~qexpy.fitting.fitting.hist` function in
            the plotting module of QExPy

        """
        import qexpy.plotting as plotting  # pylint:disable=cyclic-import
        values, bins, figure = plotting.hist(self.raw_data, **kwargs)
        figure.show()
        return values, bins, figure


class DerivedValue(ExperimentalValue):
    """Result of calculations performed with ExperimentalValue instances

    This class is automatically instantiated when the user performs calculations with other
    ExperimentalValue instances. It is created with the properly propagated uncertainties and
    units. The two available methods for error propagation are the derivative method, and the
    Monte Carlo method.

    Internally, a DerivedValue preserves information on how it is calculated, so the user is
    able to make use of that information. For example, the user can find the derivative of
    a DerivedValue with respect to another ExperimentalValue that this value is derived from.

    Examples:
        >>> import qexpy as q

        >>> # First let's create some standard measurements
        >>> a = q.Measurement(5, 0.2)
        >>> b = q.Measurement(4, 0.1)
        >>> c = q.Measurement(6.3, 0.5)
        >>> d = q.Measurement(7.2, 0.5)

        >>> # Now we can perform operations on them
        >>> result = q.sqrt(c) * d - b / q.exp(a)
        >>> result
        DerivedValue(18 +/- 1)
        >>> result.value
        18.04490478513969
        >>> result.error
        1.4454463754287323

        >>> # By default, the standard derivative method is used, but it can be changed
        >>> q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        >>> result.value
        18.03203135268583
        >>> result.error
        1.4116412532654283
        >>> # If we want this value to use a different error method from the global default
        >>> result.error_method = "derivative" # this only affects this value alone
        >>> result.error
        1.4454463754287323
        >>> # If we want to reset the error method for this value and use the global default
        >>> result.reset_error_method()
        >>> result.error
        1.4116412532654283

    """

    def __init__(self, formula: Formula):
        """Constructor for a DerivedValue"""

        # The error method used for error propagation of this value
        self.__error_method = ErrorMethod.AUTO  # type: ErrorMethod

        # The expression tree representing how this value is derived.
        self._formula = formula  # type: Formula

        # The objects used to evaluate the formula with the appropriate error methods
        self.__evaluators = {
            lit.DERIVATIVE: op.DerivativeEvaluator(),
            lit.MONTE_CARLO: op.MonteCarloEvaluator()
        }  # type: Dict[str, op.Evaluator]

        super().__init__(save=True)

        self._unit = op.propagate_units(formula)

    @property
    def value(self):
        return self.__get_value_error_pair().value

    @value.setter
    def value(self, new_value: Real):
        if not isinstance(new_value, Real):
            raise TypeError(
                "Cannot assign a {} to the value".format(type(new_value).__name__))
        warnings.warn(
            "You are trying to override the calculated value of a derived quantity. This "
            "value is casted to a regular Measurement")
        error = self.error
        self.__class__ = MeasuredValue  # casting it to MeasuredValue
        self.value, self.error = new_value, error

    @property
    def error(self):
        return self.__get_value_error_pair().error

    @error.setter
    def error(self, new_error: Real):
        if not isinstance(new_error, Real):
            raise TypeError(
                "Cannot assign a {} to the error!".format(type(new_error).__name__))
        if new_error < 0:
            raise ValueError("The error must be a positive real number!")
        warnings.warn(
            "You are trying to override the propagated error of a derived quantity. This "
            "value is casted to a regular Measurement")
        value = self.value
        self.__class__ = MeasuredValue  # casting it to MeasuredValue
        self.value, self.error = value, new_error

    @property
    def relative_error(self):
        return self.error / self.value if self.value != 0 else 0.

    @relative_error.setter
    def relative_error(self, relative_error: Real):
        if not isinstance(relative_error, Real):
            raise TypeError(
                "Cannot assign a {} to the error!".format(type(relative_error).__name__))
        if relative_error < 0:
            raise ValueError("The error must be a positive real number!")
        new_error = self.value * float(relative_error)
        warnings.warn(
            "You are trying to override the propagated relative error of a derived quantity."
            " This value is casted to a regular Measurement")
        value = self.value
        self.__class__ = MeasuredValue  # casting it to MeasuredValue
        self.value, self.error = value, new_error

    @property
    def error_method(self):
        """ErrorMethod: The default error method used for this value

        QExPy currently supports two different methods of error propagation, the derivative
        method, and the Monte-Carlo method. The user can change the global default which
        applies to all values, or set the error method of this single quantity if it is to
        be different from the global settings.

        """
        if self.__error_method == ErrorMethod.AUTO:
            return sts.get_settings().error_method
        return self.__error_method

    @error_method.setter
    def error_method(self, new_error_method: Union[ErrorMethod, str]):
        if isinstance(new_error_method, ErrorMethod):
            self.__error_method = new_error_method
        elif new_error_method in [lit.MONTE_CARLO, lit.DERIVATIVE]:
            self.__error_method = ErrorMethod(new_error_method)
        else:
            raise ValueError("Invalid error method!")

    @property
    def mc(self):
        """dut.MonteCarloSettings: The settings object for customizing Monte Carlo"""
        evaluator = self.__evaluators[lit.MONTE_CARLO]
        assert isinstance(evaluator, op.MonteCarloEvaluator)
        evaluator.regenerate_samples(self._formula)
        return evaluator.settings

    def reset_error_method(self):
        """Resets the default error method for this value to follow the global settings"""
        self.__error_method = ErrorMethod.AUTO

    def recalculate(self):
        """Recalculates the value

        A DerivedValue instance preserves information on how the value was derived. If values
        of the original measurements are changed, and you wish to update the derived value
        using the exact same formula, this method can be used.

        Examples:

            >>> import qexpy as q

            >>> a = q.Measurement(5, 0.2)
            >>> b = q.Measurement(4, 0.1)

            >>> c = a + b
            >>> c
            DerivedValue(9.0 +/- 0.2)

            >>> # Now we change the value of a
            >>> a.value = 8
            >>> c.recalculate()
            >>> c
            DerivedValue(12.0 +/- 0.2)

        """
        for evaluator in self.__evaluators.values():
            evaluator.clear()

    def derivative(self, other: ExperimentalValue) -> float:
        if not isinstance(other, ExperimentalValue):
            raise IllegalArgumentError(
                "You can only find derivative with respect to another ExperimentalValue")
        return 1 if self._id == other._id else op.differentiate(self._formula, other)

    def show_error_contributions(self):  # pragma: no cover
        """Displays measurements' contribution to the final uncertainty"""
        import matplotlib.pyplot as plt
        evaluator = self.__evaluators[lit.DERIVATIVE]
        assert isinstance(evaluator, op.DerivativeEvaluator)
        evaluator.evaluate(self._formula)
        measurements, contributions = evaluator.measurements, evaluator.error_contributions
        names = list(var.name if var.name else "var_{}".format(idx)
                     for idx, var in enumerate(measurements))
        plt.bar(list(range(len(measurements))), contributions, tick_label=names)
        plt.title("Error Contributions")
        plt.show()

    def __get_value_error_pair(self) -> ValueWithError:
        """Gets the value-error pair for the current specified error method"""
        error_method = self.error_method.value
        return self.__evaluators[error_method].evaluate(self._formula)


def get_covariance(var1: ExperimentalValue, var2: ExperimentalValue) -> float:
    """Finds the covariances between two ExperimentalValue instances

    Args:
        var1, var2 (ExperimentalValue): the two values to find covariance between

    Returns:
        The covariance between var1 and var2

    See Also:
        :py:func:`ExperimentalValue.get_covariance`

    """

    if any(not isinstance(var, ExperimentalValue) for var in [var1, var2]):
        raise IllegalArgumentError(
            "Cannot find covariance between non-QExPy defined variables")

    # As of now, only covariance between measurements are supported.
    if isinstance(var1, MeasuredValue) and isinstance(var2, MeasuredValue):
        return var1.get_covariance(var2)

    return 0


def set_covariance(var1: ExperimentalValue, var2: ExperimentalValue, cov: Real = None):
    """Sets the covariance between two measurements

    Args:
        var1, var2 (ExperimentalValue): the two values to set covariance between

    See Also:
        :py:func:`ExperimentalValue.set_covariance`

    Examples:
        >>> import qexpy as q
        >>> a = q.Measurement(5, 0.5)
        >>> b = q.Measurement(6, 0.3)

        >>> # The user can manually set the covariance between two values
        >>> q.set_covariance(a, b, 0.135)
        >>> q.get_covariance(a, b)
        0.135

    """

    if any(not isinstance(var, ExperimentalValue) for var in [var1, var2]):
        raise IllegalArgumentError(
            "Cannot set covariance between non-QExPy defined variables")

    var1.set_covariance(var2, cov)


def get_correlation(var1: ExperimentalValue, var2: ExperimentalValue) -> float:
    """Finds the correlation between two ExperimentalValue instances

    Args:
        var1, var2 (ExperimentalValue): the two values to find correlation between

    Returns:
        The correlation factor between var1 and var2

    See Also:
        :py:func:`ExperimentalValue.get_correlation`

    """

    if any(not isinstance(var, ExperimentalValue) for var in [var1, var2]):
        raise IllegalArgumentError(
            "Cannot find correlation between non-QExPy defined variables")

    # As of now, only covariance between measurements are supported.
    if isinstance(var1, MeasuredValue) and isinstance(var2, MeasuredValue):
        return var1.get_correlation(var2)

    return 0


def set_correlation(var1: MeasuredValue, var2: MeasuredValue, corr: Real = None):
    """Sets the correlation factor between two MeasuredValue objects

    Args:
        var1, var2 (ExperimentalValue): the two values to set correlation between

    See Also:
        :py:func:`ExperimentalValue.set_correlation`

    """
    if any(not isinstance(var, ExperimentalValue) for var in [var1, var2]):
        raise IllegalArgumentError(
            "Cannot set correlation between non-QExPy defined variables")

    var1.set_correlation(var2, corr)


def reset_correlations():
    """resets all correlation settings"""
    ExperimentalValue._correlations.clear()  # pylint: disable=protected-access


def get_variable_by_id(variable_id: uuid.UUID) -> ExperimentalValue:
    """Internal method used to retrieve an ExperimentalValue instance with its ID"""
    return ExperimentalValue.get(variable_id)
