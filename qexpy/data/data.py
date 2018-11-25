"""Module containing the data structure for experimental values

This module contains the base class ExperimentalValue and all its sub-classes. The
base class represents any object with a value and an uncertainty. An ExperimentalValue
can be the result of an directly recorded measurement, or the result of an operation
done with other instances of ExperimentalValue, which are represented by MeasuredValue
and DerivedValue.

This module also provides method overloads for different numeric calculations, so that
any instance of ExperimentalValue can be treated as a regular variable in Python, and
during operations, error propagation will be automatically completed in the background.

"""

import numpy as np
import numbers
import warnings
import uuid

from qexpy.utils.utils import ARRAY_TYPES
from qexpy.settings.literals import RECORDED
from . import operations as op
import qexpy.settings.literals as lit
import qexpy.utils.units as units
import qexpy.utils.printing as printing
import qexpy.settings.settings as settings


class ExperimentalValue:
    """Root class for objects with a value and an uncertainty

    This class should not be instantiated directly. Use the Measurement method to create
    new instances of a MeasuredValue object. The result of operations done with other
    ExperimentalValue objects will be recorded as a DerivedValue, which is also a child of
    this class.

    An ExperimentalValue instance can hold multiple value-error pairs. For a MeasuredValue
    object, there can only be one value-error pair. However, for DerivedValue objects which
    are the results of operations, have three value-error pairs, each the result of a
    different error method

    The values attribute contains either 1 or 3 value-error pairs. A value-error pair is
    represented as a tuple, with the first entry being the value, and the second being the
    uncertainty on the value. The keys indicate the source of the value-error pair.
    "recorded" suggests that it's a MeasuredValue object with a user recorded value. For
    DerivedValue objects, there should be 3 value-error pairs, which is the result of three
    different methods for error propagation. Values for the keys can be found in literals

    Attributes:
        _values (dict): the value-error pairs for this object
        _units (dict): the unit of this value
        _name (str): a name can be given to this value
        _id (str): the unique id of this instance

    """

    _register = {}  # module level database for all instantiated ExperimentalValue objects

    def __init__(self, unit="", name=""):
        """Default constructor, not to be called directly"""

        # initialize values
        self._values = {}
        self._units = units.parse_units(unit) if unit else {}
        self._name = name
        self._id = uuid.uuid4()

    def __str__(self):
        if isinstance(self, Constant):
            return ""  # There's no point printing a constant value
        string = ""
        # print name of the quantity
        if self.name:
            string += self.name + " = "
        # print the value and error
        string += self._print_value()
        if len(self._units) > 0:
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
    def name(self, new_name):
        if isinstance(new_name, str):
            self._name = new_name

    @property
    def unit(self) -> str:
        if len(self._units) == 0:
            return ""
        return units.construct_unit_string(self._units)

    @unit.setter
    def unit(self, new_unit):
        if isinstance(new_unit, str):
            self._units = units.parse_units(new_unit)

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __neg__(self):
        return DerivedValue({
            lit.OPERATOR: lit.NEG,
            lit.OPERANDS: [self._id]
        })

    def __add__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.ADD,
            lit.OPERANDS: [self._id, ExperimentalValue._wrap_operand(other)._id]
        })

    def __sub__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.SUB,
            lit.OPERANDS: [self._id, ExperimentalValue._wrap_operand(other)._id]
        })

    def __mul__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.MUL,
            lit.OPERANDS: [self._id, ExperimentalValue._wrap_operand(other)._id]
        })

    def __truediv__(self, other):
        return DerivedValue({
            lit.OPERATOR: lit.DIV,
            lit.OPERANDS: [self._id, ExperimentalValue._wrap_operand(other)._id]
        })

    def derivative(self, other):
        """Finds the derivative with respect to another ExperimentalValue

        This method is usually called from a DerivedValue object. When you take the derivative
        of a measured value with respect to anything other than itself, the return value should
        be 0. The derivative of any quantity with respect to itself is always 1

        Args:
            other (ExperimentalValue): the other ExperimentalValue

        """
        return 1 if self._id == other._id else 0

    def get_units(self):
        from copy import deepcopy
        return deepcopy(self._units)

    def _print_value(self) -> str:
        """Helper method that prints the value-error pair in proper format"""
        if self.value == float('inf'):
            return "inf"
        return printing.get_printer()(self.value, self.error)

    @classmethod
    def _wrap_operand(cls, operand):
        """Make sure that the operand is an ExperimentalValue

        This method is used in derivative calculations. When traversing through the syntax
        tree. If the children of a node is a string, assume the string is the unique ID of
        another ExperimentalValue, and find the reference to that value in the register.
        If the children is a number, wrap the number with a Constant object

        """
        if isinstance(operand, list):
            return list(map(cls._wrap_operand, operand))
        elif isinstance(operand, uuid.UUID) or isinstance(operand, str):
            return cls._register[operand]
        elif isinstance(operand, numbers.Real):
            return Constant(operand)
        elif isinstance(operand, ExperimentalValue):
            return operand
        else:
            raise Exception("The operand: {} of this operation is invalid!".format(operand))


class MeasuredValue(ExperimentalValue):
    """Root class for values with an uncertainty recorded by the user

    This class is used to hold raw measurement data recorded by a user. It is not to
    be instantiated directly. Use the Measurement method instead.

    """

    def __init__(self, value=0.0, error=0.0, unit="", name=""):
        super().__init__(unit, name)
        self._values[RECORDED] = (value, error)

        # register value in module level register
        ExperimentalValue._register[self._id] = self

    @property
    def value(self) -> float:
        """Gets the value for this measurement"""
        return self._values[RECORDED][0]

    @value.setter
    def value(self, new_value):
        """Modifies the value of a measurement"""
        if isinstance(new_value, numbers.Real):
            self._values[RECORDED] = (new_value, self._values[RECORDED][1])
        else:
            raise ValueError("You can only set the value of a measurement to a number")
        if hasattr(self, "_raw_data"):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the value of a repeated measurement. Doing so has "
                          "caused you to lose the original list of raw measurement data")
            self.__class__ = MeasuredValue  # casting it to base class

    @property
    def error(self) -> float:
        """Gets the uncertainty on the measurement"""
        return self._values[RECORDED][1]

    @error.setter
    def error(self, new_error):
        """Modifies the value of a measurement

        Args:
            new_error (float): The new uncertainty

        """
        if isinstance(new_error, numbers.Real) and new_error > 0:
            self._values[RECORDED] = (self._values[RECORDED][0], new_error)
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
            self._values[RECORDED] = (self.value, new_error)
        else:
            raise ValueError("The relative uncertainty of a measurement has to be a positive number")
        if hasattr(self, "_raw_data"):  # check if the instance is a repeated measurement
            warnings.warn("You are trying to modify the uncertainty of a repeated measurement.")


class RepeatedlyMeasuredValue(MeasuredValue):
    """The result of repeated measurements of a single quantity

    An instance of this class will be created when the user takes multiple measurements
    of the same quantity. The mean and error on the mean is used as the value and
    uncertainty of this measurement. The raw array of measurement data is preserved

    The user also has an option of using the standard deviation as the error on this
    measurement, by using the "use_std_for_uncertainty()" method. To set it back to
    error on the mean, use "use_error_on_mean_for_uncertainty()"

    The user can also manually set the uncertainty of this object. However, if the user
    choose to manually override the value of this measurement. The original raw data
    will be lost, and the instance will be casted to its parent class MeasuredValue

    Attributes:
        _std (float): the standard derivative of set of measurements
        _error_on_mean (float): the error on the mean of the set of measurements
        _raw_data (np.ndarray): the original list of raw measurements

    """

    def __init__(self, measurement_array, unit, name):
        super().__init__(unit=unit, name=name)
        measurements = np.array(measurement_array)
        self._std = measurements.std()
        self._error_on_mean = self._std / np.sqrt(measurements.size)
        self._values[RECORDED] = (measurements.mean(), self._error_on_mean)
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
        value = self._values[RECORDED]
        self._values[RECORDED] = (value[0], self._std)

    def use_error_on_mean_for_uncertainty(self):
        value = self._values[RECORDED]
        self._values[RECORDED] = (value[0], self._error_on_mean)

    def show_histogram(self):
        """Plots the raw measurement data in a histogram

        For the result of repeated measurements of a single quantity, the raw measurement
        data is preserved. With this method, you can visualize these values in a histogram.
        with lines corresponding to the mean and the range covered by one standard deviation

        """


# noinspection PyPep8Naming
def Measurement(*args, **kwargs):
    """Records a measurement with uncertainties

    This method is used to create a MeasuredValue object from a single measurement or
    an array of repeated measurements of a single quantity (if you want them averaged).
    This method is named upper case because it is a wrapper for constructors, and should
    look like a constructor from the outside

    When two values are passed to this method, the first argument will be recognized as
    the value, the second as the uncertainty. If the second value is not provided, the
    uncertainty is by default set to 0. If a list of values is passed to this method,
    the mean and standard deviation of the value will be calculated and returned as
    the value and error of the MeasuredValue object.

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

    if len(args) == 1 and isinstance(args[0], ARRAY_TYPES):
        return RepeatedlyMeasuredValue(args[0], unit, name)
    elif len(args) == 1 and isinstance(args[0], numbers.Real):
        return MeasuredValue(float(args[0]), 0.0, unit, name)
    elif len(args) == 2 and isinstance(args[0], numbers.Real) and isinstance(args[1], numbers.Real):
        return MeasuredValue(float(args[0]), float(args[1]), unit, name)
    else:
        raise ValueError("Input must be either a single array of values, or the central value "
                         "and its uncertainty in one measurement")


class DerivedValue(ExperimentalValue):
    """The result of operations done with other ExperimentalValue objects

    This class is not to be instantiated directly. It will be created when operations
    are done on other ExperimentalValue objects. The error of the DerivedValue object will
    be propagated from the original ExperimentalValue objects.

    The DerivedValue object stores information about how it is obtained in the "_formula"
    attribute as an expression tree. The nodes of the expression tree are the operators,
    and the leaves are the operands, which are the unique IDs of instances of the
    ExperimentalValue class

    Attributes:
        _error_method (ErrorMethod): the error method used for this value
        _is_error_method_specified (bool): true if the user specified an error method for
            this value, false if default was used
        _formula (dict): the formula of this object

    """

    def __init__(self, formula, unit="", name="", error_method=None):
        """The default constructor for the result of an operation

        The operation is stored as a syntax tree in the "_formula" attribute.

        Args:
            formula (dict): the formula organized as a tree
            unit (str): The unit of the value
            name (str): The name of this value

        """
        super().__init__(unit, name)

        # register value in module level register
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

        # propagate results
        self._values = op.execute(formula[lit.OPERATOR], ExperimentalValue._wrap_operand(formula[lit.OPERANDS]))
        self._units = op.propagate_units(formula[lit.OPERATOR], ExperimentalValue._wrap_operand(formula[lit.OPERANDS]))

    @property
    def value(self) -> float:
        """Gets the value for this calculated quantity"""
        if self._is_error_method_specified:
            return self._values[self._error_method.value][0]
        else:
            return self._values[settings.get_error_method().value][0]

    @value.setter
    def value(self, new_value):
        """Modifies the value of this quantity"""
        if isinstance(new_value, numbers.Real):
            self._values = {RECORDED: (new_value, self.error)}  # reset the values of this object
            # TODO: don't remember to reset the relations between this instance and other values
            warnings.warn("You are trying to modify the value of a calculated quantity. Doing so has caused "
                          "this value to be regarded as a Measurement and all other information lost")
            self.__class__ = MeasuredValue  # casting it to MeasuredValue
        else:
            raise ValueError("You can only set the value of a ExperimentalValue to a number")

    @property
    def error(self) -> float:
        """Gets the uncertainty on the calculated quantity"""
        if self._is_error_method_specified:
            return self._values[self._error_method.value][1]
        else:
            return self._values[settings.get_error_method().value][1]

    @error.setter
    def error(self, new_error):
        """Modifies the uncertainty of this quantity

        This is not recommended. Doing so will cause this object to be casted to MeasuredValue

        Args:
            new_error (float): The new uncertainty

        """
        if isinstance(new_error, numbers.Real) and new_error > 0:
            self._values = {RECORDED: (self.value, new_error)}  # reset the values of this object
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
            self._values = {RECORDED: (self.value, new_error)}  # reset the values of this object
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
        operands = list(map(ExperimentalValue._wrap_operand, raw_operands))
        return op.differentiator(root_operator)(other._id, *operands)


class Constant(ExperimentalValue):
    """A value with no uncertainty

    This is created when a constant (int, float, etc.) is used in operation with another
    ExperimentalValue. This class is instantiated before calculating operations to ensure
    objects can be combined.

    """

    def __init__(self, value, unit="", name=""):
        super().__init__(unit=unit, name=name)
        if isinstance(value, numbers.Real):
            self._values[RECORDED] = (value, 0)
        else:
            raise ValueError("The value must be a number")

    def derivative(self, other):
        return 0  # the derivative of a constant with respect to anything is 0


class MeasurementArray(np.ndarray):
    """An array of measurements

    This class is used to hold a series of measurements. It can be used for data analysis,
    fitting, and plotting.

    """
