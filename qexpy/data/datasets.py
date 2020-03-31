"""Defines data structures for collections of individual measurements"""

import re
import warnings

import numpy as np
import math as m

from typing import List  # pylint: disable=unused-import
from numbers import Real

from qexpy.utils import IllegalArgumentError

from . import data as dt
from . import utils as dut

import qexpy.utils as utils

ARRAY_TYPES = np.ndarray, list


class ExperimentalValueArray(np.ndarray):
    """An array of experimental values, alias: MeasurementArray

    An ExperimentalValueArray (MeasurementArray) represents a series of ExperimentalValue
    objects. It is implemented as a sub-class of numpy.ndarray. This class is given an alias
    "MeasurementArray" for more intuitive user interface.

    Args:
        *args: The first argument is an array of real numbers representing the center values
            of the measurements. The second argument (if present) is either a positive real
            number or an array of positive real numbers of the same length as the data array,
            representing the uncertainties on the measurements.

    Keyword Args:
        data (List): an array of real numbers representing the center values
        error (Real|List): the uncertainties on the measurements
        relative_error (Real|List): the relative uncertainties on the measurements
        unit (str): the unit of the measurement
        name (str): the name of the measurement

    Examples:
        >>> import qexpy as q

        >>> # We can instantiate an array of measurements with two lists
        >>> a = q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5])
        >>> a
        ExperimentalValueArray([MeasuredValue(1.0 +/- 0.1),
                    MeasuredValue(2.0 +/- 0.2),
                    MeasuredValue(3.0 +/- 0.3),
                    MeasuredValue(4.0 +/- 0.4),
                    MeasuredValue(5.0 +/- 0.5)], dtype=object)

        >>> # We can also create an array of measurements with a single uncertainty.
        >>> # As usual, if the error is not specified, they will be set to 0 by default
        >>> a = q.MeasurementArray([1, 2, 3, 4, 5], 0.5, unit="m", name="length")
        >>> a
        ExperimentalValueArray([MeasuredValue(1.0 +/- 0.5),
                    MeasuredValue(2.0 +/- 0.5),
                    MeasuredValue(3.0 +/- 0.5),
                    MeasuredValue(4.0 +/- 0.5),
                    MeasuredValue(5.0 +/- 0.5)], dtype=object)

        >>> # We can access the different statistical properties of this array
        >>> print(np.sum(a))
        15 +/- 1 [m]
        >>> print(a.mean())
        3.0 +/- 0.7 [m]
        >>> a.std()
        1.5811388300841898

        >>> # Manipulation of a MeasurementArray is also very easy. We can append or insert
        >>> # into the array values in multiple formats
        >>> a = a.append((7, 0.2))  # a measurement can be inserted as a tuple
        >>> print(a[5])
        length = 7.0 +/- 0.2 [m]
        >>> a = a.insert(0, 8)  # if error is not specified, it is set to 0 by default
        >>> print(a[0])
        length = 8 +/- 0 [m]

        >>> # The same operations also works with array-like objects, in which case they are
        >>> # concatenated into a single array
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

        >>> # The ExperimentalValueArray object is vectorized just like numpy.ndarray. You
        >>> # can perform basic arithmetic operations as well as functions with them and get
        >>> # back ExperimentalValueArray objects
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

    # pylint: disable=no-member,arguments-differ,too-many-function-args

    def __new__(cls, *args, **kwargs):
        """Constructor for an ExperimentalValueArray

        __new__ is used instead of __init__ for object initialization. This is required for
        subclassing the numpy.ndarray type.

        """

        data = kwargs.pop("data", args[0] if args else None)

        if not isinstance(data, ARRAY_TYPES):
            raise TypeError("You have not provided valid data to initialize the array.")
        data = np.asarray(data)

        error = kwargs.pop("error", args[1] if len(args) > 1 else None)
        relative_error = kwargs.pop("relative_error", None)

        error_array = _get_error_array_helper(data, error, relative_error)

        if all(isinstance(x, dt.ExperimentalValue) for x in data):
            if error is None and relative_error is None:
                error_array = None
            return ExperimentalValueArray.__wrap(data, error_array=error_array, **kwargs)

        if not all(isinstance(x, Real) for x in data):
            raise TypeError("Some values in the array are not real numbers")

        values = list(
            dt.MeasuredValue(val, err, **kwargs) for val, err in zip(data, error_array))
        for index, meas in enumerate(values):
            meas.name = "{}_{}".format(meas.name, index) if meas.name else ""

        # Initialize the instance to a numpy.ndarray
        obj = np.asarray(values, dtype=dt.ExperimentalValue).view(ExperimentalValueArray)

        # Added so that subclasses of this are of the correct type
        obj.__class__ = cls

        return obj

    def __str__(self):
        value_errors = ", ".join(variable.print_value_error() for variable in self)
        name = "{} = ".format(self.name) if self.name else ""
        unit = " ({})".format(self.unit) if self.unit else ""
        return "{}[ {} ]{}".format(name, value_errors, unit)

    def __setitem__(self, key, value):
        if isinstance(value, Real):
            self[key].value = value
        else:
            super().__setitem__(
                key, dut.wrap_in_measurement(value, unit=self.unit, name=self.name))
            if self.name:
                self[key].name = "{}_{}".format(self.name, key)

    def __pow__(self, power):
        if isinstance(power, ARRAY_TYPES):
            return super().__pow__(power)
        return super().__pow__(dut.wrap_in_experimental_value(power))

    def __rpow__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__rpow__(other)
        return super().__rpow__(dut.wrap_in_experimental_value(other))

    def __add__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__add__(other)
        return super().__add__(dut.wrap_in_experimental_value(other))

    def __radd__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__radd__(other)
        return super().__radd__(dut.wrap_in_experimental_value(other))

    def __sub__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__sub__(other)
        return super().__sub__(dut.wrap_in_experimental_value(other))

    def __rsub__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__rsub__(other)
        return super().__rsub__(dut.wrap_in_experimental_value(other))

    def __mul__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__mul__(other)
        return super().__mul__(dut.wrap_in_experimental_value(other))

    def __rmul__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__rmul__(other)
        return super().__rmul__(dut.wrap_in_experimental_value(other))

    def __truediv__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__truediv__(other)
        return super().__truediv__(dut.wrap_in_experimental_value(other))

    def __rtruediv__(self, other):
        if isinstance(other, ARRAY_TYPES):
            return super().__rtruediv__(other)
        return super().__rtruediv__(dut.wrap_in_experimental_value(other))

    def __array_finalize__(self, obj):
        """wrap up array initialization"""
        if obj is None or not (self.shape and isinstance(self[0], dt.ExperimentalValue)):
            return  # Skip if this is not a regular array of ExperimentalValue objects
        if hasattr(obj, "name"):
            name = getattr(obj, "name", "")
        else:
            name = getattr(self, "name", "")
        # re-index the names of the measurements
        for index, measurement in enumerate(self):
            measurement.name = "{}_{}".format(name, index) if name else ""

    @property
    def name(self):
        """str: Name of this array of values

        A name can be given to this data set, and each measurement within this list will be
        named in the form of "name_index". For example, if the name is specified as "length",
        the items in this array will be named "length_0", "length_1", "length_2", ...

        """
        return re.sub(r"_[0-9]+$", "", self[0].name)

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise TypeError("Cannot set name to \"{}\"!".format(type(new_name).__name__))
        for index, measurement in enumerate(self):
            measurement.name = "{}_{}".format(new_name, index)

    @property
    def unit(self):
        """str: The unit of this array of values

        It is assumed that the set of data that constitutes one ExperimentalValueArray have
        the same unit, which, when assigned, is given too all the items of the array.

        """
        return self[0].unit

    @unit.setter
    def unit(self, unit_string: str):
        if not isinstance(unit_string, str):
            raise TypeError("Cannot set unit to \"{}\"!".format(type(unit_string).__name__))
        new_unit = utils.parse_unit_string(unit_string) if unit_string else {}
        for data in self:
            data._unit = new_unit

    @property
    def values(self):
        """np.ndarray: An array consisting of the center values of each item"""
        return np.asarray(list(data.value for data in self))

    @property
    def errors(self):
        """np.ndarray: An array consisting of the uncertainties of each item"""
        return np.asarray(list(data.error for data in self))

    def append(self, value) -> "ExperimentalValueArray":
        """Adds a value to the end of this array and returns the new array

        Args:
            value: The value to be appended to this array. This can be a real number, a pair
                of value and error in a tuple, an ExperimentalValue instance, or an array
                consisting of any of the above.

        Returns:
            The new ExperimentalValueArray instance

        """
        value = dut.wrap_in_value_array(value, unit=self.unit, name=self.name)
        result = np.append(self, value).view(ExperimentalValueArray)
        for index, measurement in enumerate(result):
            measurement.name = "{}_{}".format(self.name, index)
            measurement.unit = self.unit
        return result

    def insert(self, index: int, value) -> "ExperimentalValueArray":
        """adds a value to a position in this array and returns the new array

        Args:
            index (int): the position to insert the value
            value: The value to be inserted into this array. This can be a real number, a
                pair of value and error in a tuple, an ExperimentalValue instance, or an
                array consisting of any of the above.

        Returns:
            The new ExperimentalValueArray instance

        """
        value = dut.wrap_in_value_array(value, unit=self.unit, name=self.name)
        result = np.insert(self, index, value).view(ExperimentalValueArray)
        for idx, measurement in enumerate(result):
            measurement.name = "{}_{}".format(self.name, idx)
            measurement.unit = self.unit
        return result

    def delete(self, index: int) -> "ExperimentalValueArray":
        """deletes the value on the requested position and returns the new array

        Args:
            index (int): the index of the value to be deleted

        Returns:
            The new ExperimentalValueArray instance

        """
        result = np.delete(self, index).view(ExperimentalValueArray)
        for idx, measurement in enumerate(result):
            measurement.name = "{}_{}".format(self.name, idx)
        return result

    def mean(self, **_) -> "dt.ExperimentalValue":  # pylint:disable=arguments-differ
        """The mean of the array"""
        result = np.mean(self.values)
        error = self.error_on_mean()
        name = "mean of {}".format(self.name) if self.name else ""
        return dt.MeasuredValue(float(result), error, unit=self.unit, name=name)

    def std(self, ddof=1, **_) -> float:  # pylint:disable=arguments-differ
        """The standard deviation of this array"""
        return float(np.std(self.values, ddof=ddof))

    def sum(self, **_) -> "dt.ExperimentalValue":  # pylint:disable=arguments-differ
        """The sum of the array"""
        result = np.sum(self.values)
        error = np.sqrt(np.sum(self.errors ** 2))
        return dt.MeasuredValue(float(result), float(error), unit=self.unit, name=self.name)

    def error_on_mean(self) -> float:
        """The error on the mean of this array"""
        return self.std() / m.sqrt(self.size)

    def error_weighted_mean(self) -> float:
        """The error weighted mean of this array"""
        if any(err == 0 for err in self.errors):
            warnings.warn(
                "One or more errors are 0, the error weighted mean cannot be calculated.")
            return np.nan
        weights = np.asarray(list(1 / (err ** 2) for err in self.errors))
        return float(np.sum(weights * self.values) / np.sum(weights))

    def propagated_error(self) -> float:
        """The propagated error from the error weighted mean calculation"""
        if any(err == 0 for err in self.errors):
            warnings.warn(
                "One or more errors are 0, the propagated error cannot be calculated.")
            return np.nan
        weights = np.asarray(list(1 / (err ** 2) for err in self.errors))
        return 1 / np.sqrt(np.sum(weights))

    @classmethod
    def __wrap(cls, data, **kwargs):
        """if an array of ExperimentalValue objects are passed in, simply wrap it"""

        error_array = kwargs.get("error_array", None)

        if error_array is not None:
            for x, err in zip(data, error_array):
                x.error = err  # update the errors if specified

        name = kwargs.get("name", None)
        unit = kwargs.get("unit", None)

        if name is not None:
            for idx, x in enumerate(data):
                x.name = "{}_{}".format(name, idx)
        if unit is not None:
            for x in data:
                x.unit = unit

        obj = data.view(ExperimentalValueArray)
        obj.__class__ = cls
        return obj


class XYDataSet:
    """A pair of ExperimentalValueArray objects

    QExPy is capable of multiple ways of data handling. One typical case in experimental data
    analysis is for a pair of data sets, which is usually plotted or fitted with a curve.

    Args:
        xdata (List|np.ndarray): an array of values for x-data
        ydata (List|np.ndarray): an array of values for y-data

    Keyword Args:
        xerr (Real|List): the uncertainty on x data
        yerr (Real|List): the uncertainty on y data
        xunit (str): the unit of the x data set
        yunit (str): the unit of the y data set
        xname (str): the name of the x data set
        yname (str): the name of the y data set

    Examples:

        >>> import qexpy as q

        >>> a = q.XYDataSet(xdata=[0, 1, 2, 3, 4], xerr=0.5, xunit="m", xname="length",
        >>>                 ydata=[3, 4, 5, 6, 7], yerr=[0.1,0.2,0.3,0.4,0.5],
        >>>                 yunit="kg", yname="weight")
        >>> a.xvalues
        array([0, 1, 2, 3, 4])
        >>> a.xerr
        array([0.5, 0.5, 0.5, 0.5, 0.5])
        >>> a.yerr
        array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> a.xdata
        ExperimentalValueArray([MeasuredValue(0.0 +/- 0.5),
                        MeasuredValue(1.0 +/- 0.5),
                        MeasuredValue(2.0 +/- 0.5),
                        MeasuredValue(3.0 +/- 0.5),
                        MeasuredValue(4.0 +/- 0.5)], dtype=object)

    """

    def __init__(self, *args, **kwargs):

        xunit = kwargs.get("xunit", "")
        yunit = kwargs.get("yunit", "")
        xname = kwargs.get("xname", "")
        yname = kwargs.get("yname", "")

        self._name = kwargs.get("name", "")

        xerr = kwargs.get("xerr", None)
        yerr = kwargs.get("yerr", None)

        xdata = kwargs.pop("xdata", args[0] if len(args) >= 2 else None)
        ydata = kwargs.pop("ydata", args[1] if len(args) >= 2 else None)

        xdata = XYDataSet.__wrap_data(xdata, xerr, name=xname, unit=xunit)
        ydata = XYDataSet.__wrap_data(ydata, yerr, name=yname, unit=yunit)

        if len(xdata) != len(ydata):
            raise ValueError("The length of xdata and ydata don't match!")

        self.xdata = xdata  # type: ExperimentalValueArray
        self.ydata = ydata  # type: ExperimentalValueArray

    @property
    def name(self):
        """str: The name of this data set"""
        return self._name if self._name else "XY Dataset"

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise TypeError("Cannot set name to \"{}\"".format(type(new_name).__name__))
        self._name = new_name

    @property
    def xvalues(self):
        """np.ndarray: The values of the x data set"""
        return self.xdata.values

    @property
    def xerr(self):
        """np.ndarray: The errors of the x data set"""
        return self.xdata.errors

    @property
    def yvalues(self):
        """np.ndarray: The values of the y data set"""
        return self.ydata.values

    @property
    def yerr(self):
        """np.ndarray: The errors of the x data set"""
        return self.ydata.errors

    @property
    def xname(self):
        """str: Name of the xdata set"""
        return self.xdata.name

    @xname.setter
    def xname(self, name):
        if not isinstance(name, str):
            raise TypeError("Cannot set xname to \"{}\"".format(type(name).__name__))
        self.xdata.name = name

    @property
    def xunit(self):
        """str: Unit of the xdata set"""
        return self.xdata.unit

    @xunit.setter
    def xunit(self, unit):
        if not isinstance(unit, str):
            raise TypeError("Cannot set xunit to \"{}\"".format(type(unit).__name__))
        self.xdata.unit = unit

    @property
    def yname(self):
        """str: Name of the ydata set"""
        return self.ydata.name

    @yname.setter
    def yname(self, name):
        if not isinstance(name, str):
            raise TypeError("Cannot set yname to \"{}\"".format(type(name).__name__))
        self.ydata.name = name

    @property
    def yunit(self):
        """str: Unit of the ydata set"""
        return self.ydata.unit

    @yunit.setter
    def yunit(self, unit):
        if not isinstance(unit, str):
            raise TypeError("Cannot set yunit to \"{}\"".format(type(unit).__name__))
        self.ydata.unit = unit

    def fit(self, model, **kwargs):
        """Fits the current dataset to a model

        See Also:
            The fit function in the fitting module of QExPy

        """
        import qexpy.fitting as fitting  # pylint: disable=cyclic-import
        return fitting.fit(self, model, **kwargs)

    @staticmethod
    def __wrap_data(data, error, unit, name) -> ExperimentalValueArray:
        """Wraps the data set into ExperimentalValueArray objects"""

        if isinstance(data, ExperimentalValueArray):
            if name:
                data.name = name
            if unit:
                data.unit = unit
            if error is not None:
                error_array = _get_error_array_helper(data, error, None)
                for x, e in zip(data, error_array):
                    x.error = e
            return data
        if isinstance(data, ARRAY_TYPES):
            return ExperimentalValueArray(data, error, unit=unit, name=name)

        raise IllegalArgumentError("Cannot create XYDataSet with the given arguments.")


def _get_error_array_helper(data, error, rel_error):
    """Helper method that produces an error array for an ExperimentalValueArray"""

    if error is None and rel_error is None:
        error_array = [0.0] * len(data)
    elif isinstance(error, Real):
        error_array = [float(error)] * len(data)
    elif isinstance(error, ARRAY_TYPES) and all(isinstance(err, Real) for err in error):
        if len(error) != len(data):
            raise ValueError("The length of the error data arrays don't match.")
        error_array = np.asarray(error)
    elif isinstance(rel_error, Real):
        error_array = float(rel_error) * abs(data)
    elif isinstance(rel_error, ARRAY_TYPES) and all(isinstance(e, Real) for e in rel_error):
        if len(rel_error) != len(data):
            raise ValueError("The length of the relative error and data arrays don't match.")
        error_array = rel_error * abs(data)
    else:
        raise TypeError("The error or relative error provided is invalid!")

    if any(err < 0 for err in error_array):
        raise ValueError("The uncertainty of any measurement cannot be negative!")

    return error_array
