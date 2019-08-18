"""This module contains data structures for sets of multiple experimental values"""

import math as m
import re
import warnings
from numbers import Real
from typing import List, Union, Tuple

import numpy as np

import qexpy.utils.utils as utils
import qexpy.utils.units as units
from qexpy.utils.exceptions import InvalidArgumentTypeError, IllegalArgumentError

from .data import MeasuredValue, ExperimentalValue


class ExperimentalValueArray(np.ndarray):
    """An array of experimental values. alias: MeasurementArray

    A ExperimentalValueArray (MeasurementArray) represents a series of ExperimentalValue objects, It can
    be used for data analysis, fitting, and plotting. This is implemented as a subclass of numpy.ndarray.

    For backwards compatibility, this class is given the alias "MeasurementArray". Users are encouraged
    to call "MeasurementArray" instead of "ExperimentalValueArray" to record arrays of measurements.

    Args:
        data: a list of measurements

    Keyword Args:
        error: the uncertainty of the measurements. It can either be an array of uncertainties, corresponding to
            each individual measurement, or a single uncertainty that applies to all measurements in this series.
        relative_error: the uncertainty of the measurements as a ratio to the center value
        unit: the unit of this series of measurements
        name: the name of this series of measurements

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

        >>> # We can also create an array of measurements with a single uncertainty. As usual,
        >>> # if the error is not specified, they will be set to 0 by default
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

        >>> # Manipulation of a MeasurementArray is also very easy. We can append or insert into
        >>> # the array values in multiple formats
        >>> a = a.append((7, 0.2))  # a measurement can be inserted as a tuple
        >>> print(a[5])
        length = 7.0 +/- 0.2 [m]
        >>> a = a.insert(0, 8)  # if error is not specified, it is set to 0 by default
        >>> print(a[0])
        length = 8 +/- 0 [m]

        >>> # The same operations also works with array-like objects, in which case they are
        >>> # concatenated
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

        >>> # The ExperimentalValueArray object is vectorized just like numpy.ndarray. You can
        >>> # perform basic arithmetic operations as well as functions with them and get back
        >>> # ExperimentalValueArray objects
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

    def __new__(cls, data: List[Real], *args, **kwargs):
        """Default constructor for a ExperimentalValueArray

        __new__ is used instead of __init__ for object initialization. This is the convention for
        subclassing numpy arrays.

        """

        # initialize raw data and its corresponding uncertainties
        if not isinstance(data, utils.ARRAY_TYPES):
            raise InvalidArgumentTypeError("creating MeasurementArray",
                                           got=[data], expected="a list or real numbers")
        measurements = np.asarray(data)

        error = kwargs.pop("error", args[0] if args else 0)
        relative_error = kwargs.pop("relative_error", 0)

        if relative_error and error:
            raise IllegalArgumentError("You either specify the absolute error or the relative error, not both.")

        error = 0 if not error else error

        if relative_error:
            error_array = float(relative_error) * measurements
        elif isinstance(error, Real):
            error_array = [float(error)] * len(measurements)
        elif isinstance(error, utils.ARRAY_TYPES) and len(error) == len(data):
            error_array = np.asarray(error)
        else:
            raise InvalidArgumentTypeError("uncertainties of a MeasurementArray",
                                           got=error, expected="real number or a list of real numbers")

        unit = kwargs.pop("unit", "")
        name = kwargs.pop("name", "")

        measured_values = list(MeasuredValue(val, err, unit, name) for val, err in zip(measurements, error_array))
        for index, measurement in enumerate(measured_values):
            measurement.name = "{}_{}".format(measurement.name, index) if measurement.name else ""
        obj = np.asarray(measured_values, dtype=ExperimentalValue).view(ExperimentalValueArray)

        # added so that subclasses of this are of the correct type
        obj.__class__ = cls

        return obj

    def __str__(self):
        string = ", ".join(variable.print_value() for variable in self)
        name = self.name + " = " if self.name else ""
        unit = " (" + self.unit + ")" if self.unit else ""
        return "{}[ {} ]{}".format(name, string, unit)

    def __setitem__(self, key, value):
        value = self.__wrap_measurement(value)
        super().__setitem__(key, value)

    def __array_finalize__(self, obj):
        """wrap up array initialization"""
        if obj is None or not self.shape or not isinstance(self[0], ExperimentalValue):
            return  # skip if this is not a regular array of ExperimentalValue objects
        name = getattr(obj, "name", "") if hasattr(obj, "name") else getattr(self, "name", "")
        for index, measurement in enumerate(self):
            measurement.name = "{}_{}".format(name, index) if name else ""

    @property
    def name(self) -> str:
        """name of this array of values

        A name can be given to this data set, and each measurement within this list will be named
        in the form of "name_index". For example, if the name is specified as "length", the items
        in this array will be named "length_0", "length_1", "length_2", ...

        """
        return re.sub(r"_[0-9]+$", "", self[0].name)

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise InvalidArgumentTypeError("name", got=new_name, expected="string")
        for index, measurement in enumerate(self):
            measurement.name = "{}_{}".format(new_name, index)

    @property
    def unit(self) -> str:
        """The unit of this array of values

        It is assumed that the set of data that constitutes one ExperimentalValueArray have the
        same unit, which, when assigned, is given too all the items of the array.

        """
        return self[0].unit

    @unit.setter
    def unit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise InvalidArgumentTypeError("unit", got=new_unit, expected="string")
        if not new_unit:
            return
        new_unit = units.parse_units(new_unit)
        for data in self:
            data._units = new_unit

    @property
    def values(self) -> np.ndarray:
        """An array consisting of the center values of each item"""
        return np.asarray(list(data.value for data in self))

    @property
    def errors(self) -> np.ndarray:
        """An array consisting of the uncertainties of each item"""
        return np.asarray(list(data.error for data in self))

    def append(self, value: Union[Real, List, Tuple, ExperimentalValue]) -> "ExperimentalValueArray":
        """adds a value to the end of this array and returns the new array

        Args:
            value: the value to be appended to this array.

        Returns:
            The new array

        """
        if isinstance(value, ExperimentalValueArray):
            pass  # don't do anything if the new value is already a ExperimentalValueArray
        elif isinstance(value, utils.ARRAY_TYPES):
            value = list(self.__wrap_measurement(value) for value in value)
        else:
            value = self.__wrap_measurement(value)
        # append new value and cast to ExperimentalValueArray
        result = np.append(self, value).view(ExperimentalValueArray)
        for index, measurement in enumerate(result):
            measurement.name = "{}_{}".format(self.name, index)
        return result

    def insert(self, index: int, value: Union[Real, List, Tuple, ExperimentalValue]) -> "ExperimentalValueArray":
        """adds a value to a position in this array and returns the new array

        Args:
            index: the position to insert the value
            value: the value to be inserted

        Returns:
            The new array

        """
        if isinstance(value, ExperimentalValueArray):
            pass  # don't do anything if the new value is already a ExperimentalValueArray
        elif isinstance(value, utils.ARRAY_TYPES):
            value = list(self.__wrap_measurement(value) for value in value)
        else:
            value = self.__wrap_measurement(value)
        result = np.insert(self, index, value).view(ExperimentalValueArray)
        for idx, measurement in enumerate(result):
            measurement.name = "{}_{}".format(self.name, idx)
        return result

    def delete(self, index: int) -> "ExperimentalValueArray":
        """deletes the value on the requested position and returns the new array

        Args:
            index: the index of the value to be deleted

        Returns:
            The new array

        """
        result = np.delete(self, index).view(ExperimentalValueArray)
        for idx, measurement in enumerate(result):
            measurement.name = "{}_{}".format(self.name, idx)
        return result

    def mean(self, **_) -> MeasuredValue:
        """The mean of the array"""
        result = np.mean(self.values)
        error = self.error_on_mean()
        name = "mean of {}".format(self.name) if self.name else ""
        return MeasuredValue(float(result), error, self.unit, name)

    def std(self, ddof=1, **_) -> float:
        """The standard deviation of this array"""
        return float(np.std(self.values, ddof=ddof))

    def sum(self, **_) -> MeasuredValue:
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


class XYDataSet:
    """A pair of ExperimentalValueArray objects

    QExPy is capable of multiple ways of data handling. One typical case in experimental data
    analysis is for a pair of data sets, which is usually plotted or fitted with a curve.

    Args:
        xdata (list|np.ndarray): an array of values for x-data
        ydata (list|np.ndarray): an array of values for y-data

    Keyword Args:
        xerr (list|float): the uncertainty on x data
        yerr (list|float): the uncertainty on y data
        xunit (str): the unit of the x data set
        yunit (str): the unit of the y data set
        xname (str): the name of the x data set
        yname (str): the name of the y data set

    See Also:
        There are different ways to specify an array of data, see :py:class:`.ExperimentalValueArray`
        for more details. If you have a :py:class:`.ExperimentalValueArray` object, you can pass that
        in as well.

    Examples:

        >>> import qexpy as q

        >>> a = q.XYDataSet(xdata=[0,1,2,3,4], ydata=[3,4,5,6,7], xerr=0.5, yerr=[0.1,0.2,0.3,0.4,0.5],
        >>>                 xunit="m", xname="length", yunit="kg", yname="weight")
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

    def __init__(self, xdata, ydata, **kwargs):

        xunit = kwargs.get("xunit", "")
        yunit = kwargs.get("yunit", "")
        xname = kwargs.get("xname", "")
        yname = kwargs.get("yname", "")

        self._name = kwargs.get("name", "")

        xerr = kwargs.get("xerr", None)
        yerr = kwargs.get("yerr", None)

        if len(xdata) != len(ydata):
            raise ValueError("The length of xdata and ydata don't match!")

        self.xdata = self.__wrap_data(xdata, xerr, name=xname, unit=xunit)
        self.ydata = self.__wrap_data(ydata, yerr, name=yname, unit=yunit)

    @property
    def name(self) -> str:
        """The name of this data set"""
        return self._name if self._name else "XY Dataset"

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise InvalidArgumentTypeError("name of data set", got=new_name, expected="string")
        self._name = new_name

    @property
    def xvalues(self) -> np.ndarray:
        """the values of the x data set"""
        return self.xdata.values

    @property
    def xerr(self):
        """the errors of the x data set"""
        return self.xdata.errors

    @property
    def yvalues(self) -> np.ndarray:
        """the values of the y data set"""
        return self.ydata.values

    @property
    def yerr(self):
        """the errors of the x data set"""
        return self.ydata.errors

    @property
    def xname(self) -> str:
        """name of the xdata set"""
        return self.xdata.name

    @xname.setter
    def xname(self, name):
        self.xdata.name = name

    @property
    def xunit(self) -> str:
        """unit of the xdata set"""
        return self.xdata.unit

    @xunit.setter
    def xunit(self, unit):
        self.xdata.unit = unit

    @property
    def yname(self) -> str:
        """name of the ydata set"""
        return self.ydata.name

    @yname.setter
    def yname(self, name):
        self.ydata.name = name

    @property
    def yunit(self) -> str:
        """unit of the ydata set"""
        return self.ydata.unit

    @yunit.setter
    def yunit(self, unit):
        self.ydata.unit = unit

    def fit(self, model, **kwargs):
        """fits the current dataset to a model

        See Also:
            :py:func:`.qexpy.fitting.fit`

        """
        import qexpy.fitting.fitting as fitting  # pylint: disable=cyclic-import
        return fitting.fit(self, model, **kwargs)

    @staticmethod
    def __wrap_data(data, error, unit, name):
        """wraps the data set into ExperimentalValueArray objects"""

        if isinstance(data, ExperimentalValueArray):
            data.name = name if name else data.name
            data.unit = unit if unit else data.unit
            return data
        if isinstance(data, utils.ARRAY_TYPES):
            return ExperimentalValueArray(data, error, unit=unit, name=name)

        raise InvalidArgumentTypeError("Initiate XYDataSet", got=data, expected="an array of real numbers")
