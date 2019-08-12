"""This file contains definition for objects on a plot"""

import abc
import inspect
from collections.abc import Iterable
import numpy as np
import qexpy.plotting.plot_utils as utils
import qexpy.data.data as dt
import qexpy.data.datasets as dts
from qexpy.utils.exceptions import InvalidRequestError, IllegalArgumentError


class ObjectOnPlot(abc.ABC):
    """A container for anything to be plotted"""

    def __init__(self, *args, **kwargs):
        fmt = kwargs.pop("fmt", args[0] if args and isinstance(args[0], str) else "")
        utils.validate_fmt(fmt)
        self._fmt = fmt
        self.label = kwargs.pop("label", "")
        self.kwargs = kwargs

    @property
    def fmt(self) -> str:
        """The format string to be used in PyPlot"""
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: str):
        utils.validate_fmt(fmt)
        self._fmt = fmt

    @staticmethod
    def create_xy_object_on_plot(*args, **kwargs) -> "XYObjectOnPlot":
        """Factory method that creates the appropriate object to be plotted"""

        functions_to_try = [_try_function_on_plot, _try_data_set_on_plot, _try_xdata_and_y_data]

        # The three functions above returns an XYObjectOnPlot if the provided arguments works, and
        # returns None otherwise. Try these functions in order, and return the first valid result
        obj = next((_obj for _obj in map(lambda func: func(*args, **kwargs), functions_to_try) if _obj), None)

        if not obj:
            raise IllegalArgumentError("Invalid combination of arguments for creating an object on plot.")

        return obj

    @staticmethod
    def create_histogram_on_plot(*args, **kwargs) -> "HistogramOnPlot":
        """Factory method that creates a histogram object to be plotted"""

        obj = _try_histogram_on_plot(*args, **kwargs)

        if not obj:
            raise IllegalArgumentError("Invalid combination of arguments for creating a histogram on plot.")

        return obj


class XYObjectOnPlot(ObjectOnPlot):
    """A container for objects with x and y values to be drawn on a plot"""

    def __init__(self, *args, **kwargs):
        """Constructor for XYObjectOnPlot"""

        xrange = kwargs.pop("xrange", ())
        utils.validate_xrange(xrange, allow_empty=True)
        self._xrange = xrange

        super().__init__(*args, **kwargs)

    @property
    @abc.abstractmethod
    def xvalues_to_plot(self) -> np.ndarray:
        """The array of x-values to be plotted"""

    @property
    @abc.abstractmethod
    def yvalues_to_plot(self) -> np.ndarray:
        """The array of y-values to be plotted"""

    @property
    def xrange(self) -> tuple:
        """The range of values to be plotted"""
        return self._xrange

    @xrange.setter
    def xrange(self, new_range: tuple):
        utils.validate_xrange(new_range)
        self._xrange = new_range


class XYDataSetOnPlot(dts.XYDataSet, XYObjectOnPlot):
    """A wrapper for an XYDataSet to be plotted"""

    def __init__(self, xdata, ydata, *args, **kwargs):

        # call super constructors
        XYObjectOnPlot.__init__(self, *args, **kwargs)
        dts.XYDataSet.__init__(self, xdata, ydata, **kwargs)

        # set default label and fmt if not requested
        if not self.label:
            self.label = self.name
        if not self.fmt:
            self.fmt = "o"

    @property
    def xvalues_to_plot(self):
        if self.xrange:
            return self.xvalues[self.__get_indices_from_xrange()]
        return self.xvalues

    @property
    def yvalues_to_plot(self):
        if self.xrange:
            return self.yvalues[self.__get_indices_from_xrange()]
        return self.yvalues

    @property
    def xerr_to_plot(self):
        if self.xrange:
            return self.xerr[self.__get_indices_from_xrange()]
        return self.xerr

    @property
    def yerr_to_plot(self):
        if self.xrange:
            return self.yerr[self.__get_indices_from_xrange()]
        return self.yerr

    @classmethod
    def from_xy_dataset(cls, dataset, **kwargs):
        """Wraps a regular XYDataSet object in a XYDataSetOnPlot object"""
        dataset.__class__ = cls
        XYObjectOnPlot.__init__(dataset, **kwargs)
        return dataset

    def __get_indices_from_xrange(self):
        return (self.xrange[0] <= self.xvalues) & (self.xvalues < self.xrange[1])


class FunctionOnPlot(XYObjectOnPlot):
    """This is the wrapper for a function to be plotted"""

    def __init__(self, func, **kwargs):
        """Constructor for FunctionOnPlot"""

        # this checks if the xrange of plot is specified by user or auto-generated
        self.xrange_specified = "xrange" in kwargs

        self.pars = kwargs.pop("pars", [])

        parameters = inspect.signature(func).parameters
        if len(parameters) > 1 and not self.pars:
            raise IllegalArgumentError("For a function with parameters, a list of parameters has to be supplied.")

        if len(parameters) == 1:
            self.func = func
        elif len(parameters) > 1:
            self.func = lambda x: func(x, *self.pars)
        else:
            raise IllegalArgumentError("The function supplied does not have an x-variable.")

        XYObjectOnPlot.__init__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def xvalues_to_plot(self) -> np.ndarray:
        if not self.xrange:
            raise InvalidRequestError("The domain of this function cannot be found.")
        return np.linspace(self.xrange[0], self.xrange[1], 100)

    @property
    def yvalues_to_plot(self) -> np.ndarray:
        if not self.xrange:
            raise InvalidRequestError("The domain of this function cannot be found.")
        result = self.func(self.xvalues_to_plot)
        simplified_result = list(res.value if isinstance(res, dt.DerivedValue) else res for res in result)
        return np.asarray(simplified_result)

    @property
    def yerr_to_plot(self) -> np.ndarray:
        yvalues = self.yvalues_to_plot
        if isinstance(yvalues, dts.ExperimentalValueArray):
            return yvalues.errors
        return np.empty(0)


class HistogramOnPlot(dts.ExperimentalValueArray, ObjectOnPlot):
    """Represents a histogram to be drawn on a plot"""

    def __init__(self, *args, **kwargs):
        ObjectOnPlot.__init__(self, *args, **kwargs)
        dts.ExperimentalValueArray.__init__(self, **kwargs)

    @classmethod
    def from_value_array(cls, array, **kwargs):
        """Wraps a regular XYDataSet object in a XYDataSetOnPlot object"""
        array.__class__ = cls
        ObjectOnPlot.__init__(array, **kwargs)
        return array


def _try_function_on_plot(*args, **kwargs):
    """Helper function which tries to create a FunctionOnPlot with the provided arguments"""

    func = kwargs.pop("func", args[0] if args and callable(args[0]) else None)
    # if there's a second argument, assume that is is the fmt string
    fmt = kwargs.pop("fmt", args[1] if len(args) > 1 and isinstance(args[1], str) else "")

    return FunctionOnPlot(func, fmt=fmt, **kwargs) if func else None


def _try_data_set_on_plot(*args, **kwargs):
    """Helper function which tries to create a XYDataSetOnPlot with an existing XYDataSet"""

    dataset = kwargs.pop("dataset", args[0] if args and isinstance(args[0], dts.XYDataSet) else None)
    fmt = kwargs.pop("fmt", args[1] if len(args) > 1 and isinstance(args[1], str) else "")

    return XYDataSetOnPlot.from_xy_dataset(dataset, fmt=fmt) if dataset else None


def _try_xdata_and_y_data(*args, **kwargs):
    """Helper function which tries to create an XYDataSetOnPlot with xdata and ydata"""

    xdata = kwargs.pop("xdata", args[0] if len(args) >= 2 and isinstance(args[0], Iterable) else np.empty(0))
    ydata = kwargs.pop("xdata", args[1] if len(args) >= 2 and isinstance(args[1], Iterable) else np.empty(0))
    fmt = kwargs.pop("fmt", args[2] if len(args) >= 3 and isinstance(args[2], str) else "")

    # wrapping data in numpy array objects
    if isinstance(xdata, list):
        xdata = np.asarray(xdata)
    if isinstance(ydata, list):
        ydata = np.asarray(ydata)

    return XYDataSetOnPlot(xdata, ydata, fmt=fmt, **kwargs) if xdata.size and ydata.size else None


def _try_histogram_on_plot(*args, **kwargs):
    """Helper function which tries to create a HistogramOnPlot with the arguments provided"""

    data = kwargs.pop("data", args[0] if args and isinstance(args[0], (np.ndarray, list)) else np.empty(0))
    if isinstance(data, dts.ExperimentalValueArray):
        return HistogramOnPlot.from_value_array(data, **kwargs)
    if (isinstance(data, list) and data) or (isinstance(data, np.ndarray) and data.size):
        return HistogramOnPlot(data, *args, **kwargs)

    return None
