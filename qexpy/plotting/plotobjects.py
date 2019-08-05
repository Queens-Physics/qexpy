"""This file contains definition for objects on a plot"""

import abc
import inspect
import numpy as np
import qexpy.plotting.plot_utils as utils
import qexpy.data.datasets as dts
from qexpy.utils.exceptions import InvalidRequestError, IllegalArgumentError


class ObjectOnPlot(abc.ABC):
    """A container for anything to be plotted"""

    def __init__(self, *args, **kwargs):
        fmt = kwargs.get("fmt", args[0] if args and isinstance(args[0], str) else "")
        utils.validate_fmt(fmt)
        self._fmt = fmt

    @property
    def fmt(self) -> str:
        """The format string to be used in PyPlot"""
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: str):
        utils.validate_fmt(fmt)
        self._fmt = fmt

    @staticmethod
    def create_object_on_plot(*args, **kwargs):
        """Factory method that creates the appropriate object to be plotted"""

        # first check if the object requested is a function
        func, fmt = _try_function_on_plot(*args, **kwargs)
        if func:
            return FunctionOnPlot(func, fmt=fmt, **kwargs)

        # check if a complete data set was passed in
        dataset, fmt = _try_data_set_on_plot(*args, **kwargs)
        if dataset:
            return XYDataSetOnPlot.from_xy_dataset(dataset, fmt=fmt)

        # else try to create a data set out of the arguments
        xdata, ydata, fmt = _try_xdata_and_y_data(*args, **kwargs)
        if xdata and ydata:
            return XYDataSetOnPlot(xdata, ydata, fmt=fmt, **kwargs)

        raise IllegalArgumentError("Invalid combination of arguments for creating an object on plot.")


class XYObjectOnPlot(ObjectOnPlot):
    """A container for objects with x and y values to be drawn on a plot"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        xrange = kwargs.get("xrange", ())
        utils.validate_xrange(xrange, allow_empty=True)
        self._xrange = xrange

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
        XYObjectOnPlot.__init__(self, *args, **kwargs)
        dts.XYDataSet.__init__(self, xdata, ydata, **kwargs)

    @property
    def xvalues_to_plot(self):
        if self.xrange:
            indices = (self.xrange[0] <= self.xvalues) & (self.xvalues < self.xrange[1])
            return self.xvalues[indices]
        return self.xvalues

    @property
    def yvalues_to_plot(self):
        if self.xrange:
            indices = (self.xrange[0] <= self.xvalues) & (self.xvalues < self.xrange[1])
            return self.yvalues[indices]
        return self.yvalues

    @classmethod
    def from_xy_dataset(cls, dataset, **kwargs):
        """Wraps a regular XYDataSet object in a XYDataSetOnPlot object"""
        dataset.__class__ = cls
        ObjectOnPlot.__init__(dataset, **kwargs)
        return dataset


class FunctionOnPlot(XYObjectOnPlot):
    """This is the wrapper for a function to be plotted"""

    def __init__(self, func, **kwargs):
        XYObjectOnPlot.__init__(self, **kwargs)
        self.pars = []
        self.xrange = kwargs.get("xrange", ())
        self.pars = kwargs.get("pars", [])

        # this checks if the xrange of plot is specified by user or auto-generated
        self.xrange_specified = "xrange" in kwargs

        parameters = inspect.signature(func).parameters
        if len(parameters) > 1 and not self.pars:
            raise IllegalArgumentError("For a function with parameters, a list of parameters has to be supplied.")

        if len(parameters) == 1:
            self.func = func
        elif len(parameters) > 1:
            self.func = lambda x: func(x, *self.pars)
        else:
            raise IllegalArgumentError("The function supplied does not have an x-variable.")

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def xvalues_to_plot(self):
        if not self.xrange:
            raise InvalidRequestError("The domain of this function cannot be found.")
        return np.linspace(self.xrange[0], self.xrange[1], 100)

    @property
    def yvalues_to_plot(self):
        if not self.xrange:
            raise InvalidRequestError("The domain of this function cannot be found.")
        return np.vectorize(self.func)(self.xvalues_to_plot)


class HistogramOnPlot(dts.ExperimentalValueArray, ObjectOnPlot):
    """Represents a histogram to be drawn on a plot"""


def _try_function_on_plot(*args, **kwargs):
    func = kwargs.get("func", args[0] if args and inspect.isfunction(args[0]) else None)
    # if there's a second argument, assume that is is the fmt string
    fmt = kwargs.get("fmt", args[1] if len(args) > 1 and isinstance(args[1], str) else "")
    return func, fmt


def _try_data_set_on_plot(*args, **kwargs):
    dataset = kwargs.get("dataset", args[0] if args and isinstance(args[0], dts.XYDataSet) else None)
    fmt = kwargs.get("fmt", args[1] if len(args) > 1 and isinstance(args[1], str) else "")
    return dataset, fmt


def _try_xdata_and_y_data(*args, **kwargs):
    xdata = kwargs.get("xdata", args[0] if len(args) >= 2 else None)
    ydata = kwargs.get("xdata", args[1] if len(args) >= 2 else None)
    fmt = kwargs.get("fmt", args[2] if len(args) >= 3 else "")
    return xdata, ydata, fmt
