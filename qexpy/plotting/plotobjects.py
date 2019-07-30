"""This file contains definition for objects on a plot"""

import abc
import inspect
import numpy as np
import qexpy.plotting.plot_utils as utils
import qexpy.data.datasets as dts
from qexpy.utils.exceptions import InvalidRequestError, IllegalArgumentError


class ObjectOnPlot(abc.ABC):
    """A container for each individual dataset or function to be plotted"""

    def __init__(self, *args, **kwargs):
        fmt = kwargs.get("fmt", args[0] if isinstance(args[0], str) else "")
        utils.validate_fmt(fmt)
        self._fmt = fmt
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
    def fmt(self) -> str:
        """The format string to be used in PyPlot"""
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: str):
        utils.validate_fmt(fmt)
        self._fmt = fmt

    @property
    def xrange(self) -> tuple:
        """The range of values to be plotted"""
        return self._xrange

    @xrange.setter
    def xrange(self, new_range: tuple):
        utils.validate_xrange(new_range)
        self._xrange = new_range


class XYDataSetOnPlot(dts.XYDataSet, ObjectOnPlot):
    """A wrapper for an XYDataSet to be plotted"""

    def __init__(self, xdata, ydata, *args, **kwargs):
        ObjectOnPlot.__init__(self, *args, **kwargs)
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
    def from_xy_dataset(cls, dataset):
        """Wraps a regular XYDataSet object in a XYDataSetOnPlot object"""
        dataset.__class__ = cls
        ObjectOnPlot.__init__(dataset)
        return dataset


class FunctionOnPlot(ObjectOnPlot):
    """This is the wrapper for a function to be plotted"""

    def __init__(self, func, *args, **kwargs):
        ObjectOnPlot.__init__(self, *args, **kwargs)
        parameters = inspect.signature(func).parameters
        if len(parameters) != 1:
            raise IllegalArgumentError("QExPy only supports plotting functions with one variable.")
        self.func = func
        self.xrange = kwargs.get("xrange", ())

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


def _create_object_on_plot(*args, **kwargs) -> ObjectOnPlot:
    """Factory method to create the appropriate sub-class of ObjectOnPlot"""

    # first check if the object requested is a function
    func = kwargs.get("func", args[0] if inspect.isfunction(args[0]) else None)
    if func:
        return FunctionOnPlot(func, **kwargs)

    # check if a complete data set was passed in
    dataset = kwargs.get("dataset", args[0] if isinstance(args[0], dts.XYDataSet) else None)
    if dataset:
        return XYDataSetOnPlot.from_xy_dataset(dataset)

    # else try to create a data set out of the arguments
    xdata = kwargs.get("xdata", args[0] if len(args) >= 2 else None)
    ydata = kwargs.get("xdata", args[1] if len(args) >= 2 else None)
    fmt = kwargs.get("fmt", args[2] if len(args) >= 3 else "")
    if xdata and ydata:
        return XYDataSetOnPlot(xdata, ydata, fmt=fmt, **kwargs)

    raise IllegalArgumentError("Invalid combination of arguments for creating an object on plot.")
