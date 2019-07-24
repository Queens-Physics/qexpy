"""This file contains function definitions for plotting"""

import inspect
import re
import abc

from typing import List
import matplotlib.pyplot as plt
import numpy as np

from qexpy.utils.exceptions import InvalidArgumentTypeError, IllegalArgumentError
import qexpy.settings.literals as lit
import qexpy.data.datasets as dts


class ObjectOnPlot(abc.ABC):
    """A container for each individual dataset or function to be plotted"""

    def __init__(self, *args, **kwargs):
        fmt = kwargs.get("fmt", "")
        xrange = kwargs.get("xrange", ())
        # TODO: validate fmt and xrange
        self._fmt = fmt
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
        # TODO: validate fmt string
        self._fmt = fmt

    @property
    def xrange(self) -> tuple:
        """The range of values to be plotted"""
        return self._xrange

    @xrange.setter
    def xrange(self, new_range: tuple):
        if not isinstance(new_range, tuple) and not isinstance(new_range, list):
            raise InvalidArgumentTypeError("xrange", new_range, "tuple or list of length 2")
        if len(new_range) != 2 or new_range[0] > new_range[1]:
            raise IllegalArgumentError(
                "Error: the xrange has to be a tuple of length 2 where the second number is larger than the first.")
        self._xrange = new_range


class XYDataSetOnPlot(dts.XYDataSet, ObjectOnPlot):
    """A wrapper for an XYDataSet to be plotted"""

    def __init__(self, xdata, ydata, **kwargs):
        ObjectOnPlot.__init__(self, **kwargs)
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

    def __init__(self, func, xrange, **kwargs):
        ObjectOnPlot.__init__(self, **kwargs)
        self.func = func
        self.xrange = xrange

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def xvalues_to_plot(self):
        return np.linspace(self.xrange[0], self.xrange[1], 100)

    @property
    def yvalues_to_plot(self):
        return np.vectorize(self.func)(self.xvalues_to_plot)


def create_object_on_plot(*args, **kwargs) -> ObjectOnPlot:
    """Factory method to create the appropriate sub-class of ObjectOnPlot"""


class Plot:
    """The data structure used for a plot"""

    # points to the latest Plot instance that's created
    current_plot_buffer = None  # type: Plot

    def __init__(self):
        self.objects = []  # type: List[ObjectOnPlot]
        self.plot_info = {
            lit.TITLE: "",
            lit.XLABEL: "",
            lit.YLABEL: "",
            lit.XUNIT: "",
            lit.YUNIT: ""
        }

    @property
    def title(self):
        """The title of this plot, which will appear on top of the figure"""
        return self.plot_info[lit.TITLE]

    @title.setter
    def title(self, new_title: str):
        if not isinstance(new_title, str):
            raise InvalidArgumentTypeError("plot title", got=new_title, expected="string")
        self.plot_info[lit.TITLE] = new_title

    @property
    def xlabel(self):
        """The name of the x data, which will appear as x label"""

        if self.plot_info[lit.XLABEL]:
            # use the user specified label if there is one
            return self.plot_info[lit.XLABEL]

        # else find the first data set and use the name of the data set as the label
        data_set = next((obj for obj in self.objects if isinstance(obj, XYDataSetOnPlot)), None)
        while data_set and not data_set.xname:
            data_set = next((obj for obj in self.objects if isinstance(obj, XYDataSetOnPlot)), None)
        if data_set and data_set.xname:
            return data_set.xname

        # if nothing is found, return empty string
        return ""

    @xlabel.setter
    def xlabel(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self.plot_info[lit.XLABEL] = new_label

    @property
    def ylabel(self):
        """The name of the y data, which will appear as the y axis label"""

        if self.plot_info[lit.YLABEL]:
            # use the user specified label if there is one
            return self.plot_info[lit.YLABEL]

        # else find the first data set and use the name of the data set as the label
        data_set = next((obj for obj in self.objects if isinstance(obj, XYDataSetOnPlot)), None)
        while data_set and not data_set.yname:
            data_set = next((obj for obj in self.objects if isinstance(obj, XYDataSetOnPlot)), None)
        if data_set and data_set.yname:
            return data_set.yname

        # if nothing is found, return empty string
        return ""

    @ylabel.setter
    def ylabel(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self.plot_info[lit.YLABEL] = new_label

    def plot(self, *args, **kwargs):
        """Adds a data set or function to plot"""
        self.objects.append(create_object_on_plot(*args, **kwargs))

    def show(self):
        """Draws the plot to output"""


def plot(*args, **kwargs) -> Plot:
    """Plots functions or data sets"""

    # first check the line which calls this function
    frame_stack = inspect.getouterframes(inspect.currentframe())
    code_context = frame_stack[1].code_context[0]
    is_return_value_assigned = re.match(r"\w+ *=", code_context) is not None

    # if this function call is assigned to a variable, create new Plot instance, else, the objects
    # passed into this function call will be drawn on the latest created Plot instance
    plot_obj = Plot() if is_return_value_assigned else Plot.current_plot_buffer
    Plot.current_plot_buffer = plot_obj

    # invoke the instance method of the Plot to add objects to the plot
    plot_obj.plot(*args, **kwargs)

    return plot_obj


def show(plot_obj=Plot.current_plot_buffer):
    """Draws the plot to output

    The QExPy plotting module keeps a buffer on the last plot being operated on. If no
    Plot instance is supplied to this function, the buffered plot will be shown.

    Args:
        plot_obj (Plot): the Plot instance to be shown.

    """
    plot_obj.show()
