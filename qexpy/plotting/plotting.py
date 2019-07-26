"""This file contains function definitions for plotting"""

import inspect
import re

from typing import List

from qexpy.utils.exceptions import InvalidArgumentTypeError
import qexpy.settings.literals as lit

import qexpy.plotting.plot_utils as utils
from qexpy.plotting.plotobjects import ObjectOnPlot, XYDataSetOnPlot, _create_object_on_plot


class Plot:
    """The data structure used for a plot"""

    # points to the latest Plot instance that's created
    current_plot_buffer = None  # type: Plot

    def __init__(self):
        self._objects = []  # type: List[ObjectOnPlot]
        self._plot_info = {
            lit.TITLE: "",
            lit.XLABEL: "",
            lit.YLABEL: "",
            lit.XUNIT: "",
            lit.YUNIT: ""
        }
        self._xrange = ()

    @property
    def title(self):
        """The title of this plot, which will appear on top of the figure"""
        return self._plot_info[lit.TITLE]

    @title.setter
    def title(self, new_title: str):
        if not isinstance(new_title, str):
            raise InvalidArgumentTypeError("plot title", got=new_title, expected="string")
        self._plot_info[lit.TITLE] = new_title

    @property
    def xlabel(self):
        """The name of the x data, which will appear as x label"""

        if self._plot_info[lit.XLABEL]:
            # use the user specified label if there is one
            return self._plot_info[lit.XLABEL]

        # else find the first data set and use the name of the data set as the label
        data_set = next((obj for obj in self._objects if isinstance(obj, XYDataSetOnPlot)), None)
        while data_set and not data_set.xname:
            data_set = next((obj for obj in self._objects if isinstance(obj, XYDataSetOnPlot)), None)
        if data_set and data_set.xname:
            return data_set.xname

        # if nothing is found, return empty string
        return ""

    @xlabel.setter
    def xlabel(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self._plot_info[lit.XLABEL] = new_label

    @property
    def ylabel(self):
        """The name of the y data, which will appear as the y axis label"""

        if self._plot_info[lit.YLABEL]:
            # use the user specified label if there is one
            return self._plot_info[lit.YLABEL]

        # else find the first data set and use the name of the data set as the label
        data_set = next((obj for obj in self._objects if isinstance(obj, XYDataSetOnPlot)), None)
        while data_set and not data_set.yname:
            data_set = next((obj for obj in self._objects if isinstance(obj, XYDataSetOnPlot)), None)
        if data_set and data_set.yname:
            return data_set.yname

        # if nothing is found, return empty string
        return ""

    @property
    def xrange(self):
        """The range of this plot"""
        if not self._xrange:
            low_bound = min(obj.xrange[0] for obj in self._objects)
            high_bound = max(obj.xrange[1] for obj in self._objects)
            return low_bound, high_bound
        return self._xrange

    @xrange.setter
    def xrange(self, new_range):
        utils._validate_xrange(new_range)
        self._xrange = new_range

    @ylabel.setter
    def ylabel(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self._plot_info[lit.YLABEL] = new_label

    def plot(self, *args, **kwargs):
        """Adds a data set or function to plot"""
        self._objects.append(_create_object_on_plot(*args, **kwargs))

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
