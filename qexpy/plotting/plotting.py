"""This file contains function definitions for plotting"""

import inspect
import re
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from qexpy.utils.exceptions import InvalidArgumentTypeError, InvalidRequestError
import qexpy.settings.literals as lit
import qexpy.fitting.fitting as fitting
import qexpy.plotting.plot_utils as utils
import qexpy.plotting.plotobjects as plo


class Plot:
    """The data structure used for a plot"""

    # points to the latest Plot instance that's created
    current_plot_buffer = None  # type: Plot

    def __init__(self):
        self._objects = []  # type: List[plo.ObjectOnPlot]
        self._plot_info = {
            lit.TITLE: "",
            lit.XNAME: "",
            lit.YNAME: "",
            lit.XUNIT: "",
            lit.YUNIT: ""
        }
        self._plot_settings = {
            lit.LEGEND: False,
            lit.ERROR_BAR: False
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
    def xname(self):
        """The name of the x data, which will appear as x label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.XNAME, lit.XNAME, lit.YNAME)

    @xname.setter
    def xname(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self._plot_info[lit.XNAME] = new_label

    @property
    def yname(self):
        """The name of the y data, which will appear as the y axis label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.YNAME, lit.XNAME, lit.YNAME)

    @yname.setter
    def yname(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self._plot_info[lit.YNAME] = new_label

    @property
    def xunit(self):
        """The name of the x data, which will appear as x label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.XUNIT, lit.XUNIT, lit.YUNIT)

    @xunit.setter
    def xunit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise InvalidArgumentTypeError("plot unit", got=new_unit, expected="string")
        self._plot_info[lit.XUNIT] = new_unit

    @property
    def yunit(self):
        """The name of the x data, which will appear as x label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.YUNIT, lit.XUNIT, lit.YUNIT)

    @yunit.setter
    def yunit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise InvalidArgumentTypeError("plot unit", got=new_unit, expected="string")
        self._plot_info[lit.YUNIT] = new_unit

    @property
    def xlabel(self):
        return self.xname + "[{}]".format(self.xunit) if self.xunit else ""

    @property
    def ylabel(self):
        return self.yname + "[{}]".format(self.yunit) if self.yunit else ""

    @property
    def xrange(self):
        """The range of this plot"""
        if not self._xrange:
            low_bound = min(min(obj.xvalues) for obj in self._objects if isinstance(obj, plo.XYDataSetOnPlot))
            high_bound = max(max(obj.xvalues) for obj in self._objects if isinstance(obj, plo.XYDataSetOnPlot))
            return low_bound, high_bound
        return self._xrange

    @xrange.setter
    def xrange(self, new_range):
        utils.validate_xrange(new_range)
        self._xrange = new_range

    def plot(self, *args, **kwargs):
        """Adds a data set or function to plot"""
        self._objects.append(plo.ObjectOnPlot.create_xy_object_on_plot(*args, **kwargs))

    def hist(self, *args, **kwargs):
        """Adds a histogram to plot"""

        # first add the object to plot
        obj = plo.ObjectOnPlot.create_histogram_on_plot(*args, **kwargs)
        self._objects.append(obj)

        # use numpy to get the results of the histogram without plotting
        values, bin_edges = np.histogram(obj.values, **obj.kwargs)

        return values, bin_edges

    def fit(self, *args, **kwargs):
        """Plots a curve fit to the last data set added to the figure"""
        dataset = next((obj for obj in reversed(self._objects) if isinstance(obj, plo.XYDataSetOnPlot)), None)
        if dataset:
            result = fitting.fit(dataset, *args, **kwargs)
            self._objects.append(plo.ObjectOnPlot.create_xy_object_on_plot(result.fit_function, **kwargs))
        else:
            raise InvalidRequestError("There is not data set in this plot to be fitted.")

        return result

    def legend(self, new_setting=True):
        """Add or remove legend to plot"""
        self._plot_settings[lit.LEGEND] = new_setting

    def error_bars(self, new_setting=True):
        """Add or remove error bars from plot"""
        self._plot_settings[lit.ERROR_BAR] = new_setting

    def show(self):
        """Draws the plot to output"""

        # set the xrange of functions to plot using the range of existing data sets
        for obj in self._objects:
            if isinstance(obj, plo.FunctionOnPlot) and not obj.xrange_specified:
                obj.xrange = self.xrange

        for obj in self._objects:
            self.__draw_object_on_plot(obj)

        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        if self._plot_settings[lit.LEGEND]:
            plt.legend()  # show legend if requested

        plt.show()

    def __get_plot_info_or_extract_from_datasets(self, axis: str, reference_x_axis: str, other_axis: str):
        """Helper method to get info from datasets for plot"""

        if self._plot_info[axis]:
            # use the user specified label if there is one
            return self._plot_info[axis]

        # else find the first data set and use the name of the data set as the label
        data_sets_in_plot = (obj for obj in self._objects if isinstance(obj, plo.XYDataSetOnPlot))

        data_set = next(data_sets_in_plot, None)
        info_str = getattr(data_set, axis if axis == reference_x_axis else other_axis) if data_set else ""
        while data_set and not info_str:
            data_set = next(data_sets_in_plot, None)
            info_str = getattr(data_set, axis if axis == reference_x_axis else other_axis) if data_set else ""

        if data_set and info_str:
            return info_str

        return ""

    def __draw_object_on_plot(self, obj: plo.ObjectOnPlot):
        """Helper method that draws an ObjectOnPlot to the output"""

        if isinstance(obj, plo.HistogramOnPlot):
            plt.hist(obj.values, **obj.kwargs)
        elif isinstance(obj, plo.XYObjectOnPlot) and not self._plot_settings[lit.ERROR_BAR]:
            kwargs = utils.extract_plt_plot_arguments(obj.kwargs)
            plt.plot(obj.xvalues_to_plot, obj.yvalues_to_plot, obj.fmt, label=obj.label, **kwargs)
        elif isinstance(obj, plo.XYDataSetOnPlot):
            kwargs = utils.extract_plt_errorbar_arguments(obj.kwargs)
            plt.errorbar(obj.xvalues_to_plot, obj.yvalues_to_plot, obj.yerr_to_plot, obj.xerr_to_plot,
                         fmt=obj.fmt, label=obj.label, **kwargs)
        elif isinstance(obj, plo.FunctionOnPlot):
            xvalues = obj.xvalues_to_plot
            yvalues = obj.yvalues_to_plot
            kwargs = utils.extract_plt_plot_arguments(obj.kwargs)
            plt.plot(xvalues, yvalues, obj.fmt, label=obj.label, **kwargs)
            yerr = obj.yerr_to_plot
            if yerr.size > 0:
                _plot_error_band(xvalues, yvalues, yerr)


def plot(*args, **kwargs) -> Plot:
    """Plots functions or data sets"""

    plot_obj = __get_plot_obj()

    # invoke the instance method of the Plot to add objects to the plot
    plot_obj.plot(*args, **kwargs)

    return plot_obj


def hist(*args, **kwargs) -> tuple:
    """Plots a histogram with a data set"""

    plot_obj = __get_plot_obj()

    # invoke the instance method of the Plot to add objects to the plot
    values, bin_edges = plot_obj.hist(*args, **kwargs)

    return values, bin_edges, plot_obj


def show(plot_obj=None):
    """Draws the plot to output

    The QExPy plotting module keeps a buffer on the last plot being operated on. If no
    Plot instance is supplied to this function, the buffered plot will be shown.

    Args:
        plot_obj (Plot): the Plot instance to be shown.

    """
    if not plot_obj:
        plot_obj = Plot.current_plot_buffer
    plot_obj.show()


def new_plot():
    """Clears the current plot buffer and start a new one"""
    Plot.current_plot_buffer = Plot()


def __get_plot_obj():
    """Helper function that gets the appropriate Plot instance to draw on"""

    # initialize buffer if not initialized
    if not Plot.current_plot_buffer:
        Plot.current_plot_buffer = Plot()

    # first check the line which calls this function
    frame_stack = inspect.getouterframes(inspect.currentframe())
    code_context = frame_stack[2].code_context[0]
    is_return_value_assigned = re.match(r"\w+ *=", code_context) is not None

    # if this function call is assigned to a variable, create new Plot instance, else, the objects
    # passed into this function call will be drawn on the latest created Plot instance
    if is_return_value_assigned:
        plot_obj = Plot()
        Plot.current_plot_buffer = plot_obj
    else:
        plot_obj = Plot.current_plot_buffer

    return plot_obj


def _plot_error_band(xvalues, yvalues, yerr):
    max_values = yvalues + yerr
    min_values = yvalues - yerr
    plt.fill_between(xvalues, min_values, max_values, interpolate=True, edgecolor='none', alpha=0.3, zorder=0)
