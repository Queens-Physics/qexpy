"""This file contains function definitions for plotting"""

import inspect
import re
from typing import List
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt

from qexpy.utils.exceptions import InvalidArgumentTypeError, InvalidRequestError, IllegalArgumentError
import qexpy.data.datasets as dts
import qexpy.settings.literals as lit
import qexpy.fitting.fitting as ft
import qexpy.plotting.plot_utils as utils
import qexpy.plotting.plotobjects as plo

from qexpy.settings import get_settings


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
            lit.ERROR_BAR: False,
            lit.RESIDUALS: False,
            lit.PLOT_STYLE: lit.DEFAULT,
        }
        self._color_palette = ["C{}".format(idx) for idx in range(20)]
        self._xrange = ()

    @property
    def title(self) -> str:
        """str: The title of this plot, which will appear on top of the figure"""
        return self._plot_info[lit.TITLE]

    @title.setter
    def title(self, new_title: str):
        if not isinstance(new_title, str):
            raise InvalidArgumentTypeError("plot title", got=new_title, expected="string")
        self._plot_info[lit.TITLE] = new_title

    @property
    def xname(self) -> str:
        """str: The name of the x data, which will appear as x label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.XNAME, lit.XNAME, lit.YNAME)

    @xname.setter
    def xname(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self._plot_info[lit.XNAME] = new_label

    @property
    def yname(self) -> str:
        """str: The name of the y data, which will appear as the y axis label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.YNAME, lit.XNAME, lit.YNAME)

    @yname.setter
    def yname(self, new_label: str):
        if not isinstance(new_label, str):
            raise InvalidArgumentTypeError("plot label", got=new_label, expected="string")
        self._plot_info[lit.YNAME] = new_label

    @property
    def xunit(self) -> str:
        """str: The name of the x data, which will appear as x label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.XUNIT, lit.XUNIT, lit.YUNIT)

    @xunit.setter
    def xunit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise InvalidArgumentTypeError("plot unit", got=new_unit, expected="string")
        self._plot_info[lit.XUNIT] = new_unit

    @property
    def yunit(self) -> str:
        """str: The name of the x data, which will appear as x label"""
        return self.__get_plot_info_or_extract_from_datasets(lit.YUNIT, lit.XUNIT, lit.YUNIT)

    @yunit.setter
    def yunit(self, new_unit: str):
        if not isinstance(new_unit, str):
            raise InvalidArgumentTypeError("plot unit", got=new_unit, expected="string")
        self._plot_info[lit.YUNIT] = new_unit

    @property
    def xlabel(self) -> str:
        """str: The xlabel of the plot"""
        return self.xname + "[{}]".format(self.xunit) if self.xunit else ""

    @property
    def ylabel(self) -> str:
        """str: the ylabel of the plot"""
        return self.yname + "[{}]".format(self.yunit) if self.yunit else ""

    @property
    def xrange(self) -> tuple:
        """tuple: The x-value domain of this plot"""
        if not self._xrange:
            low_bound = min(min(obj.dataset.xvalues) for obj in self._objects if isinstance(obj, plo.XYDataSetOnPlot))
            high_bound = max(max(obj.dataset.xvalues) for obj in self._objects if isinstance(obj, plo.XYDataSetOnPlot))
            return low_bound, high_bound
        return self._xrange

    @xrange.setter
    def xrange(self, new_range):
        utils.validate_xrange(new_range)
        self._xrange = new_range

    def plot(self, *args, **kwargs):
        """Adds a data set or function to plot"""
        new_obj = self.__create_object_on_plot(*args, **kwargs)
        self._objects.append(new_obj)

    def hist(self, *args, **kwargs) -> tuple:
        """Adds a histogram to plot"""

        # first add the object to plot
        obj = self.__create_histogram_on_plot(*args, **kwargs)
        self._objects.append(obj)

        # use numpy to get the results of the histogram without plotting
        values, bin_edges = np.histogram(obj.values, **obj.kwargs)

        return values, bin_edges

    def fit(self, *args, **kwargs):
        """Plots a curve fit to the last data set added to the figure"""
        obj = next((_obj for _obj in reversed(self._objects) if isinstance(_obj, plo.XYDataSetOnPlot)), None)
        if obj:
            result = ft.fit(obj.dataset, *args, **kwargs)
            color = kwargs.pop("color", obj.color)
            self._objects.append(self.__create_object_on_plot(result, color=color, **kwargs))
        else:
            raise InvalidRequestError("There is not data set in this plot to be fitted.")

        return result

    def legend(self, new_setting=True):
        """Add or remove legend to plot"""
        self._plot_settings[lit.LEGEND] = new_setting

    def error_bars(self, new_setting=True):
        """Add or remove error bars from plot"""
        self._plot_settings[lit.ERROR_BAR] = new_setting

    def residuals(self, new_setting=True):
        """Add or remove subplot to show residuals"""
        self._plot_settings[lit.RESIDUALS] = new_setting

    def show(self):
        """Draws the plot to output"""

        main_ax, res_ax = self.__setup_figure_and_subplots()

        # set the xrange of functions to plot using the range of existing data sets
        for obj in self._objects:
            if isinstance(obj, plo.FunctionOnPlot) and not obj.xrange_specified:
                obj.xrange = self.xrange

        for obj in self._objects:
            _draw_object_on_plot(main_ax, res_ax, obj, errorbar=self._plot_settings[lit.ERROR_BAR])

        main_ax.set_title(self.title)
        main_ax.set_xlabel(self.xlabel)
        main_ax.set_ylabel(self.ylabel)
        main_ax.grid()

        if res_ax:
            res_ax.set_xlabel(self.xlabel)
            res_ax.set_ylabel("residuals")
            res_ax.grid()

        if self._plot_settings[lit.LEGEND]:
            main_ax.legend()  # show legend if requested

        plt.show()

    def __get_plot_info_or_extract_from_datasets(self, axis: str, reference_x_axis: str, other_axis: str):
        """Helper method to get info from datasets for plot"""

        if self._plot_info[axis]:
            # use the user specified label if there is one
            return self._plot_info[axis]

        # else find the first data set and use the name of the data set as the label
        data_sets_in_plot = (obj.dataset for obj in self._objects if isinstance(obj, plo.XYDataSetOnPlot))

        data_set = next(data_sets_in_plot, None)
        info_str = getattr(data_set, axis if axis == reference_x_axis else other_axis) if data_set else ""
        while data_set and not info_str:
            data_set = next(data_sets_in_plot, None)
            info_str = getattr(data_set, axis if axis == reference_x_axis else other_axis) if data_set else ""

        return info_str if data_set and info_str else ""

    def __create_object_on_plot(self, *args, **kwargs) -> "plo.ObjectOnPlot":
        """Factory method that chooses the appropriate ObjectOnPlot to create with the arguments"""

        color = kwargs.pop("color", None)

        obj = _try_fit_result_on_plot(*args, **kwargs)
        if obj:
            dataset = next((_obj for _obj in self._objects if obj.fit_result.dataset == _obj), None)
            obj.color = color if color else dataset.color if dataset else self._color_palette.pop(0)
            return obj

        try_funcs = [_try_function_on_plot, _try_data_set_on_plot, _try_xdata_and_y_data]
        obj = next((_obj for _obj in (test(*args, **kwargs) for test in try_funcs) if _obj), None)
        if obj:
            obj.color = color if color else self._color_palette.pop(0)
            return obj

        raise IllegalArgumentError("Invalid combination of arguments for creating an object on plot.")

    def __create_histogram_on_plot(self, *args, **kwargs) -> "plo.HistogramOnPlot":
        """Factory method that creates a histogram object to be plotted"""

        color = kwargs.pop("color", self._color_palette.pop(0))
        obj = _try_histogram_on_plot(*args, color=color, **kwargs)

        if not obj:
            raise IllegalArgumentError("Invalid combination of arguments for creating a histogram on plot.")

        return obj

    def __setup_figure_and_subplots(self) -> (plt.Axes, plt.Axes):
        """Create the mpl figure and subplots"""

        has_residuals = self._plot_settings[lit.RESIDUALS]

        width, height = get_settings().plot_dimensions

        if has_residuals:
            height = height * 1.5

        figure = plt.figure(figsize=(width, height), constrained_layout=True)

        if has_residuals:
            gs = figure.add_gridspec(3, 1)
            main_ax = figure.add_subplot(gs[:-1, :])
            res_ax = figure.add_subplot(gs[-1:, :])
        else:
            main_ax = figure.add_subplot()
            res_ax = None

        return main_ax, res_ax


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


def get_plot():
    """Gets the current plot buffer"""
    return Plot.current_plot_buffer


def new_plot():
    """Clears the current plot buffer and start a new one"""
    Plot.current_plot_buffer = Plot()


def _draw_object_on_plot(main_ax: plt.Axes, res_ax: plt.Axes, obj: plo.ObjectOnPlot, errorbar):
    """Helper method that draws an ObjectOnPlot to the output"""

    if isinstance(obj, plo.HistogramOnPlot):
        main_ax.hist(obj.values, **obj.kwargs)
    elif isinstance(obj, plo.XYObjectOnPlot) and not errorbar:
        kwargs = utils.extract_plt_plot_arguments(obj.kwargs)
        main_ax.plot(obj.xvalues_to_plot, obj.yvalues_to_plot, obj.fmt, color=obj.color, label=obj.label, **kwargs)
    elif isinstance(obj, plo.XYDataSetOnPlot):
        kwargs = utils.extract_plt_errorbar_arguments(obj.kwargs)
        main_ax.errorbar(obj.xvalues_to_plot, obj.yvalues_to_plot, obj.yerr_to_plot, obj.xerr_to_plot,
                         fmt=obj.fmt, color=obj.color, label=obj.label, **kwargs)
    elif isinstance(obj, plo.FunctionOnPlot):
        xvalues = obj.xvalues_to_plot
        yvalues = obj.yvalues_to_plot
        kwargs = utils.extract_plt_plot_arguments(obj.kwargs)
        main_ax.plot(xvalues, yvalues, obj.fmt, color=obj.color, label=obj.label, **kwargs)
        yerr = obj.yerr_to_plot
        if yerr.size > 0:
            _plot_error_band(main_ax, xvalues, yvalues, yerr, obj.color)
    elif isinstance(obj, plo.XYFitResultOnPlot):
        _draw_object_on_plot(main_ax, res_ax, obj.func_on_plot, errorbar)
        if res_ax:
            _draw_object_on_plot(res_ax, main_ax, obj.residuals_on_plot, errorbar)


def _plot_error_band(ax: plt.Axes, xvalues, yvalues, yerr, color):
    """Adds an error band to plot around a set of x-y values"""
    max_vals = yvalues + yerr
    min_vals = yvalues - yerr
    ax.fill_between(xvalues, min_vals, max_vals, interpolate=True, edgecolor='none', color=color, alpha=0.3, zorder=0)


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


def _try_function_on_plot(*args, **kwargs):
    """Helper function which tries to create a FunctionOnPlot with the provided arguments"""

    func = kwargs.pop("func", args[0] if args and callable(args[0]) else None)
    # if there's a second argument, assume that is is the fmt string
    fmt = kwargs.pop("fmt", args[1] if len(args) > 1 and isinstance(args[1], str) else "")

    return plo.FunctionOnPlot(func, fmt=fmt, **kwargs) if func else None


def _try_fit_result_on_plot(*args, **kwargs) -> "plo.XYFitResultOnPlot":
    """Helper function that tries to plot an XYFitResult to plot"""

    fit = args[0] if args and isinstance(args[0], ft.XYFitResult) else None
    return plo.XYFitResultOnPlot(fit, **kwargs) if fit else None


def _try_data_set_on_plot(*args, **kwargs) -> "plo.XYDataSetOnPlot":
    """Helper function which tries to create a XYDataSetOnPlot with an existing XYDataSet"""

    dataset = kwargs.pop("dataset", args[0] if args and isinstance(args[0], dts.XYDataSet) else None)
    fmt = kwargs.pop("fmt", args[1] if len(args) > 1 and isinstance(args[1], str) else "")

    return plo.XYDataSetOnPlot.from_xy_dataset(dataset, fmt=fmt, **kwargs) if dataset else None


def _try_xdata_and_y_data(*args, **kwargs) -> "plo.XYDataSetOnPlot":
    """Helper function which tries to create an XYDataSetOnPlot with xdata and ydata"""

    xdata = kwargs.pop("xdata", args[0] if len(args) >= 2 and isinstance(args[0], Iterable) else np.empty(0))
    ydata = kwargs.pop("ydata", args[1] if len(args) >= 2 and isinstance(args[1], Iterable) else np.empty(0))
    fmt = kwargs.pop("fmt", args[2] if len(args) >= 3 and isinstance(args[2], str) else "")

    # wrapping data in numpy array objects
    if isinstance(xdata, list):
        xdata = np.asarray(xdata)
    if isinstance(ydata, list):
        ydata = np.asarray(ydata)

    return plo.XYDataSetOnPlot(xdata, ydata, fmt=fmt, **kwargs) if xdata.size and ydata.size else None


def _try_histogram_on_plot(*args, **kwargs):
    """Helper function which tries to create a HistogramOnPlot with the arguments provided"""

    data = kwargs.pop("data", args[0] if args and isinstance(args[0], (np.ndarray, list)) else np.empty(0))

    if isinstance(data, dts.ExperimentalValueArray):
        return plo.HistogramOnPlot.from_value_array(data, **kwargs)
    if (isinstance(data, list) and data) or (isinstance(data, np.ndarray) and data.size):
        return plo.HistogramOnPlot(data, *args, **kwargs)

    return None
