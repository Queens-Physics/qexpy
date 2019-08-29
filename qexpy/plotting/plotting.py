"""This file contains function definitions for plotting"""

import re
import inspect
import matplotlib.pyplot as plt

from typing import List
from qexpy.utils.exceptions import IllegalArgumentError, UndefinedActionError
from qexpy.plotting.plotobjects import ObjectOnPlot, XYObjectOnPlot, XYDataSetOnPlot, \
    FunctionOnPlot, XYFitResultOnPlot, HistogramOnPlot, FitTarget, ObjectWithRange

import qexpy.utils as utils
import qexpy.fitting as ft
import qexpy.settings as sts
import qexpy.settings.literals as lit


class Plot:
    """The data structure used for a plot"""

    # points to the latest Plot instance that's created
    current_plot_buffer = None  # type: Plot

    def __init__(self):
        self._objects = []  # type: List[ObjectOnPlot]
        self._plot_info = {
            lit.TITLE: "",
            lit.XNAME: "",
            lit.YNAME: "",
            lit.XUNIT: "",
            lit.YUNIT: ""
        }
        self.plot_settings = {
            lit.LEGEND: False,
            lit.ERROR_BAR: False,
            lit.RESIDUALS: False,
            lit.PLOT_STYLE: lit.DEFAULT,
        }
        self._color_palette = ["C{}".format(idx) for idx in range(20)]
        self._xrange = ()
        self.main_ax = None
        self.res_ax = None

    def plot(self, *args, **kwargs):
        """Adds a data set or function to plot"""
        new_obj = self.__create_object_on_plot(*args, **kwargs)
        self._objects.append(new_obj)

    def hist(self, *args, **kwargs):
        """Adds a histogram to plot"""

        new_obj = HistogramOnPlot(*args, **kwargs)

        # add color to the histogram
        color = kwargs.pop("color", self._color_palette.pop(0))
        new_obj.color = color

        self._objects.append(new_obj)

        return new_obj.n, new_obj.bin_edges

    def fit(self, *args, **kwargs):
        """Plots a curve fit to the last data set added to the figure"""

        fit_targets = list(_obj for _obj in self._objects if isinstance(_obj, FitTarget))
        target = next(reversed(fit_targets), None)

        if not target:
            raise UndefinedActionError("There is no dataset in this plot to be fitted.")

        result = ft.fit(target.fit_target_dataset, *args, **kwargs)
        color = kwargs.pop(
            "color", target.color if isinstance(target, ObjectOnPlot) else "")
        obj = self.__create_object_on_plot(result, color=color, **kwargs)

        if isinstance(target, HistogramOnPlot) and isinstance(obj, XYFitResultOnPlot):
            target.kwargs["alpha"] = 0.8
            obj.func_on_plot.plot_kwargs["lw"] = 2

        self._objects.append(obj)
        return result

    def show(self):
        """Draws the plot to output"""

        self.__setup_figure_and_subplots()

        # set the xrange of functions to plot using the range of existing data sets
        xrange = self.xrange
        for obj in self._objects:
            if isinstance(obj, FunctionOnPlot) and not obj.xrange_specified:
                obj.xrange = xrange

        for obj in self._objects:
            obj.show(self.main_ax, self)

        self.main_ax.set_title(self.title)
        self.main_ax.set_xlabel(self.xlabel)
        self.main_ax.set_ylabel(self.ylabel)
        self.main_ax.grid()

        if self.res_ax:
            self.res_ax.set_xlabel(self.xlabel)
            self.res_ax.set_ylabel("residuals")
            self.res_ax.grid()

        if self.plot_settings[lit.LEGEND]:
            self.main_ax.legend()  # show legend if requested

        plt.show()

    def legend(self, new_setting=True):
        """Add or remove legend to plot"""
        self.plot_settings[lit.LEGEND] = new_setting

    def error_bars(self, new_setting=True):
        """Add or remove error bars from plot"""
        self.plot_settings[lit.ERROR_BAR] = new_setting

    def residuals(self, new_setting=True):
        """Add or remove subplot to show residuals"""
        self.plot_settings[lit.RESIDUALS] = new_setting

    @property
    def title(self) -> str:
        """str: The title of this plot, which will appear on top of the figure"""
        return self._plot_info[lit.TITLE]

    @title.setter
    def title(self, new_title: str):
        if not isinstance(new_title, str):
            raise TypeError("The new title is not a string!")
        self._plot_info[lit.TITLE] = new_title

    @property
    def xname(self) -> str:
        """str: The name of the x data, which will appear as x label"""
        if self._plot_info[lit.XNAME]:
            return self._plot_info[lit.XNAME]
        xy_objects = (obj for obj in self._objects if isinstance(obj, XYObjectOnPlot))
        return next((obj.xname for obj in xy_objects if obj.xname), "")

    @xname.setter
    def xname(self, name):
        if not isinstance(name, str):
            raise TypeError("Cannot set xname to \"{}\"".format(type(name)))
        self._plot_info[lit.XNAME] = name

    @property
    def yname(self) -> str:
        """str: The name of the y data, which will appear as y label"""
        if self._plot_info[lit.YNAME]:
            return self._plot_info[lit.YNAME]
        xy_objects = (obj for obj in self._objects if isinstance(obj, XYObjectOnPlot))
        return next((obj.yname for obj in xy_objects if obj.yname), "")

    @yname.setter
    def yname(self, name):
        if not isinstance(name, str):
            raise TypeError("Cannot set yname to \"{}\"".format(type(name)))
        self._plot_info[lit.YNAME] = name

    @property
    def xunit(self) -> str:
        """str: The unit of the x data, which will appear on the x label"""
        if self._plot_info[lit.XUNIT]:
            return self._plot_info[lit.XUNIT]
        xy_objects = (obj for obj in self._objects if isinstance(obj, XYObjectOnPlot))
        return next((obj.xunit for obj in xy_objects if obj.xunit), "")

    @xunit.setter
    def xunit(self, unit):
        if not isinstance(unit, str):
            raise TypeError("Cannot set xunit to \"{}\"".format(type(unit)))
        self._plot_info[lit.XUNIT] = unit

    @property
    def yunit(self) -> str:
        """str: The unit of the y data, which will appear on the y label"""
        if self._plot_info[lit.YUNIT]:
            return self._plot_info[lit.YUNIT]
        xy_objects = (obj for obj in self._objects if isinstance(obj, XYObjectOnPlot))
        return next((obj.yunit for obj in xy_objects if obj.yunit), "")

    @yunit.setter
    def yunit(self, unit):
        if not isinstance(unit, str):
            raise TypeError("Cannot set yunit to \"{}\"".format(type(unit)))
        self._plot_info[lit.YUNIT] = unit

    @property
    def xlabel(self) -> str:
        """str: The xlabel of the plot"""
        return self.xname + "[{}]".format(self.xunit) if self.xunit else ""

    @property
    def ylabel(self) -> str:
        """str: the ylabel of the plot"""
        return self.yname + "[{}]".format(self.yunit) if self.yunit else ""

    @property
    def xrange(self) -> (float, float):
        """(float, float): The x-value domain of this plot"""
        if not self._xrange:
            objs = list(obj for obj in self._objects if isinstance(obj, ObjectWithRange))
            low_bound = min(obj.xrange[0] for obj in objs if obj.xrange)
            high_bound = max(obj.xrange[1] for obj in objs if obj.xrange)
            return low_bound, high_bound
        return self._xrange

    @xrange.setter
    def xrange(self, new_range):
        utils.validate_xrange(new_range)
        self._xrange = new_range

    def __create_object_on_plot(self, *args, **kwargs) -> "ObjectOnPlot":
        """Factory method for creating ObjectOnPlot instances"""

        color = kwargs.pop("color", None)
        obj = None

        try:
            obj = XYFitResultOnPlot(*args, **kwargs)

            # find if the data set related to the fit is on the plot and get its color
            dataset = next((_obj for _obj in self._objects if obj.dataset == _obj), None)
            if not color:
                color = dataset.color if dataset else ""

        except IllegalArgumentError:
            pass

        try:
            obj = FunctionOnPlot(*args, **kwargs)
        except IllegalArgumentError:
            pass

        try:
            obj = XYDataSetOnPlot(*args, **kwargs)
        except IllegalArgumentError:
            pass

        # check if an object is actually created
        if not obj:
            raise IllegalArgumentError("Invalid combination of arguments for plotting.")

        # add color to the object and return
        obj.color = color if color else self._color_palette.pop(0)
        return obj

    def __setup_figure_and_subplots(self):
        """Create the mpl figure and subplots"""

        has_residuals = self.plot_settings[lit.RESIDUALS]

        width, height = sts.get_settings().plot_dimensions

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

        self.main_ax, self.res_ax = main_ax, res_ax


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


def __get_plot_obj():
    """Helper function that gets the appropriate Plot instance to draw on"""

    # initialize buffer if not initialized
    if not Plot.current_plot_buffer:
        Plot.current_plot_buffer = Plot()

    # first check the line which calls this function
    frame_stack = inspect.getouterframes(inspect.currentframe())
    code_context = frame_stack[2].code_context[0]
    is_return_value_assigned = re.match(r"[\w+ ,]*=", code_context) is not None

    # If this function call is assigned to a variable, create new Plot instance, else, the
    # objects passed into this function call will be drawn on the latest created Plot
    if is_return_value_assigned:
        plot_obj = Plot()
        Plot.current_plot_buffer = plot_obj
    else:
        plot_obj = Plot.current_plot_buffer

    return plot_obj
