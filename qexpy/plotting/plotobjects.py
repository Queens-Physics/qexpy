"""Contains definitions for objects to be drawn on plot"""

import numpy as np
import inspect

from abc import ABC, abstractmethod
from matplotlib.pyplot import Axes
from qexpy.utils.exceptions import IllegalArgumentError, UndefinedActionError
from qexpy.fitting.fitting import XYFitResult

import qexpy.data.data as dt
import qexpy.utils as utils
import qexpy.settings.settings as sts
import qexpy.settings.literals as lit
import qexpy.data.datasets as dts

from . import plotting as plt  # pylint: disable=cyclic-import,unused-import


class ObjectOnPlot(ABC):
    """A container for anything to be plotted"""

    def __init__(self, *args, **kwargs):
        """Constructor for ObjectOnPlot"""

        # process format string
        fmt = kwargs.pop("fmt", args[0] if args and isinstance(args[0], str) else None)
        if fmt and not isinstance(fmt, str):
            raise TypeError("The fmt provided is not a string!")
        self._fmt = fmt

        # process color
        color = kwargs.pop("color", None)
        if color and not isinstance(color, str):
            raise TypeError("The color provided is not a string!")
        self._color = color

        # add the rest to the object
        label = kwargs.pop("label", "")
        if label and not isinstance(label, str):
            raise TypeError("The label of this plot object is not a string!")
        self.label = label

    @property
    def fmt(self):
        """str: The format string to be used in PyPlot"""
        return self._fmt

    @property
    def color(self):
        """str: The color of the object"""
        return self._color

    @color.setter
    def color(self, new_color: str):
        if not new_color:
            return
        if not isinstance(new_color, str):
            raise TypeError("The color has to be a string.")
        self._color = new_color

    @abstractmethod
    def show(self, ax: Axes, plot: "plt.Plot"):
        """Draw the object itself onto the given axes"""
        raise NotImplementedError


class FitTarget(ABC):  # pylint: disable=too-few-public-methods
    """Interface for anything to which a fit can be applied"""

    @property
    @abstractmethod
    def fit_target_dataset(self):
        """dts.XYDataSet: The target dataset instance to apply the fit to"""
        raise NotImplementedError


class ObjectWithRange(ABC):  # pylint: disable=too-few-public-methods
    """Interface for anything with an xrange"""

    @property
    @abstractmethod
    def xrange(self):
        """tuple: The xrange of the object"""
        raise NotImplementedError


class XYObjectOnPlot(ObjectOnPlot, ObjectWithRange):
    """A container for objects with x and y values to be drawn on a plot"""

    def __init__(self, *args, **kwargs):
        """Constructor for XYObjectOnPlot"""

        xrange = kwargs.pop("xrange", ())
        if xrange:
            utils.validate_xrange(xrange)
        self._xrange = xrange

        xname = kwargs.pop("xname", "")
        if not isinstance(xname, str):
            raise TypeError("The xname provided is not a string!")
        self._xname = xname

        yname = kwargs.pop("yname", "")
        if not isinstance(yname, str):
            raise TypeError("The yname provided is not a string!")
        self._yname = yname

        xunit = kwargs.pop("xunit", "")
        if not isinstance(xunit, str):
            raise TypeError("The xunit provided is not a string!")
        self._xunit = xunit

        yunit = kwargs.pop("yunit", "")
        if not isinstance(yname, str):
            raise TypeError("The yunit provided is not a string!")
        self._yunit = yunit

        # save the plot kwargs
        self.plot_kwargs = {k: v for k, v in kwargs.items() if k in PLOT_VALID_KWARGS}
        self.err_kwargs = {k: v for k, v in kwargs.items() if k in ERRORBAR_VALID_KWARGS}

        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def xvalues(self):
        """np.ndarray: The array of x-values to be plotted"""
        raise NotImplementedError

    @property
    @abstractmethod
    def yvalues(self):
        """np.ndarray: The array of y-values to be plotted"""
        raise NotImplementedError

    @property
    def xrange(self):
        """tuple: The range of values to be plotted"""
        return self._xrange

    @xrange.setter
    def xrange(self, new_range: tuple):
        if new_range:
            utils.validate_xrange(new_range)
        self._xrange = new_range

    @property
    def xname(self):
        """str: The name of the x-axis"""
        return self._xname

    @property
    def xunit(self):
        """str: The unit of the x-axis"""
        return self._xunit

    @property
    def yname(self):
        """str: The name of the x-axis"""
        return self._yname

    @property
    def yunit(self):
        """str: The unit of the x-axis"""
        return self._yunit


class XYDataSetOnPlot(XYObjectOnPlot, FitTarget):
    """A wrapper for an XYDataSet to be plotted"""

    def __init__(self, *args, **kwargs):

        # set the data set object
        if args and isinstance(args[0], dts.XYDataSet):
            self.dataset = args[0]
            fmt = kwargs.pop("fmt", args[1] if len(args) >= 2 else "")
        else:
            self.dataset = dts.XYDataSet(*args, **kwargs)
            fmt = kwargs.pop("fmt", "")

        label = kwargs.pop("label", self.dataset.name)
        fmt = fmt if fmt else "o"

        # call super constructors
        XYObjectOnPlot.__init__(self, label=label, fmt=fmt, **kwargs)

    def show(self, ax: Axes, plot: "plt.Plot"):
        if not plot.plot_settings[lit.ERROR_BAR]:
            ax.plot(
                self.xvalues, self.yvalues, self.fmt, color=self.color,
                label=self.label, **self.plot_kwargs)
        else:
            ax.errorbar(
                self.xvalues, self.yvalues, self.yerr, self.xerr, fmt=self.fmt,
                color=self.color, label=self.label, **self.plot_kwargs, **self.err_kwargs)

    @property
    def xrange(self):
        if not self._xrange:
            return min(self.dataset.xvalues), max(self.dataset.xvalues)
        return self._xrange

    @property
    def xvalues(self):
        if self._xrange:
            return self.dataset.xvalues[self.__get_indices_from_xrange()]
        return self.dataset.xvalues

    @property
    def yvalues(self):
        if self._xrange:
            return self.dataset.yvalues[self.__get_indices_from_xrange()]
        return self.dataset.yvalues

    @property
    def xerr(self):
        """np.ndarray: the array of x-value uncertainties to show up on plot"""
        if self._xrange:
            return self.dataset.xerr[self.__get_indices_from_xrange()]
        return self.dataset.xerr

    @property
    def yerr(self):
        """np.ndarray: the array of y-value uncertainties to show up on plot"""
        if self._xrange:
            return self.dataset.yerr[self.__get_indices_from_xrange()]
        return self.dataset.yerr

    @property
    def xname(self):
        return self.dataset.xname

    @property
    def xunit(self):
        return self.dataset.xunit

    @property
    def yname(self):
        return self.dataset.yname

    @property
    def yunit(self):
        return self.dataset.yunit

    @property
    def fit_target_dataset(self) -> dts.XYDataSet:
        return self.dataset

    def __get_indices_from_xrange(self):
        low, high = self._xrange
        return (low <= self.dataset.xvalues) & (self.dataset.xvalues < high)


class FunctionOnPlot(XYObjectOnPlot):
    """This is the wrapper for a function to be plotted"""

    def __init__(self, *args, **kwargs):
        """Constructor for FunctionOnPlot"""

        func = args[0] if args else None

        # check input
        if not callable(func):
            raise IllegalArgumentError("The function provided is not a callable object!")

        # this checks if the xrange of plot is specified by user or auto-generated
        self.xrange_specified = "xrange" in kwargs

        self.pars = kwargs.pop("pars", [])

        self.error_method = kwargs.pop("error_method", None)

        self._ydata = None  # buffer for calculated y data

        parameters = inspect.signature(func).parameters
        if len(parameters) > 1 and not self.pars:
            raise ValueError(
                "For a function with parameters, a list of parameters has to be supplied.")

        if len(parameters) == 1:
            self.func = func
        elif len(parameters) > 1:
            self.func = lambda x: func(x, *self.pars)  # pylint:disable=not-callable
        else:
            raise ValueError("The function supplied does not have an x-variable.")

        XYObjectOnPlot.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def show(self, ax: Axes, plot: "plt.Plot"):
        xvalues = self.xvalues
        yvalues = self.yvalues
        ax.plot(
            xvalues, yvalues, self.fmt if self.fmt else "-", color=self.color,
            label=self.label, **self.plot_kwargs)
        yerr = self.yerr
        if yerr.size > 0 and plot.plot_settings[lit.ERROR_BAR]:
            max_vals = yvalues + yerr
            min_vals = yvalues - yerr
            ax.fill_between(
                xvalues, min_vals, max_vals, edgecolor='none', color=self.color,
                alpha=0.3, interpolate=True, zorder=0)

    @property
    def xrange(self):
        return self._xrange

    @xrange.setter
    def xrange(self, new_range: tuple):
        if new_range:
            utils.validate_xrange(new_range)
        self._xrange = new_range
        self._ydata = None  # clear y data since it would need to be re-calculated

    @property
    def xvalues(self):
        if not self.xrange:
            raise UndefinedActionError("The domain of this function cannot be found.")
        return np.linspace(self.xrange[0], self.xrange[1], 100)

    @property
    def ydata(self):
        """The raw y data of the function"""
        if self._ydata:
            return self._ydata
        if not self.xrange:
            raise UndefinedActionError("The domain of this function cannot be found.")
        result = self.func(self.xvalues)
        derived_values = (res for res in result if isinstance(res, dt.DerivedValue))
        if self.error_method:
            for value in derived_values:
                value.error_method = self.error_method
        return result

    @property
    @sts.use_mc_sample_size(10000)
    def yvalues(self):
        simplified_result = list(
            res.value if isinstance(res, dt.DerivedValue) else res for res in self.ydata)
        return np.asarray(simplified_result)

    @property
    @sts.use_mc_sample_size(10000)
    def yerr(self):
        """The array of y-value uncertainties to show up on plot"""
        errors = np.asarray(list(
            res.error if isinstance(res, dt.DerivedValue) else 0 for res in self.ydata))
        return errors if errors.size else np.empty(0)


class XYFitResultOnPlot(ObjectOnPlot):
    """Wrapper for an XYFitResult to be plotted"""

    def __init__(self, *args, **kwargs):
        """Constructor for an XYFitResultOnPlot"""

        result = args[0] if args else None

        # check input
        if not isinstance(result, XYFitResult):
            raise IllegalArgumentError("The fit result is not an XYFitResult instance")

        # initialize object
        ObjectOnPlot.__init__(self, **kwargs)
        self.fit_result = result

        xrange = result.xrange if result.xrange else (
            min(result.dataset.xvalues), max(result.dataset.xvalues))

        self.func_on_plot = FunctionOnPlot(
            result.fit_function, xrange=xrange, error_method=lit.MONTE_CARLO, **kwargs)
        self.residuals_on_plot = XYDataSetOnPlot(
            result.dataset.xdata, result.residuals, **kwargs)

    # pylint: disable=protected-access
    def show(self, ax: Axes, plot: "plt.Plot"):
        if not self.color:
            datasets = (obj for obj in plot._objects if isinstance(obj, XYDataSetOnPlot))
            color = next((
                obj.color for obj in datasets if obj.dataset == self.fit_result.dataset), "")
            self.color = color if color else plot._color_palette.pop(0)
        self.func_on_plot.show(ax, plot)
        if plot.res_ax:
            self.residuals_on_plot.show(plot.res_ax, plot)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, new_color: str):
        if not new_color:
            return
        if not isinstance(new_color, str):
            raise TypeError("The color has to be a string.")
        self._color = new_color
        self.func_on_plot.color = new_color
        self.residuals_on_plot.color = new_color

    @property
    def dataset(self):
        """dts.XYDataSet: The dataset that the fit is associated with"""
        return self.fit_result.dataset


class HistogramOnPlot(ObjectOnPlot, FitTarget, ObjectWithRange):
    """Represents a histogram to be drawn on a plot"""

    def __init__(self, *args, **kwargs):
        """Constructor for histogram on plots"""

        ObjectOnPlot.__init__(self, **kwargs)

        if args and isinstance(args[0], dts.ExperimentalValueArray):
            self.samples = args[0]
        else:
            self.samples = dts.ExperimentalValueArray(*args, **kwargs)

        self.kwargs = {k: v for k, v in kwargs.items() if k in HIST_VALID_KWARGS}

        hist_kwargs = {k: v for k, v in kwargs.items() if k in NP_HIST_VALID_KWARGS}
        self.n, self.bin_edges = np.histogram(self.samples.values, **hist_kwargs)

        self._xrange = self.bin_edges[0], self.bin_edges[-1]

    def show(self, ax: Axes, plot: "plt.Plot"):
        ax.hist(self.sample_values, **self.kwargs)

    @property
    def sample_values(self):
        """np.ndarray: The values of the samples in this histogram"""
        return self.samples.values

    @property
    def fit_target_dataset(self) -> dts.XYDataSet:
        bins = self.bin_edges
        xvalues = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        return dts.XYDataSet(xvalues, self.n, name="histogram")

    @property
    def xrange(self) -> (float, float):
        return self._xrange


# Valid keyword arguments for pyplot.plot()
PLOT_VALID_KWARGS = [
    "agg_filter", "alpha", "animated", "antialiased", "clip_box", "clip_on", "clip_path",
    "lw", "contains", "dash_capstyle", "dash_joinstyle", "dashes", "drawstyle", "figure",
    "fillstyle", "gid", "in_layout", "linestyle", "linewidth", "marker", "ls", "ds",
    "markeredgecolor", "markeredgewidth", "markerfacecolor", "markersize", "mfc", "mew",
    "markerfacecoloralt", "markevery", "path_effects", "picker", "pickradius", "rasterized",
    "sketch_params", "snap", "solid_capstyle", "solid_joinstyle", "transform", "url",
    "visible", "zorder", "aa", "c", "ms", "mfcalt", "mec"
]

# Valid keyword arguments for pyplot.errorbar()
ERRORBAR_VALID_KWARGS = [
    "ecolor", "elinewidth", "capsize", "capthick", "barsabove", "lolims", "uplims",
    "xlolims", "xuplims", "errorevery"
]

# Valid keyword arguments for pyplot.hist()
HIST_VALID_KWARGS = [
    "bins", "range", "density", "weights", "cumulative", "bottom", "align", "histtype",
    "orientation", "rwidth", "log", "stacked", "agg_filter", "alpha", "animated",
    "antialiased", "aa", "capstyle", "clip_box", "clip_on", "clip_path", "color",
    "contains", "edgecolor", "ec", "facecolor", "fc", "figure", "fill", "gid", "hatch",
    "in_layout", "joinstyle", "label", "linestyle", "ls", "linewidth", "lw", "path_effects",
    "picker", "rasterized", "sketch_params", "snap", "transform", "url", "visible", "zorder"
]

# Valid keyword arguments for numpy.histogram()
NP_HIST_VALID_KWARGS = ["bins", "range", "density", "weights"]
