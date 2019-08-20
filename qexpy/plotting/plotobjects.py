"""This file contains definition for objects on a plot"""

import abc
import inspect
import numpy as np
import qexpy.plotting.plot_utils as utils
import qexpy.data.data as dt
import qexpy.data.datasets as dts
from qexpy.utils.exceptions import InvalidRequestError, IllegalArgumentError


class ObjectOnPlot(abc.ABC):
    """A container for anything to be plotted"""

    def __init__(self, *args, **kwargs):
        # process format string
        fmt = kwargs.pop("fmt", args[0] if args and isinstance(args[0], str) else "")
        utils.validate_fmt(fmt)
        self._fmt = fmt

        # process color
        self._color = kwargs.pop("color", None)

        # add the rest to the object
        self.label = kwargs.pop("label", "")
        self.kwargs = kwargs

    @property
    def fmt(self) -> str:
        """The format string to be used in PyPlot"""
        return self._fmt

    @property
    def color(self) -> str:
        """The color of the object"""
        return self._color

    @color.setter
    def color(self, new_color):
        self._color = new_color


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


class XYDataSetOnPlot(XYObjectOnPlot):
    """A wrapper for an XYDataSet to be plotted"""

    def __init__(self, xdata, ydata, *args, **kwargs):

        # call super constructors
        XYObjectOnPlot.__init__(self, *args, **kwargs)
        self.dataset = dts.XYDataSet(xdata, ydata, **kwargs)

        # set default label and fmt if not requested
        if not self.label:
            self.label = self.dataset.name
        if not self.fmt:
            self._fmt = "o"

    @property
    def xvalues_to_plot(self):
        if self.xrange:
            return self.dataset.xvalues[self.__get_indices_from_xrange()]
        return self.dataset.xvalues

    @property
    def yvalues_to_plot(self):
        if self.xrange:
            return self.dataset.yvalues[self.__get_indices_from_xrange()]
        return self.dataset.yvalues

    @property
    def xerr_to_plot(self):
        if self.xrange:
            return self.dataset.xerr[self.__get_indices_from_xrange()]
        return self.dataset.xerr

    @property
    def yerr_to_plot(self):
        if self.xrange:
            return self.dataset.yerr[self.__get_indices_from_xrange()]
        return self.dataset.yerr

    @classmethod
    def from_xy_dataset(cls, dataset, **kwargs):
        """Wraps a regular XYDataSet object in a XYDataSetOnPlot object"""
        dataset.__class__ = cls
        XYObjectOnPlot.__init__(dataset, **kwargs)
        return dataset

    def __get_indices_from_xrange(self):
        return (self.xrange[0] <= self.dataset.xvalues) & (self.dataset.xvalues < self.xrange[1])


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
        if not self.xrange:
            raise InvalidRequestError("The domain of this function cannot be found.")
        result = self.func(self.xvalues_to_plot)
        errors = np.asarray(list(res.error if isinstance(res, dt.DerivedValue) else 0 for res in result))
        return errors if errors.size else np.empty(0)


class XYFitResultOnPlot(ObjectOnPlot):
    """Wrapper for an XYFitResult to be plotted"""

    def __init__(self, result, **kwargs):
        # initialize object
        ObjectOnPlot.__init__(self, **kwargs)
        self.fit_result = result

        xrange = result.xrange if result.xrange else (min(result.dataset.xvalues), max(result.dataset.xvalues))
        self.func_on_plot = FunctionOnPlot(result.fit_function, xrange=xrange, **kwargs)
        self.residuals_on_plot = XYDataSetOnPlot(result.dataset.xdata, result.residuals, **kwargs)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, new_color: str):
        self._color = new_color
        self.func_on_plot.color = new_color
        self.residuals_on_plot.color = new_color


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
