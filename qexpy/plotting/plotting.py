"""This file contains function definitions for plotting"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt

from qexpy.utils.exceptions import InvalidArgumentTypeError
import qexpy.settings.literals as lit
import qexpy.data.datasets as dts


class ObjectOnPlot:
    """A container for each individual dataset or function to be plotted"""

    def __init__(self):
        self.format = ""
        self.xrange = None

    @property
    def xvalues_to_plot(self):
        return np.array(0)

    @property
    def yvalues_to_plot(self):
        return np.array(0)


class XYDataSetOnPlot(dts.XYDataSet, ObjectOnPlot):
    """A wrapper for an XYDataSet to be plotted"""

    def __init__(self, xdata, ydata, **kwargs):
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
        return dataset


class FunctionOnPlot(ObjectOnPlot):
    """This is the wrapper for a function to be plotted"""

    def __init__(self, func):
        ObjectOnPlot.__init__(self)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def xvalues_to_plot(self):
        return np.array(0)

    @property
    def yvalues_to_plot(self):
        return np.array(0)


class Plot:
    """The data structure used for a plot"""

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
