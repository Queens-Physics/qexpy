"""This file contains helper methods for the plotting module"""

import re
from qexpy.utils.exceptions import InvalidArgumentTypeError, IllegalArgumentError


def validate_xrange(new_range: tuple, allow_empty=False):
    """Helper function to validate the xrange specified"""
    if not isinstance(new_range, tuple) and not isinstance(new_range, list):
        raise InvalidArgumentTypeError("xrange", new_range, "tuple or list of length 2")
    if allow_empty and new_range == ():
        return
    if len(new_range) != 2 or new_range[0] > new_range[1]:
        raise IllegalArgumentError(
            "Error: the xrange has to be a tuple of length 2 where the second number is larger than the first.")


def validate_fmt(new_fmt: str):
    """Helper function to validate the format string of a plot object"""
    if not isinstance(new_fmt, str):
        raise InvalidArgumentTypeError("fmt", new_fmt, "string")
    if not re.fullmatch(r"(\.|,|o|v|^|<|>|1|2|3|4|s|p|\*|h|H|\+|x|D|d|\||_|-|(--)|(-.)|(:)|[a-z])+", new_fmt):
        raise IllegalArgumentError(
            "The format string is invalid. Please refer to documentations for a list of valid format strings.")


def extract_plt_plot_arguments(kwargs: dict):
    """extract the arguments valid for plotting with pyplot.plot"""

    valid_args = ["agg_filter", "alpha", "animated", "antialiased", "clip_box", "clip_on", "clip_path", "lw",
                  "color", "contains", "dash_capstyle", "dash_joinstyle", "dashes", "drawstyle", "figure",
                  "fillstyle", "gid", "in_layout", "label", "linestyle", "linewidth", "marker", "ls", "ds",
                  "markeredgecolor", "markeredgewidth", "markerfacecolor", "markerfacecoloralt", "markersize",
                  "markevery", "path_effects", "picker", "pickradius", "rasterized", "sketch_params", "snap",
                  "solid_capstyle", "solid_joinstyle", "transform", "url", "visible", "xdata", "ydata",
                  "zorder", "aa", "c", "ms", "mfcalt", "mfc", "mew", "mec"]

    return dict(filter(lambda item: item[0] in valid_args, kwargs.items()))


def extract_plt_errorbar_arguments(kwargs: dict):
    """extract arguments valid for plotting with pyplot.errorbar"""

    valid_args = ["ecolor", "elinewidth", "capsize", "capthick", "barsabove", "lolims", "uplims", "xlolims",
                  "xuplims", "errorevery", "agg_filter", "alpha", "animated", "antialiased", "clip_box", "clip_on",
                  "clip_path", "lw", "color", "contains", "dash_capstyle", "dash_joinstyle", "dashes", "drawstyle",
                  "figure", "fillstyle", "gid", "in_layout", "label", "linestyle", "linewidth", "marker", "ls", "ds",
                  "markeredgecolor", "markeredgewidth", "markerfacecolor", "markerfacecoloralt", "markersize",
                  "markevery", "path_effects", "picker", "pickradius", "rasterized", "sketch_params", "snap",
                  "solid_capstyle", "solid_joinstyle", "transform", "url", "visible", "xdata", "ydata",
                  "zorder", "aa", "c", "ms", "mfcalt", "mfc", "mew", "mec"]

    return dict(filter(lambda item: item[0] in valid_args, kwargs.items()))
