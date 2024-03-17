"""Initializes all package-wide configurations

This script is imported at the top level of the package to initialize all configurations

"""

import qexpy._config.config as cf

cf.register_option("style.unit", "fraction", cf.is_one_of_factory(["fraction", "exponent"]))

cf.register_option("style.value_format", "default", cf.is_one_of_factory(["default", "scientific"]))

cf.register_option("style.latex", False, cf.is_boolean)

cf.register_option("style.sig_fig.value", 1, cf.is_positive_integer)

cf.register_option("style.sig_fig.mode", "error", cf.is_one_of_factory(["value", "error"]))

cf.register_option(
    "error.method", "derivative", cf.is_one_of_factory(["derivative", "monte-carlo"])
)

cf.register_option("error.mc_sample_size", 100000, cf.is_positive_integer)

cf.register_option("plot.dimensions", (6.4, 4.8), cf.is_tuple_of_floats)
