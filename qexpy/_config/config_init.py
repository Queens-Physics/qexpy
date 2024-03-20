"""Initializes all package-wide configurations

This script is imported at the top level of the package to initialize all configurations

"""

# pylint: disable-msg=invalid-name

import qexpy._config.config as cf

format_unit_doc = """
: {"fraction", "exponent"}
    How units are formatted, as a fraction, e.g., "m/s^2", or with exponents,
    e.g., "m⋅s^-2".
"""

cf.register_option(
    "format.style.unit", "fraction", format_unit_doc, cf.is_one_of_factory(["fraction", "exponent"])
)

format_value_doc = """
: {"default", "scientific"}
    How value strings are formatted, by default, e.g., "2.1 +/- 0.5", or using
    the scientific notation, e.g., "(1.200 +/- 0.004) * 10^5".
"""

cf.register_option(
    "format.style.value",
    "default",
    format_value_doc,
    cf.is_one_of_factory(["default", "scientific"]),
)

style_latex_doc = """
: bool
    Whether values are formatted with LaTeX grammar.
"""

cf.register_option("format.style.latex", False, style_latex_doc, cf.is_boolean)

precision_doc = """
: int
    The number of significant figures keep when displaying values.
"""

cf.register_option("format.precision.sig_fig", 1, precision_doc, cf.is_positive_integer)

precision_mode_doc = """
: {"value", "error"}
    Specifies whether the number of significant figures is imposed on the value
    or the error.
"""

cf.register_option(
    "format.precision.mode", "value", precision_mode_doc, cf.is_one_of_factory(["value", "error"])
)

error_method_doc = """
: {"derivative", "monte-carlo"}
    The preferred method of error propagation.
"""

cf.register_option(
    "error.method",
    "derivative",
    error_method_doc,
    cf.is_one_of_factory(["derivative", "monte-carlo"]),
)

mc_sample_size_doc = """
: int
    The sample size used in Monte Carlo error propagation.
"""

cf.register_option("error.mc.sample_size", 100000, mc_sample_size_doc, cf.is_positive_integer)
