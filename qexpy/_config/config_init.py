"""Initialize all configurable options."""

from . import config as cf

format_unit_doc = """
: {"fraction", "product"}
    How units are displayed, specifically how division in a compound unit is
    represented, as a fraction, e.g., "m/s^2", or as a product with negative
    exponents, e.g., "m⋅s^-2".
"""

cf.register_option(
    "format.unit",
    "fraction",
    format_unit_doc,
    cf.is_one_of_factory(("fraction", "product")),
)

format_value_doc = """
: {"simple", "scientific"}
    How values are displayed, in the simple form, e.g., "123.4 +/- 0.5", or
    using the scientific notation, e.g., "(1.234 +/- 0.005) * 10^2"
"""

cf.register_option(
    "format.value",
    "simple",
    format_value_doc,
    cf.is_one_of_factory(("simple", "scientific")),
)

format_sigfigs_doc = """
: int
    The number of significant figures to display for numerical values.
"""

cf.register_option(
    "format.precision.sigfigs", 1, format_sigfigs_doc, cf.is_positive_integer
)

precision_mode_doc = """
: {"value", "error"}
    Controls whether to fix the number of significant figures for the value
    or the error. The other quantity will automatically have the same number
    of decimal places for consistency.
"""

cf.register_option(
    "format.precision.mode",
    "error",
    precision_mode_doc,
    cf.is_one_of_factory(("value", "error")),
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
    The sample size used in the Monte Carlo simulation for error propagation.
"""

cf.register_option(
    "error.mc.sample_size", 100000, mc_sample_size_doc, cf.is_positive_integer
)

mc_hist_resolution_doc = """
: int
    The number of bins in the histogram used to analyze the distribution and
    statistical properties of the samples in a Monte Carlo simulation.
"""

cf.register_option(
    "error.mc.hist_resolution", 100, mc_hist_resolution_doc, cf.is_positive_integer
)

mc_confidence_doc = """
: float
    The confidence interval used to estimate the error from the histogram
    analysis of a Monte Carlo simulation.
"""

cf.register_option(
    "error.mc.confidence", 0.68, mc_confidence_doc, cf.is_number_in_range(0.0, 1.0)
)
