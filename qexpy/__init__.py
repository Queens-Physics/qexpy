"""Python package for scientific data analysis"""

import sys

__version__ = '2.0.1'

from .utils import load_data_from_file

from .settings import reset_default_configuration, UnitStyle, PrintStyle, ErrorMethod
from .settings import set_print_style, set_unit_style, set_error_method, set_monte_carlo_sample_size
from .settings import set_sig_figs_for_value, set_sig_figs_for_error

from .data import Measurement, MeasurementArray, XYDataSet
from .data import get_covariance, set_covariance, get_correlation, set_correlation
from .data import sqrt, exp, sin, cos, tan, asin, acos, atan, csc, sec, cot, log, log10, sind, cosd, tand, pi, e

from .fitting import fit, FitModel

# Check the python interpreter version
if sys.version_info[0] < 3:  # No reason to assume a future Python 4 will break comparability.
    raise ImportError("Error: QExPy is only supported on Python 3. Please upgrade your interpreter.\n"
                      "If you're using Anaconda, you can download the correct version here:\n"
                      "https://www.continuum.io/downloads")
