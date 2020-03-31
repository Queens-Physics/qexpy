"""Python library for scientific data analysis"""

#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#

import sys

__version__ = '2.0.0'

from .utils import load_data_from_file

from .settings import ErrorMethod, PrintStyle, UnitStyle, SigFigMode
from .settings import get_settings, reset_default_configuration
from .settings import set_sig_figs_for_value, set_sig_figs_for_error, set_error_method, \
    set_print_style, set_unit_style, set_monte_carlo_sample_size, set_plot_dimensions

from .data import Measurement, MeasurementArray, XYDataSet
from .data import get_covariance, set_covariance, get_correlation, set_correlation
from .data import sqrt, exp, sin, sind, cos, cosd, tan, tand, sec, secd, cot, cotd, \
    csc, cscd, asin, acos, atan, log, log10, pi, e
from .data import std, mean, sum  # pylint: disable=redefined-builtin
from .data import reset_correlations

from .fitting import fit, FitModel

# Check the python interpreter version
if sys.version_info[0] < 3:  # pragma: no coverage
    raise ImportError(
        "Error: QExPy is only supported on Python 3. Please upgrade your interpreter. "
        "If you're using Anaconda, you can download the correct version here: "
        "https://www.continuum.io/downloads")
