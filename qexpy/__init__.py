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

__version__ = '3.0.0'

from .utils import load_data_from_file

from .settings import UnitStyle, PrintStyle, ErrorMethod, SigFigMode
from .settings import get_settings, set_sig_figs_for_error, set_sig_figs_for_value, set_error_method, \
    set_monte_carlo_sample_size, set_print_style, set_unit_style, reset_default_configuration

from .data import Measurement, MeasurementArray, XYDataSet
from .data import get_covariance, set_covariance, get_correlation, set_correlation
from .data import sqrt, exp, sin, cos, tan, asin, acos, atan, csc, sec, cot, log, log10, sind, cosd, tand, pi, e

from .fitting import fit, FitModel

from .plotting import plot, new_plot
from .plotting import show as show_plot

# Check the python interpreter version
if sys.version_info[0] < 3:  # No reason to assume a future Python 4 will break comparability.
    raise ImportError("Error: QExPy is only supported on Python 3. Please upgrade your interpreter.\n"
                      "If you're using Anaconda, you can download the correct version here:\n"
                      "https://www.continuum.io/downloads")
