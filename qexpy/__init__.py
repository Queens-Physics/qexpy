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

__version__ = "2.0.4"

from .data import (  # pylint: disable=redefined-builtin
    Measurement,
    MeasurementArray,
    XYDataSet,
    acos,
    asin,
    atan,
    cos,
    cosd,
    cot,
    cotd,
    csc,
    cscd,
    e,
    exp,
    get_correlation,
    get_covariance,
    log,
    log10,
    mean,
    pi,
    reset_correlations,
    sec,
    secd,
    set_correlation,
    set_covariance,
    sin,
    sind,
    sqrt,
    std,
    sum,
    tan,
    tand,
)
from .fitting import FitModel, fit
from .settings import (
    ErrorMethod,
    PrintStyle,
    SigFigMode,
    UnitStyle,
    get_settings,
    reset_default_configuration,
    set_error_method,
    set_monte_carlo_sample_size,
    set_plot_dimensions,
    set_print_style,
    set_sig_figs_for_error,
    set_sig_figs_for_value,
    set_unit_style,
)
from .utils import clear_unit_definitions, define_unit, load_data_from_file

# Check the python interpreter version
if sys.version_info[0] < 3:  # pragma: no coverage
    raise ImportError(
        "Error: QExPy is only supported on Python 3. Please upgrade your interpreter. "
        "If you're using Anaconda, you can download the correct version here: "
        "https://www.continuum.io/downloads"
    )
