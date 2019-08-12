"""Utility functions mostly for internal use"""

from .utils import load_data_from_file, count_significant_figures
from . import utils as _utils

if _utils._in_notebook():
    # call matplotlib inline
    _utils._mpl_output_notebook()
