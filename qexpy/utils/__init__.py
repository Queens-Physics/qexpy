"""Package containing utility functions mostly for internal use"""

from .utils import load_data_from_file
from .utils import vectorize, check_operand_type, validate_xrange
from .utils import numerical_derivative, calculate_covariance, cov2corr, \
    find_mode_and_uncertainty
from .exceptions import IllegalArgumentError, UndefinedActionError, UndefinedOperationError
from .units import parse_unit_string, construct_unit_string, operate_with_units
from .printing import get_printer

import sys
import IPython

if "ipykernel" in sys.modules:  # pragma: no cover
    IPython.get_ipython().magic("matplotlib inline")
