"""Package containing utility functions mostly for internal use"""

import sys

from IPython import get_ipython

from .exceptions import (
    IllegalArgumentError,
    UndefinedActionError,
    UndefinedOperationError,
)
from .printing import get_printer
from .units import (
    clear_unit_definitions,
    construct_unit_string,
    define_unit,
    operate_with_units,
    parse_unit_string,
)
from .utils import (
    calculate_covariance,
    check_operand_type,
    cov2corr,
    find_mode_and_uncertainty,
    load_data_from_file,
    numerical_derivative,
    validate_xrange,
    vectorize,
)

if "ipykernel" in sys.modules:  # pragma: no cover
    get_ipython().run_line_magic("matplotlib", "inline")
