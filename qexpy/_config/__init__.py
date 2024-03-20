"""Holds package-wide configurations for data analysis and visualization

QExPy adopts an options API similar to that of `pandas` to configure global behaviour related to
displaying data, methods of error propagation and more. Options can be accessed like attributes
using the "dotted-style".

"""

<<<<<<< HEAD
from qexpy._config.config import (
    get_option,
    set_option,
    describe_option,
    reset_option,
    options,
    option_context,
)
=======
from qexpy._config.config import get_option, set_option, describe_option, reset_option, options

__all__ = ["get_option", "set_option", "describe_option", "reset_option", "options"]
>>>>>>> 5aa3578 (Implement the core data structure for experimental values)
