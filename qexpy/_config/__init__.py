"""Holds package-wide configurations for data analysis and visualization

QExPy adopts an options API similar to that of `pandas` to configure global behaviour related to
displaying data, methods of error propagation and more. Options can be accessed like attributes
using the "dotted-style".

"""

from qexpy._config.config import get_option, set_option, reset_option, options
