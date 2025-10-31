"""Holds global configurations for qexpy.

qexpy adopts a pandas-style options API to customize data display, methods of
error propagation and more. Options can be accessed using the "dotted-style".

"""

from .config import (
    describe_option,
    get_option,
    options,
    reset_options,
    set_option,
    set_option_context,
)
