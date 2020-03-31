"""Package containing configurations for processing and displaying data"""

from .settings import ErrorMethod, PrintStyle, UnitStyle, SigFigMode
from .settings import get_settings, reset_default_configuration
from .settings import set_sig_figs_for_value, set_sig_figs_for_error, set_error_method, \
    set_print_style, set_unit_style, set_monte_carlo_sample_size, set_plot_dimensions
