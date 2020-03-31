"""Unit tests for the settings sub-package"""

import pytest

from qexpy.settings.settings import ErrorMethod, PrintStyle, UnitStyle, SigFigMode

import qexpy.settings.literals as lit
import qexpy.settings.settings as sts


class TestSettings:
    """Unit tests for global settings"""

    def test_settings(self):
        """test change and get settings"""

        sts.set_unit_style(lit.FRACTION)
        assert sts.get_settings().unit_style == UnitStyle.FRACTION
        sts.set_unit_style(UnitStyle.EXPONENTS)
        assert sts.get_settings().unit_style == UnitStyle.EXPONENTS

        sts.set_print_style(PrintStyle.SCIENTIFIC)
        assert sts.get_settings().print_style == PrintStyle.SCIENTIFIC
        sts.set_print_style(lit.DEFAULT)
        assert sts.get_settings().print_style == PrintStyle.DEFAULT

        sts.set_error_method(ErrorMethod.MONTE_CARLO)
        assert sts.get_settings().error_method == ErrorMethod.MONTE_CARLO
        sts.set_error_method(lit.DERIVATIVE)
        assert sts.get_settings().error_method == ErrorMethod.DERIVATIVE

        sts.set_plot_dimensions((8, 4))
        assert sts.get_settings().plot_dimensions == (8, 4)

        sts.set_monte_carlo_sample_size(10000)
        assert sts.get_settings().monte_carlo_sample_size == 10000

        sts.set_sig_figs_for_error(4)
        assert sts.get_settings().sig_fig_value == 4
        assert sts.get_settings().sig_fig_mode == SigFigMode.ERROR
        sts.set_sig_figs_for_value(3)
        assert sts.get_settings().sig_fig_value == 3
        assert sts.get_settings().sig_fig_mode == SigFigMode.VALUE

    def test_invalid_settings(self):
        """tests for rejecting invalid settings"""

        with pytest.raises(ValueError):
            sts.set_sig_figs_for_value(-1)

        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            sts.set_sig_figs_for_error(0.5)

        with pytest.raises(ValueError):
            sts.set_monte_carlo_sample_size(-1)

        with pytest.raises(ValueError):
            sts.set_plot_dimensions((0, 0))

        with pytest.raises(ValueError):
            sts.set_error_method(lit.DEFAULT)

        with pytest.raises(ValueError):
            sts.set_print_style(lit.DERIVATIVE)

        with pytest.raises(ValueError):
            sts.set_unit_style(lit.DERIVATIVE)

        with pytest.raises(ValueError):
            sts.set_plot_dimensions(10)

    def test_reset_error_configurations(self):
        """test for reset all configurations to default"""

        sts.reset_default_configuration()
        assert sts.get_settings().error_method == ErrorMethod.DERIVATIVE
        assert sts.get_settings().sig_fig_mode == SigFigMode.AUTOMATIC
        assert sts.get_settings().sig_fig_value == 1
        assert sts.get_settings().monte_carlo_sample_size == 10000
        assert sts.get_settings().unit_style == UnitStyle.EXPONENTS
        assert sts.get_settings().plot_dimensions == (6.4, 4.8)

    def test_use_mc_sample_size(self):
        """test for temporarily setting monte-carlo sample size"""

        @sts.use_mc_sample_size(100)
        def test_func():
            assert sts.get_settings().monte_carlo_sample_size == 100

        sts.set_monte_carlo_sample_size(10000)
        test_func()
        assert sts.get_settings().monte_carlo_sample_size == 10000
