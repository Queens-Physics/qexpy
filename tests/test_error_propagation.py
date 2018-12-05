"""Unit tests for error propagation

This file contains tests for error propagation. Both the derivative method and the Monte Carlo
method are tested in this module. Error propagation is tested for independent values as well
as correlated values, raw measurements as well as derived values.

"""

import pytest
import qexpy as q


class TestErrorPropagation:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        q.reset_default_configuration()

    def test_derivative_error_propagation(self):
        a = q.Measurement(5, 0.2)
        b = q.Measurement(4, 0.1)
        c = q.Measurement(6.3, 0.5)
        d = q.Measurement(7.2, 0.5)
        result = c * d - b / a
        assert result.value == 44.56
        assert result.error == pytest.approx(4.783714456361291)
        result = q.sqrt(c) * d - b / q.exp(a)
        assert result.value == pytest.approx(18.04490478513969)
        assert result.error == pytest.approx(1.4454463754287323)

    def test_derivative_error_propagation_for_correlated_values(self):
        a = q.Measurement(5, 0.2)
        b = q.Measurement(4, 0.1)
        c = q.Measurement(6.3, 0.5)
        d = q.Measurement(7.2, 0.5)
        q.set_covariance(a, c, 0.09)
        q.set_covariance(b, d, -0.04)
        result = c * d - b / a
        assert result.value == 44.56
        assert result.error == pytest.approx(4.799792078830082)
        result = q.sqrt(c) * d - b / q.exp(a)
        assert result.value == pytest.approx(18.04490478513969)
        assert result.error == pytest.approx(1.4468831231108983)

    def test_monte_carlo_error_propagation(self):
        q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        a = q.Measurement(5, 0.2)
        b = q.Measurement(4, 0.1)
        c = q.Measurement(6.3, 0.5)
        d = q.Measurement(7.2, 0.5)
        result = c * d - b / a
        assert result.value == pytest.approx(44.56, 1e-1)
        assert result.error == pytest.approx(4.783714456361291, 1e-1)
        result = q.sqrt(c) * d - b / q.exp(a)
        assert result.value == pytest.approx(18.04490478513969, 1e-1)
        assert result.error == pytest.approx(1.4454463754287323, 1e-1)

    def test_monte_carlo_error_propagation_for_correlated_values(self):
        q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        a = q.Measurement(5, 0.2)
        b = q.Measurement(4, 0.1)
        c = q.Measurement(6.3, 0.5)
        d = q.Measurement(7.2, 0.5)
        q.set_covariance(a, c, 0.09)
        q.set_covariance(b, d, -0.04)
        result = c * d - b / a
        assert result.value == pytest.approx(44.56, 5e-2)
        assert result.error == pytest.approx(4.799792078830082, 5e-2)
        result = q.sqrt(c) * d - b / q.exp(a)
        assert result.value == pytest.approx(18.04490478513969, 5e-2)
        assert result.error == pytest.approx(1.4468831231108983, 5e-2)
