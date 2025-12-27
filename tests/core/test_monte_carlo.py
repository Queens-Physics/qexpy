"""Unit tests for the Monte Carlo method of error propagation."""

import numpy as np
import pytest
from conftest import mc_tolerances

import qexpy as q
from qexpy.core.derived_value import DerivedValue
from qexpy.core.monte_carlo import _find_mode_and_error, monte_carlo


class TestMonteCarlo:
    """Unit tests for the Monte Carlo simulation."""

    @pytest.mark.flaky(reruns=1)
    def test_monte_carlo(self):
        """Test performing a Monte Carlo simulation."""

        a = q.Measurement(5, 0.2)
        b = q.Measurement(2, 0.1)
        c = q.Measurement(10, 0.5)

        res = a * c + b**2
        assert isinstance(res, DerivedValue)
        samples = monte_carlo(res._formula, sample_size=10000)
        assert samples.size == 10000

        mean_tol, err_tol = mc_tolerances(10000, res.error)
        assert np.isclose(np.mean(samples), res.value, atol=mean_tol)
        assert np.isclose(np.std(samples, ddof=1), res.error, atol=err_tol)

    @pytest.mark.flaky(reruns=1)
    def test_correlated_measurements(self):
        """Tests the Monte Carlo method with correlated measurements."""

        a = q.Measurement(5, 0.2)
        b = q.Measurement(2, 0.1)
        c = q.Measurement(10, 0.5)
        q.set_correlation(a, b, 0.5)

        res = a * c + b**2
        assert isinstance(res, DerivedValue)
        samples = monte_carlo(res._formula, sample_size=10000)
        assert samples.size == 10000

        mean_tol, err_tol = mc_tolerances(10000, res.error)
        assert np.isclose(np.mean(samples), res.value, atol=mean_tol)
        assert np.isclose(np.std(samples, ddof=1), res.error, atol=err_tol)

    def test_find_mode_and_error(self):
        """Tests calculating the mode and confidence interval from a histogram."""

        samples = np.random.normal(5, 0.5, 10000)
        n, bins = np.histogram(samples, bins=500)
        mode, mode_error = _find_mode_and_error(samples, n, bins, 0.68)
        assert np.isclose(mode, 5, rtol=0.1)
        assert np.isclose(mode_error, 0.5, rtol=0.1)


class TestMonteCarloController:
    """Tests configuring a Monte Carlo simulation."""

    def test_sample_size(self):
        """Tests configuring the sample size."""

    def test_hist_resolution(self):
        """Tests configuring the histogram resolution."""

    def test_confidence(self):
        """Tests configuring the confidence interval."""
