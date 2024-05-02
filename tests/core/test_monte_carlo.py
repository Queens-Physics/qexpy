"""Tests for more fine-grained control of the Monte Carlo error method"""

import pytest

import qexpy as q


class TestMonteCarloConfig:
    """Tests the MonteCarloConfig class"""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleans up global configurations"""
        yield
        q.reset_option()

    def test_sample_size(self):
        """Tests the Monte Carlo sample size"""

        m1 = q.Measurement(1.23, 0.02)
        m2 = q.Measurement(4.56, 0.03)
        res = m1 + m2
        res.error_method = "monte-carlo"
        assert res.mc.sample_size == 100000
        assert len(res.mc.samples) == 100000

        q.options.error.mc.sample_size = 1000
        assert res.mc.sample_size == 1000
        assert len(res.mc.samples) == 1000

        res.mc.sample_size = 200
        assert res.mc.sample_size == 200
        assert len(res.mc.samples) == 200

    def test_invalid_sample_size(self):
        """Tests that error is raised for invalid sample sizes"""

        m1 = q.Measurement(1.23, 0.02)
        m2 = q.Measurement(4.56, 0.03)
        res = m1 + m2
        res.error_method = "monte-carlo"
        with pytest.raises(ValueError, match="The sample size must be a positive integer!"):
            res.mc.sample_size = -100
