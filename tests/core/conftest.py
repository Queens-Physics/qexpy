"""Utility functions and fixtures for testing."""

from __future__ import annotations

import math

from scipy.stats import chi2, norm


def mc_tolerances(
    sample_size: int, error_theory: float, alpha: float = 0.05
) -> tuple[float, float]:
    r"""Compute acceptance tolerances for the sample mean and the sample
    standard deviation of a Monte Carlo simulation.

    .. note::

        This test was implemented with the assistance of ChatGPT (OpenAI) and
        reviewed for correctness by the author.

    Parameters
    ----------
    sample_size : int
        The sample size of the Monte Carlo simulation.
    error_theory : float
        Theoretical standard deviation of the MC output.
    alpha : float, optional
        Two-sided significance level. Default is 0.05 (95% acceptance).

    Returns
    -------
    mean_tol : float
    std_tol : float

    """
    ndof = sample_size - 1

    z = norm.ppf(1.0 - alpha / 2.0)
    mean_tol = float(z * error_theory / math.sqrt(sample_size))

    q_lo = chi2.ppf(alpha / 2.0, ndof)
    q_hi = chi2.ppf(1.0 - alpha / 2.0, ndof)

    s_lo = math.sqrt(q_lo / ndof) * error_theory
    s_hi = math.sqrt(q_hi / ndof) * error_theory

    std_tol = max(error_theory - s_lo, s_hi - error_theory)

    return mean_tol, std_tol
