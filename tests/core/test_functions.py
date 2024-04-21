"""Unit tests for general functions in the core module"""

import pytest

import numpy as np
import qexpy as q


def test_correlation():
    """Tests calculating the correlation between two arrays"""

    a = np.array([4.9, 5, 5.1])
    b = np.array([3.1, 3.3, 3.2])
    expected = np.corrcoef(a, b)[0][1]
    assert q.correlation(a, b) == pytest.approx(expected)

    a = q.Measurement([4.9, 5, 5.1])
    b = q.Measurement([3.1, 3.3, 3.2])
    assert q.correlation(a, b) == 0

    a.set_correlation(b)
    assert q.correlation(a, b) == pytest.approx(expected)


def test_covariance():
    """Tests calculating the covariance between two arrays"""

    a = np.array([4.9, 5, 5.1])
    b = np.array([3.1, 3.3, 3.2])
    expected = np.cov(a, b)[0][1]
    assert q.covariance(a, b) == pytest.approx(expected)

    a = q.Measurement([4.9, 5, 5.1])
    b = q.Measurement([3.1, 3.3, 3.2])
    assert q.covariance(a, b) == 0

    a.set_covariance(b)
    assert q.covariance(a, b) == pytest.approx(expected)
