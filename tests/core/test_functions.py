"""Tests for various functions."""

import numpy as np
import pytest

import qexpy as q


def test_correlation_array():
    """Tests calculating the correlation between two arrays."""

    a = np.array([4.9, 5, 5.1])
    b = np.array([3.1, 3.3, 3.2])
    expected = np.corrcoef(a, b)[0][1]
    assert np.isclose(q.correlation(a, b), expected)


def test_covariance_array():
    """Tests calculating the covariance between two arrays."""

    a = np.array([4.9, 5, 5.1])
    b = np.array([3.1, 3.3, 3.2])
    expected = np.cov(a, b)[0][1]
    assert np.isclose(q.covariance(a, b), expected)


def test_correlation_measurements():
    """Tests obtaining the correlation between two measurements."""

    a = q.Measurement(5, 0.2)
    b = q.Measurement(6, 0.1)
    assert q.correlation(a, b) == 0

    q.set_correlation(a, b, 0.5)
    assert q.correlation(a, b) == 0.5
    assert np.isclose(q.covariance(a, b), 0.5 * 0.1 * 0.2)


def test_infer_correlation():
    """Tests inferring the correlation between two repeated measurements."""

    a = q.Measurement([4.9, 5, 5.1])
    b = q.Measurement([3.1, 3.3, 3.2])
    assert q.correlation(a, b) == 0

    q.set_correlation(a, b)
    expected = np.corrcoef([4.9, 5, 5.1], [3.1, 3.3, 3.2])[0][1]
    assert np.isclose(q.correlation(a, b), expected)


def test_covariance():
    """Tests calculating the covariance between two arrays."""

    a = q.Measurement(5, 0.2)
    b = q.Measurement(6, 0.1)
    assert q.covariance(a, b) == 0

    q.set_covariance(a, b, 0.01)
    assert q.covariance(a, b) == 0.01
    assert np.isclose(q.correlation(a, b), 0.01 / (0.1 * 0.2))


def test_infer_covariance():
    """Tests inferring the covariance between two repeated measurements."""

    a = q.Measurement([4.9, 5, 5.1])
    b = q.Measurement([3.1, 3.3, 3.2])
    assert q.covariance(a, b) == 0

    q.set_covariance(a, b)
    expected = np.cov([4.9, 5, 5.1], [3.1, 3.3, 3.2])[0][1]
    assert q.covariance(a, b) == expected
    assert np.isclose(
        q.correlation(a, b),
        expected / (np.std([4.9, 5, 5.1], ddof=1) * np.std([3.1, 3.3, 3.2], ddof=1)),
    )


def test_correlation_with_self():
    """Tests the covariance and correlation between a measurement and itself."""

    a = q.Measurement(1.5, 0.2)
    assert np.isclose(q.covariance(a, a), 0.04)
    assert q.correlation(a, a) == 1


def test_length_mismatch_error():
    """Tests that an error is raised when the length of examples is not the same."""

    a = q.Measurement([4.9, 5, 5.1, 5.2])
    b = q.Measurement([3.1, 3.2, 3.3])
    with pytest.raises(ValueError, match="must have the same sample size"):
        q.set_covariance(a, b)


def test_invalid_covariance_and_correlation():
    """Tests when the covariance or correlation is unphysical."""

    a = q.Measurement(5, 0.2)
    b = q.Measurement(6, 0.1)

    with pytest.raises(ValueError, match="between -1 and 1"):
        q.set_correlation(a, b, 100)

    with pytest.raises(ValueError, match="is non-physical"):
        q.set_covariance(a, b, 0.1)
