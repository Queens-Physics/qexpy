"""Monte Carlo method of error propagation"""

from __future__ import annotations

from typing import Iterable, Dict, List

import numpy as np

import qexpy as q
from qexpy.core.formula import _Formula, _Operation, _find_measurements
from qexpy.utils import Unit


class _MeasurementSample(_Formula):
    """An array of random samples that simulates a measurement."""

    def __init__(self, samples: np.ndarray, unit: Unit):
        self._samples = samples
        self._unit = unit

    @property
    def value(self) -> float | np.ndarray:
        return self._samples

    def _derivative(self, x: _Formula) -> float:
        return 0  # pragma: no cover

    @property
    def unit(self) -> Unit:
        return self._unit  # pragma: no cover


def monte_carlo(formula: _Formula, sample_size: int) -> np.ndarray:
    """Use a Monte Carlo simulation to evaluate a formula."""

    sources = _find_measurements(formula)
    samples = _populate_samples(sources, sample_size)
    formula = _reconstruct_formula(formula, samples)
    return formula.value


def _populate_samples(
    sources: Iterable[q.core.Measurement], sample_size: int
) -> Dict[q.core.Measurement, np.ndarray]:
    """Populates the samples for all measurements"""

    samples = {}
    sources = list(sources)
    offset_matrix = np.vstack([np.random.normal(0, 1, sample_size) for _ in sources])
    offset_matrix = _correlate_samples(sources, offset_matrix)
    for measurement, offset in zip(sources, offset_matrix):
        samples[measurement] = measurement.value + offset * measurement.error
    return samples


def _correlate_samples(sources: List[q.core.Measurement], offsets: np.ndarray) -> np.ndarray:
    """Apply correlation to the offset matrix"""

    corr_matrix = np.array([[q.correlation(row, col) for col in sources] for row in sources])

    if np.count_nonzero(corr_matrix - np.diag(np.diagonal(corr_matrix))) == 0:
        return offsets  # if no correlations are present

    cholesky_decomposition = np.linalg.cholesky(corr_matrix)
    return np.dot(cholesky_decomposition, offsets)


def _reconstruct_formula(
    formula: _Formula, samples: Dict[q.core.Measurement, np.ndarray]
) -> _Formula:
    """Reconstruct the formula from the samples"""

    if isinstance(formula, q.core.Measurement):
        return _MeasurementSample(samples[formula], formula.unit)

    if isinstance(formula, _Operation):
        operands = tuple(_reconstruct_formula(operand, samples) for operand in formula.operands)
        return formula.__class__(*operands)

    return formula
