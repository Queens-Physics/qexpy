"""Implements the Monte Carlo method of error propagation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from qexpy import options
from qexpy.typing import ArrayLike
from qexpy.units import Unit

from .formula import Formula, _collect_measurements, _Operation
from .functions import correlation
from .measurements import Measurement


class _SampleArray:
    """An array of random samples that simulates a measurement."""

    def __init__(self, samples: ArrayLike, unit: Unit):
        self._samples = np.asarray(samples)
        self._unit = unit

    @property
    def value(self) -> np.ndarray:
        return self._samples

    @property
    def error(self) -> float:
        return 0

    @property
    def unit(self) -> Unit:
        return self._unit

    def _derivative(self, x: Formula) -> float:
        return 0


def monte_carlo(formula: Formula, sample_size: int) -> np.ndarray:
    """Use a Monte Carlo simulation to evaluate a formula."""

    sources = _collect_measurements(formula)
    samples = _populate_samples(sources, sample_size)
    formula = _reconstruct_formula(formula, samples)
    return np.asarray(formula.value)


def _populate_samples(sources: Iterable, sample_size: int) -> dict:
    """Populates the samples for all measurements."""

    sources = list(sources)
    offsets = np.vstack([np.random.normal(0, 1, sample_size) for _ in sources])
    offsets = _correlate_samples(sources, offsets)
    return {m: m.value + o * m.error for m, o in zip(sources, offsets, strict=True)}


def _correlate_samples(sources: list[Measurement], offsets: np.ndarray) -> np.ndarray:
    """Apply correlation to the offset matrix."""

    corr_matrix = np.array([[correlation(r, c) for c in sources] for r in sources])

    if np.count_nonzero(corr_matrix - np.diag(np.diagonal(corr_matrix))) == 0:
        return offsets  # if no correlations are present

    cholesky_decomposition = np.linalg.cholesky(corr_matrix)
    return np.dot(cholesky_decomposition, offsets)


def _reconstruct_formula(formula: Formula, samples: dict) -> Formula:
    """Reconstruct the formula from the samples."""

    if isinstance(formula, Measurement):
        # Replace a measurement with its corresponding sample array.
        return _SampleArray(samples[formula], formula.unit)

    if isinstance(formula, _Operation):
        operands = (_reconstruct_formula(op, samples) for op in formula.operands)
        return formula.__class__(*operands)

    return formula


@dataclass(frozen=True)
class SampleStats:
    """Statistical properties of a sample."""

    mean: float
    std: float
    mode: float
    mode_error: float

    @classmethod
    def from_sample(cls, sample: np.ndarray, config: MonteCarloConfig) -> SampleStats:
        """Gather statistics from an array of samples."""
        mean, std = sample.mean(), sample.std(ddof=1)
        n, bins = np.histogram(sample, bins=config.hist_resolution)
        mode, mode_error = _find_mode_and_error(sample, n, bins, config.confidence)
        return SampleStats(mean, std, mode, mode_error)


@dataclass
class MonteCarloCache:
    """Stores the results of a Monte Carlo simulation."""

    samples: np.ndarray
    stats: SampleStats
    config_snapshot: MonteCarloConfig

    @property
    def sample_size(self) -> int:
        """The sample size used for this simulation."""
        return len(self.samples)

    @property
    def hist_resolution(self) -> int:
        """The histogram resolution used in this analysis."""
        return self.config_snapshot.hist_resolution

    @property
    def confidence(self) -> float:
        """The confidence interval used in the histogram analysis."""
        return self.config_snapshot.confidence

    @property
    def hist_config(self) -> tuple[int, float]:
        """The configuration for histogram analysis."""
        return self.hist_resolution, self.confidence


@dataclass
class MonteCarloConfig:
    """Stores the configurations of a Monte Carlo simulation."""

    _sample_size: int | None = None
    _hist_resolution: int | None = None
    _confidence: float = 0.68
    strategy: Literal["mean", "mode"] = "mean"

    @property
    def sample_size(self) -> int:
        """The sample size used in the simulation."""
        return self.sample_size or options.error.mc.sample_size

    @sample_size.setter
    def sample_size(self, sample_size: int):
        self._sample_size = sample_size

    @property
    def hist_resolution(self) -> int:
        """The number of bins in the histogram analysis of the samples."""
        return self.hist_resolution or options.error.mc.hist_resolution

    @hist_resolution.setter
    def hist_resolution(self, hist_resolution: int):
        self._hist_resolution = hist_resolution

    @property
    def confidence(self) -> float:
        """The confidence interval used in the histogram analysis."""
        return self._confidence or options.error.mc.confidence

    @confidence.setter
    def confidence(self, confidence: float):
        self._confidence = confidence

    def snapshot(self) -> MonteCarloConfig:
        """A snapshot of the current configuration."""
        return MonteCarloConfig(
            self.sample_size, self.hist_resolution, self.confidence, self.strategy
        )


class MonteCarloController:
    """Controls the Monte Carlo error method for a derived value."""

    def __init__(self, formula: Formula) -> None:
        self._formula = formula
        self._config = MonteCarloConfig()
        self._cache = None

    @property
    def value(self) -> float:
        """The value estimated from the simulated samples."""
        self._update_cache()
        assert self._cache
        if self._config.strategy == "mode":
            return float(self._cache.stats.mode)
        return float(self._cache.stats.mean)

    @property
    def error(self) -> float:
        """The uncertainty derived from the simulated sample distribution."""
        self._update_cache()
        assert self._cache
        if self._config.strategy == "mode":
            return float(self._cache.stats.mode_error)
        return float(self._cache.stats.std)

    @property
    def sample_size(self) -> int:
        """The sample size used in the simulation."""
        return self._config.sample_size

    @sample_size.setter
    def sample_size(self, sample_size: int):
        self._config.sample_size = sample_size

    @property
    def confidence(self):
        """The confidence interval used to derive the error from the histogram."""
        return self._config.confidence

    @confidence.setter
    def confidence(self, confidence: float):
        self._config.confidence = confidence

    @property
    def hist_resolution(self) -> int:
        """The number of bins in the histogram analysis of the samples."""
        return self._config.hist_resolution

    @hist_resolution.setter
    def hist_resolution(self, hist_resolution: int):
        self._config.hist_resolution = hist_resolution

    @property
    def hist_config(self) -> tuple[int, float]:
        """The configuration for histogram analysis."""
        return self.hist_resolution, self.confidence

    def _update_cache(self):
        """Update the cached samples if necessary."""

        if self._cache is None or self._cache.sample_size != self.sample_size:
            samples = monte_carlo(self._formula, self._config.sample_size)
            stats = SampleStats.from_sample(samples, self._config)
            self._cache = MonteCarloCache(samples, stats, self._config.snapshot())

        elif self.hist_config != self._cache.hist_config:
            samples = self._cache.samples
            n, bins = np.histogram(samples, bins=self.hist_resolution)
            mode, mode_error = _find_mode_and_error(samples, n, bins, self.confidence)
            new_stats = replace(self._cache.stats, mode=mode, mode_error=mode_error)
            new_snapshot = self._config.snapshot()
            self._cache.stats = new_stats
            self._cache.config_snapshot = new_snapshot


def _find_mode_and_error(samples, n, bins, confidence):
    """Calculates the mode and the confidence interval from a histogram distribution."""

    size = len(samples)
    max_idx = n.argmax()
    value = (bins[max_idx] + bins[max_idx + 1]) / 2
    count = n[max_idx]
    low_idx, high_idx = max_idx, max_idx
    while count < confidence * size:
        low_idx -= 1
        high_idx += 1
        count += n[low_idx] + n[high_idx]
    error = (bins[high_idx] + bins[high_idx + 1]) / 2 - value
    return value, error
