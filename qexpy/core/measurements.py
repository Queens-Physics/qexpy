"""Define the data structure for a measurement."""

from __future__ import annotations

import numpy as np

from qexpy.typing import Number

from .quantity import Quantity


class Measurement(Quantity):
    """A measured value recorded with an uncertainty."""

    def __init__(
        self,
        data: Number,
        error: Number = 0.0,
        relative_error: Number | None = None,
        name: str = "",
        unit: str = "",
    ):
        super().__init__(name, unit)
        self._value = float(data)
        self._error, self._relative_error = _resolve_error(data, error, relative_error)

    @property
    def value(self) -> float:
        """The measured value."""
        return self._value

    @property
    def error(self) -> float:
        """The uncertainty of the measurement."""
        return self._error

    @property
    def relative_error(self) -> float:
        """The relative uncertainty of the measurement."""
        return self._relative_error


def _resolve_error(
    value: Number, error: Number, rel_error: Number | None
) -> tuple[float, float]:
    """Return the error and relative error resolved from user arguments."""

    if error < 0:
        raise ValueError(f"The error must be non-negative, got: {error}")
    if rel_error is not None and rel_error < 0:
        raise ValueError(f"The relative error must be non-negative, got {rel_error}")

    if rel_error is not None:
        return float(abs(value * rel_error)), float(rel_error)

    if np.isclose(float(value), 0.0):
        return float(error), np.inf

    return float(error), float(abs(error / value))
