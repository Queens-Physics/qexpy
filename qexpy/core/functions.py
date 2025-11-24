"""Module for various functions."""

from collections import defaultdict
from typing import NamedTuple, overload

import numpy as np

from qexpy.typing import ArrayLike

from .measurements import Measurement, RepeatedMeasurement


class _StatDependence(NamedTuple):
    """The statistical dependence between two quantities."""

    corr: float
    cov: float


class _StatDependenceGraph:
    """A graph of correlations between measurements."""

    _graph: dict[Measurement, dict[Measurement, _StatDependence]]

    def __init__(self) -> None:
        self._graph = defaultdict(dict)

    def get(self, var1: Measurement, var2: Measurement) -> _StatDependence:
        """Get the statistical dependence between two measurements."""
        if var1 is var2:
            return _StatDependence(1, var1.error**2)
        if var1 not in self._graph:
            return _StatDependence(0, 0)
        return self._graph[var1].get(var2, _StatDependence(0, 0))

    def add(self, var1: Measurement, var2: Measurement, dep: _StatDependence):
        """Set the statistical dependence between two measurements."""
        self._graph[var1][var2] = self._graph[var2][var1] = dep


_dependence_graph = _StatDependenceGraph()


@overload
def correlation(var1: ArrayLike, var2: ArrayLike) -> float: ...
@overload
def correlation(var1: Measurement, var2: Measurement) -> float: ...
def correlation(var1: Measurement | ArrayLike, var2: Measurement | ArrayLike) -> float:
    r"""Compute the correlation coefficient.

    The correlation coefficient is the normalized covariance, defined as

    .. math::
        \rho_{xy} = \frac{cov_{xy}}{\sigma_x\sigma_y}

    where :math:`\sigma_x` and :math:`\sigma_y` are the standard deviations.

    It measures the joint variability of two variables.

    See Also
    --------
    :func:`~qexpy.core.functions.covariance`

    """
    if isinstance(var1, ArrayLike) and isinstance(var2, ArrayLike):
        return float(np.corrcoef(var1, var2)[0][1])
    if isinstance(var1, Measurement) and isinstance(var2, Measurement):
        return _dependence_graph.get(var1, var2).corr
    raise TypeError(
        "The correlation is undefined between variables of type "
        f"{type(var1)} and {type(var2)}"
    )


@overload
def covariance(var1: ArrayLike, var2: ArrayLike) -> float: ...
@overload
def covariance(var1: Measurement, var2: Measurement) -> float: ...
def covariance(var1: Measurement | ArrayLike, var2: Measurement | ArrayLike) -> float:
    r"""Compute the covariance.

    The covariance is defined as

    .. math::
        cov_{xy} = \frac{\sum_{i}(x_i-\bar{x})(y_i-\bar{y})}{N-1}

    It measures the joint variability of two variables.

    See Also
    --------
    :func:`~qexpy.core.functions.correlation`

    """
    if isinstance(var1, ArrayLike) and isinstance(var2, ArrayLike):
        return float(np.cov(var1, var2)[0][1])
    if isinstance(var1, Measurement) and isinstance(var2, Measurement):
        return _dependence_graph.get(var1, var2).cov
    raise TypeError(
        "The covariance is undefined between variables of type "
        f"{type(var1)} and {type(var2)}."
    )


def set_correlation(var1: Measurement, var2: Measurement, corr: float | None = None):
    """Set the correlation between two measurements.

    Parameters
    ----------
    var1, var2 : Measurement
        The pair of measurements to set the correlation for.
    corr : float
        The correlation coefficient between the two measurements.

    """

    if not isinstance(var1, Measurement) or not isinstance(var2, Measurement):
        raise TypeError("Cannot set the correlation between non-measurements.")

    if var1.error == 0 or var2.error == 0:
        raise ArithmeticError("Cannot set correlation between values with 0 errors.")

    if (
        isinstance(var1, RepeatedMeasurement)
        and isinstance(var2, RepeatedMeasurement)
        and corr is None
    ):
        return _infer_dependence(var1, var2)

    if corr is None:
        raise ValueError("The correlation must be specified.")

    if corr > 1 or corr < -1:
        raise ValueError("The correlation coefficient must be between -1 and 1!")

    cov = corr * var1.error * var2.error

    _dependence_graph.add(var1, var2, _StatDependence(corr, cov))


def set_covariance(var1: Measurement, var2: Measurement, cov: float | None = None):
    """Set the covariance between two measurements.

    Parameters
    ----------
    var1, var2 : Measurement
        The pair of measurements to set the covariance for.
    cov : float
        The covariance between the two measurements.

    """

    if not isinstance(var1, Measurement) or not isinstance(var2, Measurement):
        raise TypeError("Cannot set the covariance between non-measurements.")

    if var1.error == 0 or var2.error == 0:
        raise ArithmeticError("Cannot set covariance between values with 0 errors.")

    if (
        isinstance(var1, RepeatedMeasurement)
        and isinstance(var2, RepeatedMeasurement)
        and cov is None
    ):
        return _infer_dependence(var1, var2)

    if cov is None:
        raise ValueError("The covariance must be specified.")

    corr = float(np.round(cov / (var1.error * var2.error), 14))

    if corr > 1 or corr < -1:
        raise ValueError(f"The covariance {cov} is non-physical!")

    _dependence_graph.add(var1, var2, _StatDependence(corr, cov))


def _infer_dependence(var1: RepeatedMeasurement, var2: RepeatedMeasurement) -> None:
    """Infer the statistical dependence between two repeated measurements."""

    if len(var1._data) != len(var2._data):
        raise ValueError(
            "The two repeated measurements must have the same sample size to "
            "infer their covariance or correlation coefficient."
        )

    cov = covariance(var1._data, var2._data)
    corr = correlation(var1._data, var2._data)
    _dependence_graph.add(var1, var2, _StatDependence(corr, cov))
