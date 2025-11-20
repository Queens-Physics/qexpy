"""Define the data structure for a measurement."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import NamedTuple, overload

import numpy as np
import scipy
from typing_extensions import override

from qexpy.typing import ArrayLike, Number

from .functions import correlation, covariance
from .quantity import Quantity


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


class Measurement(Quantity):
    """Record a measurement with an uncertainty.

    Parameters
    ----------
    data : float or array
        The measured value(s) of the quantity. This can be a single value or
        an array representing repeated measurements of the same quantity.
    error : float or array, optional
        The uncertainty of the measurement.
    relative_error : float or array, optional
        The relative error of the measurement.
    name : str, optional
        The name of this quantity (e.g., ``"mass"``, ``"voltage"``).
    unit : str, optional
        The unit of this quantity (e.g., ``"kg*m/s^2"``, ``"m^1s^-2"``).

    Examples
    --------
    >>> import qexpy as q
    >>> d = q.Measurement(5, 0.1, name="distance", unit="m")
    >>> d
    distance = 5.0 +/- 0.1 [m]
    >>> d.value
    5.0
    >>> d.error
    0.1

    Multiple measurements of the same quantity can be combined into a single
    measured value by passing them to ``Measurement`` as a list:

    >>> t = q.Measurement([8.01, 7.96, 8.02, 8.03], name="time", unit="s")
    >>> t
    time = 8.00 +/- 0.02 [s]

    In this case, if individual measurement uncertainties are not provided,
    the uncertainty of the measured quantity is estimated to be the standard
    error on the mean of the samples. On the other hand, if the uncertainties
    of the individual samples are provided, the estimated value and error of
    the quantity is given by an error-weighted average of the sample values
    and errors, which gives more importance to the measurements with smaller
    uncertainties:

    >>> t = q.Measurement(
    ...     [8.01, 7.96, 8.02, 8.03],
    ...     error=[0.02, 0.05, 0.05, 0.02],
    ...     name="time",
    ...     unit="s",
    ... )
    >>> t
    time = 8.02 +/- 0.01 [s]

    """

    @overload
    def __new__(
        cls,
        data: Number,
        error: Number = 0.0,
        *,
        relative_error: Number | None = None,
        name: str = "",
        unit: str = "",
    ) -> Measurement: ...
    @overload
    def __new__(
        cls,
        data: ArrayLike,
        error: Number | ArrayLike = 0.0,
        *,
        relative_error: Number | ArrayLike | None = None,
        name: str = "",
        unit: str = "",
    ) -> RepeatedMeasurement: ...
    def __new__(
        cls,
        data: Number | ArrayLike,
        error: Number | ArrayLike = 0.0,
        *,
        relative_error: Number | ArrayLike | None = None,
        name: str = "",
        unit: str = "",
    ):
        """Dispatch the construction of a Measurement."""
        if isinstance(data, Number):
            return object.__new__(Measurement)
        if isinstance(data, ArrayLike):
            return object.__new__(RepeatedMeasurement)
        raise TypeError(f"The data must be a real number or array, got: {type(data)}")

    @overload
    def __init__(
        self,
        data: ArrayLike,
        error: Number | ArrayLike = 0.0,
        *,
        relative_error: Number | ArrayLike | None = None,
        name: str = "",
        unit: str = "",
    ): ...
    @overload
    def __init__(
        self,
        data: Number,
        error: Number = 0.0,
        *,
        relative_error: Number | None = None,
        name: str = "",
        unit: str = "",
    ): ...
    def __init__(
        self,
        data: Number | ArrayLike,
        error: Number | ArrayLike = 0.0,
        *,
        relative_error: Number | ArrayLike | None = None,
        name: str = "",
        unit: str = "",
    ):
        super().__init__(name, unit)
        assert isinstance(data, Number)
        self._value = float(data)
        self._error, self._relative_error = _resolve_error(data, error, relative_error)
        self._id = uuid.uuid4()

    def __hash__(self):
        return hash(self._id)

    @property
    @override
    def value(self) -> float:
        """The measured value.

        :type: float

        """
        return self._value

    @property
    @override
    def error(self) -> float:
        """The uncertainty of the measurement.

        :type: float

        """
        return self._error

    @property
    @override
    def relative_error(self) -> float:
        """The relative uncertainty of the measurement.

        :type: float

        """
        return self._relative_error

    def use_standard_error(self):
        r"""Use the mean and the standard error on the mean of the samples as
        the value and uncertainty of this quantity.

        .. note::
            This method is only relevant when the measured value is recorded
            with an array of values representing multiple measurements taken
            of the same quantity.

        The standard error is defined as

        .. math::
            \sigma_{\bar{x}} = \frac{\sigma_x}{\sqrt{N}}

        where :math:`\sigma_x` and :math:`N` are the standard deviation and
        size of the samples. This method of combining multiple measurements
        ignores the individual measurement uncertainties, and relies on the
        observed scatter of the samples to estimate the error empirically.
        This method is the default when individual measurement uncertainties
        are not provided.

        """
        raise NotImplementedError

    def use_standard_deviation(self):
        r"""Use the mean and the standard deviation of the samples as the value
        and error of this quantity.

        .. note::
            This method is only relevant when the measured value is recorded
            with an array of values representing multiple measurements taken
            of the same quantity.

        The standard deviation is defined as

        .. math::
            \sigma_x = \sqrt{\frac{\sum{(x_i-\mu)^2}}{N}}

        where :math:`\mu` is the sample mean, and :math:`N` is the sample size.

        This method is typically not recommended, as the standard deviation
        reflects the uncertainty of each individual measurement, which defeats
        the purpose of performing repeated measurements in the first place.

        """
        raise NotImplementedError

    def use_error_weighted_mean(self):
        r"""Use the error-weighted average of the sample values and errors as
        the value and uncertainty of this quantity.

        .. note::
            This method is only relevant when the measured value is recorded
            with an array of values representing multiple measurements taken
            of the same quantity.

        The error weighted mean is defined as

        .. math::
            \mu_x = \frac{\sum_i w_i x_i}{\sum_i w_i}

        where :math:`w_i` is defined as

        .. math::
            w_i \equiv \frac{1}{\sigma_{x_i}^2}

        where :math:`\sigma_{x_i}` is the uncertainty of the :math:`i`-th
        measurement. The estimated uncertainty of the measured quantity is
        given by:

        .. math::
            \sigma_x = \sqrt{\frac{1}{\sum_i{w_i}}}

        This is the default and the most accurate method to combine multiple
        measurements of the same quantity, especially when those measurements
        have different individual uncertainties.

        """
        raise NotImplementedError


class RepeatedMeasurement(Measurement):
    """A repeatedly taken measurement."""

    def __init__(
        self,
        data: ArrayLike,
        error: Number | ArrayLike = 0,
        *,
        relative_error: Number | ArrayLike | None = None,
        name: str = "",
        unit: str = "",
    ):
        self._data = np.asarray(data)
        self._error = _resolve_error_array(data, error, relative_error)
        weighted_mean, weighted_error = _error_weighted_mean(self._data, self._error)
        self._stats = {
            "mean": np.mean(self._data),
            "std": np.std(self._data, ddof=1),
            "sem": scipy.stats.sem(self._data),
            "weighted_mean": weighted_mean,
            "weighted_error": weighted_error,
        }
        # By default, use the error weighted mean and error if errors are
        # specified. Otherwise, use the standard error on the mean.
        val, err = (
            (self._stats["mean"], self._stats["sem"])
            if np.any(self._error == 0)
            else (weighted_mean, weighted_error)
        )
        super().__init__(val, err, name=name, unit=unit)

    @override
    def use_standard_error(self):
        self._value = self._stats["mean"]
        self._error = self._stats["sem"]

    @override
    def use_standard_deviation(self):
        self._value = self._stats["mean"]
        self._error = self._stats["std"]

    @override
    def use_error_weighted_mean(self):
        self._value = self._stats["weighted_mean"]
        self._error = self._stats["weighted_error"]


@correlation.register
def _(var1: Measurement, var2: Measurement):
    if not isinstance(var2, Measurement):
        raise TypeError(
            "The correlation is undefined between variables of type "
            f"{type(var1)} and {type(var2)}."
        )
    if var1 is var2:
        return 1.0
    return _dependence_graph.get(var1, var2).corr


@covariance.register
def _(var1: Measurement, var2: Measurement):
    if not isinstance(var2, Measurement):
        raise TypeError(
            "The covariance is undefined between variables of type "
            f"{type(var1)} and {type(var2)}."
        )
    if var1 is var2:
        return var1.error**2
    return _dependence_graph.get(var1, var2).cov


def set_correlation(var1: Measurement, var2: Measurement, corr: float | None = None):
    """Set the correlation coefficient between two measurements.

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


def _resolve_error(
    value: Number, error: Number | ArrayLike, rel_error: Number | ArrayLike | None
) -> tuple[float, float]:
    """Return the error and relative error resolved from user arguments."""

    if isinstance(error, ArrayLike) or isinstance(rel_error, ArrayLike):
        raise TypeError("Cannot provide an array of errors to a single measurement.")

    if error < 0:
        raise ValueError(f"The error must be non-negative, got: {error}")
    if rel_error is not None and rel_error < 0:
        raise ValueError(f"The relative error must be non-negative, got {rel_error}")

    if rel_error is not None:
        return float(abs(value * rel_error)), float(rel_error)

    if float(value) == 0:
        return float(error), np.inf

    return float(error), float(abs(error / value))


def _resolve_error_array(
    data: ArrayLike, error: Number | ArrayLike, rel_error: Number | ArrayLike | None
) -> np.ndarray:
    """Return the error array resolved from user arguments."""

    if isinstance(rel_error, ArrayLike):
        if len(rel_error) != len(data):
            raise ValueError(
                f"The length of the relative_error array: {len(rel_error)} "
                "does not match the length of the data array: {data}"
            )
        return np.array(data) * np.array(rel_error)

    if isinstance(rel_error, Number):
        if rel_error < 0:
            raise ValueError(f"relative_error must be non-negative, got {rel_error}")
        return np.array(data) * rel_error

    if isinstance(error, Number):
        if error < 0:
            raise ValueError(f"error must be non-negative, got {error}")
        return np.ones_like(data) * error

    if isinstance(error, ArrayLike):
        if len(error) != len(data):
            raise ValueError(
                f"The length of the error array: {len(error)} does "
                "not match the length of the data array: {data}"
            )
        return np.array(error)

    raise TypeError("Invalid error or relative_error specified!")


def _error_weighted_mean(data, error) -> tuple[float, float]:
    """Calculate the error weighted mean and uncertainty."""
    if np.any(error == 0):
        return np.nan, np.nan
    weights = 1 / (error**2)
    return np.average(data, weights=weights), 1 / np.sqrt(np.sum(weights))
