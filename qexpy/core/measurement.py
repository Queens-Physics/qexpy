"""Defines measurements"""

# pylint: disable=too-many-arguments,protected-access,too-many-instance-attributes

from __future__ import annotations

import uuid
from copy import copy
from numbers import Real
from typing import NamedTuple, Dict, Tuple

import numpy as np

import qexpy as q
from qexpy._typing import ArrayLikeT, ArrayLike
from .experimental_value import ExperimentalValue
from .formula import _Formula


class _Correlation(NamedTuple):
    """The correlation between two quantities"""

    corr: float
    cov: float


_correlations: Dict[Measurement, Dict[Measurement, _Correlation]] = {}


class Measurement(ExperimentalValue, _Formula):
    """A measurement recorded with an uncertainty

    Stores a measured value and the uncertainty of the measurement. A measurement can be recorded
    as a single value or a series of repeated takes. The latter should be interpreted as a single
    quantity measured multiple times, typically to mitigate the uncertainty. In this case, a
    :py:class:`~qexpy.core.RepeatedMeasurement` is returned.

    Parameters
    ----------

    data : float or array
        The measured value(s) of this quantity. This can be a single value or an array of values
        representing multiple measurements taken towards the same quantity.
    error : float or array, default=0
        The uncertainty of the measurement. If ``data`` is an array, this can be a single value
        that represents the uncertainty of all measurements, or an array that correspond to the
        uncertainties of each measurement in ``data``.
    relative_error : float or array, optional
        The relative error of the measurement. If provided, it will override ``error``.
    name : str, optional
        The name of this quantity (e.g. ``"mass"``, ``"voltage"``).
    unit : str, optional
        The unit of this quantity (e.g. ``"kg*m/s^2"``, ``"m^1s^-2"``).

    Examples
    --------

    >>> import qexpy as q
    >>> a = q.Measurement(5, 0.5, name='mass', unit='kg')
    >>> a
    mass = 5.0 +/- 0.5 [kg]

    And to take an array of repeated measurements:

    >>> b = q.Measurement([4.9, 4.8, 5])
    >>> b
    4.90 +/- 0.06

    See Also
    --------

    :class:`~qexpy.core.RepeatedMeasurement`
    :class:`~qexpy.core.ExperimentalValue`

    Attributes
    ----------

    value : float
    error : float
    relative_error : float
    std : float
    name : str
    unit : str

    Methods
    -------

    get_covariance
    get_correlation
    set_covariance
    set_correlation

    """

    def __new__(
        cls,
        data: Real | ArrayLike,
        error: Real | ArrayLike = 0.0,
        relative_error: Real | ArrayLike = None,
        name: str = "",
        unit: str = "",
    ):
        if isinstance(data, Real):
            return object.__new__(Measurement)

        if isinstance(data, ArrayLikeT):
            return object.__new__(RepeatedMeasurement)

        raise TypeError(f"The data must be a real number or an array, got: {type(data)}")

    def __init__(
        self,
        data: Real,
        error: Real = 0.0,
        relative_error: Real = None,
        name: str = "",
        unit: str = "",
    ):
        self._value = float(data)
        if not isinstance(error, Real):
            raise TypeError(f"The error must be a real number, got: {type(error)}")
        error = float(error) if relative_error is None else float(relative_error) * float(data)
        if error < 0:
            raise ValueError(f"The error must be non-negative, got: {error}")
        self._error = error
        self._id = uuid.uuid4()
        super().__init__(name=name, unit=unit)

    def __abs__(self):
        if self.value < 0:
            return -self
        return copy(self)

    def __copy__(self):
        obj = object.__new__(Measurement)
        obj._value = self._value
        obj._error = self._error
        obj._id = uuid.uuid4()
        obj._name = self._name
        obj._unit = self._unit
        return obj

    def __hash__(self):
        return self._id.__hash__()

    @property
    def value(self) -> float:
        return self._value

    @property
    def error(self) -> float:
        return self._error

    @property
    def std(self) -> float:
        """The standard deviation of the measured value

        For a single measurement, the standard deviation is just the error.

        :type: float

        """
        return self.error

    def _derivative(self, x: _Formula) -> float:
        return 1 if self is x else 0

    def get_covariance(self, other: Measurement):
        """Gets the covariance between two measurements

        The covariance is defined as

        .. math::
            cov_{xy} = \\frac{\\sum_{i}(x_i-\\bar{x})(y_i-\\bar{y})}{N-1}

        It measures the joint variability of two variables

        QExPy does not assume that any two measurements are correlated, so 0 is returned here by
        default, unless the covariance between the two variables is explicitly set or declared.

        Parameters
        ----------

        other : Measurement
            The other measurement

        Returns
        -------

        cov : float
            The covariance between this and the other measurement.

        See Also
        --------

        :py:func:`~Measurement.set_covariance`

        """
        if not isinstance(other, Measurement):
            raise TypeError(
                f"The covariance is only defined between measurements, got: {type(other)}."
            )
        if self is other:
            return self.std**2
        return _correlations.get(self, {}).get(other, _Correlation(0, 0)).cov

    def get_correlation(self, other: Measurement):
        """Gets the correlation coefficient between two measurements

        The correlation coefficient is the normalized covariance, defined as

        .. math::
            \\rho_{xy} = \\frac{cov_{xy}}{\\sigma_x\\sigma_y}

        where :math:`\\sigma_x` and :math:`\\sigma_y` are the standard deviations.

        It measures the joint variability of two variables

        QExPy does not assume that any two measurements are correlated, so 0 is returned here by
        default, unless the correlation between the two variables is explicitly set or declared.

        Parameters
        ----------

        other : Measurement
            The other measurement

        Returns
        -------

        corr : float
            The correlation coefficient between this and the other measurement.

        See Also
        --------

        :py:func:`~Measurement.set_correlation`

        """
        if not isinstance(other, Measurement):
            raise TypeError(
                f"The correlation is only defined between measurements, got: {type(other)}."
            )
        if self is other:
            return 1.0
        return _correlations.get(self, {}).get(other, _Correlation(0, 0)).corr

    def set_covariance(self, other: Measurement, cov: float):
        """Sets the covariance between two measurements

        The covariance is defined as

        .. math::
            cov_{xy} = \\frac{\\sum_{i}(x_i-\\bar{x})(y_i-\\bar{y})}{N-1}

        It measures the joint variability of two variables. If two measurements are correlated,
        the covariance will be taken into account in error propagation.

        Parameters
        ----------

        other : Measurement
            The other measurement
        cov : float
            The covariance between this and the other measurement

        See Also
        --------

        :py:func:`~Measurement.get_covariance`

        """

        if not isinstance(other, Measurement):
            raise TypeError(
                f"The covariance is only defined between measurements, got: {type(other)}."
            )

        if self.std == 0 or other.std == 0:
            raise ArithmeticError("Cannot set covariance between values with a 0 uncertainty")

        corr = np.round(cov / (self.std * other.std), 14)

        # Check that the result makes sense
        if corr > 1 or corr < -1:
            raise ValueError(f"The covariance: {cov} is non-physical!")

        _correlations.setdefault(self, {})[other] = _Correlation(corr, cov)
        _correlations.setdefault(other, {})[self] = _Correlation(corr, cov)

    def set_correlation(self, other: Measurement, corr: float):
        """Sets the correlation between two measurements

        The correlation coefficient is the normalized covariance, defined as

        .. math::
            \\rho_{xy} = \\frac{cov_{xy}}{\\sigma_x\\sigma_y}

        where :math:`\\sigma_x` and :math:`\\sigma_y` are the standard deviations.

        It measures the joint variability of two variables.  If two measurements are correlated,
        the correlation will be taken into account in error propagation.

        Parameters
        ----------

        other : Measurement
            The other measurement
        corr : float
            The correlation coefficient between this and the other measurement

        See Also
        --------

        :py:func:`~Measurement.get_correlation`

        """
        if not isinstance(other, Measurement):
            raise TypeError(
                f"The correlation is only defined between measurements, got: {type(other)}."
            )

        if self.std == 0 or other.std == 0:
            raise ArithmeticError("Cannot set correlation between values with a 0 uncertainty")

        if corr < -1 or corr > 1:
            raise ValueError("The correlation coefficient must be between -1 and 1!")

        cov = np.round(corr * self.std * other.std, 14)

        _correlations.setdefault(self, {})[other] = _Correlation(corr, cov)
        _correlations.setdefault(other, {})[self] = _Correlation(corr, cov)


def _error_weighted_mean(_data, _errors) -> Tuple[float, float]:
    if np.any(_errors == 0):
        return np.nan, np.nan
    weights = 1 / (_errors**2)
    return np.average(_data, weights=weights), 1 / np.sqrt(np.sum(weights))


class RepeatedMeasurement(Measurement):
    """A measurement with repeated takes

    A ``RepeatedMeasurement`` is produced if a :py:class:`~qexpy.core.Measurement` is taken with an
    array of values. By default, the mean of the samples is used as the value, and the standard
    error (error on the mean) of the sample distribution is used as the uncertainty. Note that in
    this case, the error of each individual measurement is not taken into account.

    Attributes
    ----------

    value : float
    error : float
    relative_error : float
    std : float
    name : str
    unit : str

    Methods
    -------

    get_covariance
    get_correlation
    set_covariance
    set_correlation
    use_error_weighted_mean
    use_mean_and_std
    use_mean_and_std_error

    See Also
    --------

    :class:`~qexpy.core.Measurement`

    """

    def __init__(
        self,
        data: ArrayLike,
        error: Real | ArrayLike = 0.0,
        relative_error: Real | ArrayLike = None,
        name: str = "",
        unit: str = "",
    ):
        self._data, self._errors = q.core.array.pack_data_arrays(data, error, relative_error)
        self._mean = float(np.mean(self._data))
        self._err_weighted_mean, self._prop_error = _error_weighted_mean(self._data, self._errors)
        self._std = float(np.std(self._data, ddof=1))
        self._std_err = self._std / np.sqrt(len(self._data))
        super().__init__(self._mean, self._std_err, name=name, unit=unit)

    def __copy__(self):
        obj = object.__new__(RepeatedMeasurement)
        obj._data = self._data
        obj._errors = self._errors
        obj._mean = self._mean
        obj._err_weighted_mean = self._err_weighted_mean
        obj._prop_error = self._prop_error
        obj._std = self._std
        obj._std_err = self._std_err
        obj._value = self._value
        obj._error = self._error
        obj._name = self._name
        obj._unit = self._unit
        obj._id = uuid.uuid4()
        return obj

    @property
    def value(self) -> float:
        """The value of this quantity

        For a quantity recorded with a series of repeated measurements, by default, the estimated
        centre value is the average of all measurements.

        :type: float

        """
        return self._value

    @property
    def error(self) -> float:
        """The uncertainty of this value

        The uncertainty of a repeatedly measured value is typically the standard error (error on
        the mean) of the repeated measurements. In other words, the uncertainty is estimated from
        the statistical properties of the original measurement array. In this case, the uncertainty
        of each individual measurement within this series of takes is irrelevant.

        :type: float

        """
        return self._error

    @property
    def std(self) -> float:
        """The standard deviation of the array of samples

        For a measurement recorded as an array of repeated takes, the standard deviation is a
        statistical property of the samples, which reflects the uncertainty of each individual
        measurement within this series of takes.

        :type: float

        """
        return self._std

    @property
    def data(self):
        """The array of samples

        :type: numpy.ndarray

        """
        return self._data

    def use_error_weighted_mean(self):
        """Sets the value of this quantity to the error weighted mean of the samples

        The error weighted mean is defined as

        .. math::
            \\mu_x = \\frac{1}{N} \\sum_{i=1}^N \\frac{x_i}{\\sigma_i^2}

        This method gives more importance to the measurements with smaller uncertainties. If the
        error weighted mean is chosen to represent the value of this measurement, the error will
        be set to the weight propagated error, i.e., the error from standard error propagation
        when computing the error weighted mean.

        Examples
        --------

        >>> import qexpy as q
        >>> a = q.Measurement([4.9, 5, 5.1], [0.1, 0.5, 0.5])
        >>> a.value, a.error
        (5.0, 0.05773502691896237)

        By default, the value is exactly equal to 5, which is the mean of the samples.

        >>> a.use_error_weighted_mean()
        >>> a.value, a.error
        (4.911111111111111, 0.09622504486493764)

        We can see now that the value is closer to 4.9. This is because the uncertainty of the
        sample 4.9 is 0.1, significantly smaller than the uncertainty of the other samples. The
        error weighted mean gives more importance to the samples with smaller uncertainties.

        """
        self._value, self._error = self._err_weighted_mean, self._prop_error

    def use_mean_and_std_error(self):
        """Sets the value of this quantity to the mean and standard error of the samples

        This is the default behaviour.

        """
        self._value, self._error = self._mean, self._std_err

    def use_mean_and_std(self):
        """Sets the value to the mean of the samples and error to the standard deviation

        This is typically not recommended, as the standard deviation reflects the uncertainty of
        each individual measurement within this series of takes, which defeats the purpose of
        recording a series of repeated measurements of the same quantity.

        Examples
        --------

        >>> import qexpy as q
        >>> a = q.Measurement([4.9, 5, 5.1], [0.1, 0.5, 0.5])
        >>> a.value, a.error
        (5.0, 0.05773502691896237)
        >>> a.use_mean_and_std()
        >>> a.value, a.error  # this would be typically be an over estimation
        (5.0, 0.09999999999999964)

        """
        self._value, self._error = self._mean, self._std

    def set_covariance(self, other: Measurement, cov: float = None):
        """Sets the covariance between two measurements

        For two repeated measurements, if the `cov` argument is not provided, the covariance will
        be calculated from the raw samples taken towards each value. Keep in mind that this does
        not necessarily make sense for every pair of repeated measurements.

        Parameters
        ----------

        other : Measurement
            The other measurement
        cov : float, optional
            The value of the covariance

        Examples
        --------

        >>> import qexpy as q
        >>> a = q.Measurement([4.9, 5, 5.1])
        >>> b = q.Measurement([3.1, 3.3, 3.2])
        >>> a.set_covariance(b)  # implicitly calculates the covariance from the two arrays.
        >>> a.get_covariance(b)
        0.005

        See Also
        --------

        :py:func:`~Measurement.set_covariance`

        """

        if isinstance(other, RepeatedMeasurement) and cov is None:
            cov = np.round(q.covariance(self._data, other._data), 14)

        super().set_covariance(other, cov)

    def set_correlation(self, other: Measurement, corr: float = None):
        """Sets the correlation coefficient between two measurements

        For two repeated measurements, if the `corr` argument is not provided, the correlation will
        be calculated from the raw samples taken towards each value. Keep in mind that this does
        not necessarily make sense for every pair of repeated measurements.

        Parameters
        ----------

        other : Measurement
            The other measurement
        corr : float, optional
            The value of the correlation coefficient

        Examples
        --------

        >>> import qexpy as q
        >>> a = q.Measurement([4.9, 5, 5.1])
        >>> b = q.Measurement([3.1, 3.3, 3.2])
        >>> a.set_correlation(b)  # The correlation here is implicitly calculated
        >>> a.get_correlation(b)
        0.5

        """

        if isinstance(other, RepeatedMeasurement) and corr is None:
            corr = np.round(q.correlation(self._data, other._data), 14)

        super().set_correlation(other, corr)
