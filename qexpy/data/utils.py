"""Utility methods for the data module"""
import warnings

import numpy as np

from numbers import Real

from . import data as dt, datasets as dts  # pylint: disable=cyclic-import

import qexpy.settings.literals as lit
import qexpy.settings as sts

ARRAY_TYPES = np.ndarray, list


class MonteCarloSettings:
    """The object for customizing the Monte Carlo error propagation process"""

    def __init__(self, evaluator):
        self.__evaluator = evaluator
        self.__settings = {
            lit.MONTE_CARLO_SAMPLE_SIZE: 0,
            lit.MONTE_CARLO_STRATEGY: lit.MC_MEAN_AND_STD,
            lit.MONTE_CARLO_CONFIDENCE: 0.68
        }

    @property
    def sample_size(self) -> int:
        """int: The Monte Carlo sample size"""
        default_size = sts.get_settings().monte_carlo_sample_size
        set_size = self.__settings[lit.MONTE_CARLO_SAMPLE_SIZE]
        return set_size if set_size else default_size

    @sample_size.setter
    def sample_size(self, new_size: int):
        if not isinstance(new_size, int) or new_size < 0:
            raise ValueError("The sample size has to be a positive integer")
        self.__settings[lit.MONTE_CARLO_SAMPLE_SIZE] = new_size
        self.__evaluator.clear()

    @property
    def confidence(self):
        """float: The confidence level for choosing the mode of a Monte Carlo distribution"""
        return self.__settings[lit.MONTE_CARLO_CONFIDENCE]

    @confidence.setter
    def confidence(self, new_level: float):
        if not isinstance(new_level, float) or new_level > 1 or new_level < 0:
            raise ValueError("The MC confidence level has to be a number between 0 and 1")
        self.__settings[lit.MONTE_CARLO_CONFIDENCE] = new_level

    def use_mode_with_confidence(self):
        """Sets the strategy to mode and uncertainty from confidence coverage"""
        self.strategy = lit.MC_MODE_AND_CONFIDENCE

    def use_mean_and_std(self):
        """Sets the strategy to use mean and std of the distribution"""
        self.strategy = lit.MC_MEAN_AND_STD

    @property
    def strategy(self):
        """str: the strategy used to extract value and error from a histogram"""
        return self.__settings[lit.MONTE_CARLO_STRATEGY]

    @strategy.setter
    def strategy(self, new_strategy: str):
        if new_strategy not in [lit.MC_MEAN_AND_STD, lit.MC_MODE_AND_CONFIDENCE]:
            raise ValueError("Invalid strategy string")
        self.__settings[lit.MONTE_CARLO_STRATEGY] = new_strategy


def generate_offset_matrix(measurements, sample_size):
    """Generates offsets from mean for each measurement

    Each sample set generated has 0 mean and unit variance. Then covariance is applied to the
    set of samples using the Chelosky algorithm.

    Args:
        measurements (List[dt.ExperimentalValue]): a set of measurements to simulate
        sample_size (int): the size of the samples

    Returns:
        A N row times M column matrix where N is the number of measurements to simulate and
        M is the requested sample size for Monte Carlo simulations. Each row of this matrix
        is an array of random values with 0 mean and unit variance

    """

    offset_matrix = np.vstack(
        [np.random.normal(0, 1, sample_size) for _ in measurements])
    offset_matrix = correlate_samples(measurements, offset_matrix)
    return offset_matrix


def correlate_samples(variables, sample_vector):
    """Uses the Chelosky algorithm to add correlation to random samples

    This method finds the Chelosky decomposition of the correlation matrix of the given list
    of measurements, then applies it to the sample vector.

    The sample vector is a list of random samples, each entry correspond to each variable
    passed in. Each random sample, corresponding to each entry, is an array of random numbers
    with 0 mean and unit variance.

    Args:
        variables (List[dt.ExperimentalValue]): the source measurements
        sample_vector (np.ndarray): the list of random samples to apply correlation to

    Returns:
        The same list sample vector with correlation applied

    """

    corr_matrix = np.array(
        [[dt.get_correlation(row, col) for col in variables] for row in variables])
    if np.count_nonzero(corr_matrix - np.diag(np.diagonal(corr_matrix))) == 0:
        return sample_vector  # if no correlations are present

    try:
        chelosky_decomposition = np.linalg.cholesky(corr_matrix)
        result_vector = np.dot(chelosky_decomposition, sample_vector)
        return result_vector
    except np.linalg.linalg.LinAlgError:
        warnings.warn(
            "Fail to generate a physical correlation matrix for the values provided, using "
            "uncorrelated samples instead. Please check that the covariance or  correlation "
            "factors assigned to the measurements are physical.")
        return sample_vector


def wrap_in_experimental_value(operand) -> "dt.ExperimentalValue":
    """Wraps a variable in an ExperimentalValue object

    Wraps single numbers in a Constant, number pairs in a MeasuredValue. If the argument
    is already an ExperimentalValue instance, return directly. If the

    """

    if isinstance(operand, Real):
        return dt.Constant(operand)
    if isinstance(operand, dt.ExperimentalValue):
        return operand
    if isinstance(operand, tuple) and len(operand) == 2:
        return dt.MeasuredValue(operand[0], operand[1])
    raise TypeError("Cannot parse a {} into an ExperimentalValue".format(type(operand)))


def wrap_in_measurement(value, **kwargs) -> "dt.ExperimentalValue":
    """Wraps a value in a Measurement object"""

    if isinstance(value, Real):
        return dt.MeasuredValue(value, 0, **kwargs)
    if isinstance(value, tuple) and len(value) == 2:
        return dt.MeasuredValue(*value, **kwargs)
    if isinstance(value, dt.ExperimentalValue):
        return value

    raise ValueError(
        "Elements of a MeasurementArray must be convertible to an ExperimentalValue")


def wrap_in_value_array(operand, **kwargs) -> np.ndarray:
    """Wraps input in an ExperimentalValueArray"""

    # wrap array times in numpy arrays
    if isinstance(operand, dts.ExperimentalValueArray):
        return operand
    if isinstance(operand, ARRAY_TYPES):
        return np.asarray([wrap_in_measurement(value, **kwargs) for value in operand])

    # wrap single value times in array
    return np.asarray([wrap_in_measurement(operand, **kwargs)])
