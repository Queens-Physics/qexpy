"""Utility methods for the data module"""
import warnings

import numpy as np

from numbers import Real

from . import data as dt, datasets as dts  # pylint: disable=cyclic-import

import qexpy.settings.literals as lit
import qexpy.settings as sts
import qexpy.utils as utils

ARRAY_TYPES = np.ndarray, list


class MonteCarloSettings:
    """The object for customizing the Monte Carlo error propagation process"""

    def __init__(self, evaluator):
        self.__evaluator = evaluator
        self.__settings = {
            lit.MONTE_CARLO_SAMPLE_SIZE: 0,
            lit.MONTE_CARLO_STRATEGY: lit.MC_MEAN_AND_STD,
            lit.MONTE_CARLO_CONFIDENCE: 0.68,
            lit.XRANGE: ()
        }

    @property
    def sample_size(self):
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

    def reset_sample_size(self):
        """reset the sample size to default"""
        self.__settings[lit.MONTE_CARLO_SAMPLE_SIZE] = 0

    @property
    def confidence(self):
        """float: The confidence level for choosing the mode of a Monte Carlo distribution"""
        return self.__settings[lit.MONTE_CARLO_CONFIDENCE]

    @confidence.setter
    def confidence(self, new_level: float):
        if not isinstance(new_level, Real):
            raise TypeError("The MC confidence level has to be a number")
        if new_level > 1 or new_level < 0:
            raise ValueError("The MC confidence level has to be a number between 0 and 1")
        self.__settings[lit.MONTE_CARLO_CONFIDENCE] = new_level
        if lit.MC_MODE_AND_CONFIDENCE in self.__evaluator.values:
            self.__evaluator.values.pop(lit.MC_MODE_AND_CONFIDENCE)

    @property
    def xrange(self):
        """tuple: The x-range of the simulation

        This is really the y-range, which means it's the range of the y-values to show,
        but also this is the x-range of the histogram.

        """
        return self.__settings[lit.XRANGE]

    def set_xrange(self, *args):
        """set the range for the monte carlo simulation"""

        if not args:
            self.__settings[lit.XRANGE] = ()
        else:
            new_range = (args[0], args[1]) if len(args) > 1 else args
            utils.validate_xrange(new_range)
            self.__settings[lit.XRANGE] = new_range

        self.__evaluator.values.clear()

    def use_mode_with_confidence(self, confidence=None):
        """Use the mode of the distribution with a confidence coverage for this value"""
        self.__settings[lit.MONTE_CARLO_STRATEGY] = lit.MC_MODE_AND_CONFIDENCE
        if confidence:
            self.confidence = confidence

    def use_mean_and_std(self):
        """Use the mean and std of the distribution for this value"""
        self.__settings[lit.MONTE_CARLO_STRATEGY] = lit.MC_MEAN_AND_STD

    def use_custom_value_and_error(self, value, error):
        """Manually set the value and uncertainty for this quantity

        Sometimes when the distribution is not typical, and you wish to see for yourself what
        the best approach is to choose the center value and uncertainty for this quantity,
        use this method to manually set these values.

        """
        self.__settings[lit.MONTE_CARLO_STRATEGY] = lit.MC_CUSTOM
        if not isinstance(value, Real):
            raise TypeError("Cannot assign a {} to the value!".format(type(value).__name__))
        if not isinstance(error, Real):
            raise TypeError("Cannot assign a {} to the error!".format(type(error).__name__))
        if error < 0:
            raise ValueError("The error must be a positive real number!")
        self.__evaluator.values[self.strategy] = dt.ValueWithError(value, error)

    @property
    def strategy(self):
        """str: the strategy used to extract value and error from a histogram"""
        return self.__settings[lit.MONTE_CARLO_STRATEGY]

    def show_histogram(self, bins=100, **kwargs):  # pragma: no cover
        """Shows the distribution of the Monte Carlo simulated samples"""
        self.__evaluator.show_histogram(bins, **kwargs)

    def samples(self):
        """The raw samples generated in the Monte Carlo simulation

        Sometimes when the distribution is not typical, you might wish to do your own analysis
        with the raw samples generated in the Monte Carlo simulation. This method allows you
        to access a copy of the raw data.

        """
        return self.__evaluator.raw_samples.copy()


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
    except np.linalg.linalg.LinAlgError:  # pragma: no cover
        warnings.warn(
            "Fail to generate a physical correlation matrix for the values provided, using "
            "uncorrelated samples instead. Please check that the covariance or correlation "
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
    raise TypeError(
        "Cannot parse a {} into an ExperimentalValue".format(type(operand).__name__))


def wrap_in_measurement(value, **kwargs) -> "dt.ExperimentalValue":
    """Wraps a value in a Measurement object"""

    if isinstance(value, Real):
        return dt.MeasuredValue(value, 0, **kwargs)
    if isinstance(value, tuple) and len(value) == 2:
        return dt.MeasuredValue(*value, **kwargs)
    if isinstance(value, dt.ExperimentalValue):
        value.name = kwargs.get("name", "")
        value.unit = kwargs.get("unit", "")
        return value

    raise TypeError(
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
