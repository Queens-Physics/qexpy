"""Defines arithmetic and math operations with ExperimentalValue objects"""

import itertools
import warnings
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Callable, List, Set, Generator
from numbers import Real
from collections import OrderedDict

from qexpy.utils import UndefinedOperationError, UndefinedActionError
from uuid import UUID

import qexpy.utils as utils
import qexpy.settings as sts
import qexpy.settings.literals as lit

from . import data as dt  # pylint: disable=cyclic-import
from . import datasets as dts  # pylint: disable=cyclic-import
from . import utils as dut

pi, e = np.pi, np.e

ARRAY_TYPES = np.ndarray, list


class Evaluator(ABC):
    """Used to calculate the value and uncertainty of a derived value"""

    @abstractmethod
    def evaluate(self, formula: "dt.Formula") -> "dt.ValueWithError":
        """Evaluates a formula with the proper error method"""
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        """Clears the buffered results in this evaluator"""
        raise NotImplementedError


class DerivativeEvaluator(Evaluator):
    """The calculator that uses the derivative method to propagate errors"""

    def __init__(self):
        self.result = ()  # type: dt.ValueWithError
        self.measurements = []
        self.error_contributions = []

    def evaluate(self, formula: "dt.Formula") -> "dt.ValueWithError":
        if not self.result:
            self.result = self.__evaluate(formula)
        return self.result

    def clear(self):
        self.result = ()
        self.measurements = []
        self.error_contributions = []

    def __evaluate(self, formula: "dt.Formula"):
        """Executes an operation with propagated results using the derivative method

        This is also known as the method of adding quadratures. It also takes into account
        the covariance between measurements if they are specified. It is only valid when the
        relative uncertainties in the quantities are small (less than ~10%)

        """

        # Execute the operation
        result_value = _evaluate_formula(formula)

        # Find measurements that this formula is derived from
        source_meas_ids = _find_source_measurement_ids(formula)  # type: Set[UUID]
        sources = list(dt.get_variable_by_id(_id) for _id in source_meas_ids)

        # record source measurements
        self.measurements = sources

        # Find the quadrature terms
        quads = list(map(lambda x: (x.error * differentiate(formula, x)) ** 2, sources))

        # Handle covariance between measurements
        covariance_terms = DerivativeEvaluator.__find_cov_terms(formula, sources)

        # Calculate the result
        result_sums = sum(quads) + sum(covariance_terms)
        if result_sums < 0:  # pragma: no cover
            raise UndefinedActionError(
                "The error propagated for the given operation is negative. This is likely "
                "to be incorrect! Check your values, maybe you have unphysical covariance.")

        # record error contributions
        if result_sums > 0:
            self.error_contributions = np.array([quad / result_sums for quad in quads])
        else:
            self.error_contributions = np.zeros(len(quads))

        result_error = np.sqrt(result_sums)

        return dt.ValueWithError(result_value, result_error)

    @staticmethod
    def __find_cov_terms(_formula: "dt.Formula", _measurements: List) -> Generator:
        """Finds the contributing covariance terms for the quadrature method"""
        for var1, var2 in itertools.combinations(_measurements, 2):
            corr = dt.get_correlation(var1, var2)
            # Re-calculate the covariance between two measurements, because in the case of
            # repeated measurements, sometimes the covariance is calculated from the raw
            # measurement array, which is closely coupled with the standard deviation of the
            # raw samples. This is misleading because with repeated measurements, we use the
            # error on the mean, not the standard deviation of the raw measurements, as the
            # uncertainty on the quantity. Essentially, with repeatedly measured values, we
            # are ignoring the array of raw measurements, and treating its value and error
            # as the mean and standard deviation just like we would with any other single
            # measurements. This would make the most physical sense.
            cov = corr * var1.error * var2.error
            if cov != 0:
                yield 2 * cov * differentiate(_formula, var1) * differentiate(_formula, var2)


class MonteCarloEvaluator(Evaluator):
    """The calculator that uses the Monte Carlo method to propagate errors"""

    def __init__(self):
        self.raw_samples = np.empty(0)
        self.values = {}
        self.settings = dut.MonteCarloSettings(self)

    @property
    def samples(self):
        """np.ndarray: the raw samples of this simulation"""
        if not self.settings.xrange:
            return self.raw_samples
        xrange = self.settings.xrange
        return np.ma.masked_outside(self.raw_samples, xrange[0], xrange[1], copy=False)

    def evaluate(self, formula: "dt.Formula") -> "dt.ValueWithError":

        self.regenerate_samples(formula)

        strategy = self.settings.strategy

        if strategy == lit.MC_CUSTOM not in self.values:
            strategy = lit.MC_MEAN_AND_STD
            self.settings.use_mean_and_std()

        if strategy == lit.MC_MEAN_AND_STD not in self.values:
            result = dt.ValueWithError(np.mean(self.samples), np.std(self.samples, ddof=1))
            self.values[strategy] = result

        if strategy == lit.MC_MODE_AND_CONFIDENCE not in self.values:
            n, bins = np.histogram(self.samples, bins=100)
            value, error = utils.find_mode_and_uncertainty(n, bins, self.settings.confidence)
            self.values[strategy] = dt.ValueWithError(value, error)

        return self.values[strategy]

    def regenerate_samples(self, formula):
        """generates raw samples if none is present"""
        if not self.raw_samples.size:
            self.raw_samples = self.__compute_samples(formula)

    def clear(self):
        self.raw_samples = np.empty(0)
        self.values.clear()

    def show_histogram(self, bins=100, **kwargs):  # pragma: no cover
        """Shows the distribution of the Monte Carlo simulated samples"""

        samples = self.samples
        if "range" in kwargs:
            xrange = kwargs.pop('range')
            samples = np.ma.masked_outside(samples, xrange[0], xrange[1], copy=False)

        import matplotlib.pyplot as plt
        n, edges, _ = plt.hist(samples, bins=bins, **kwargs)

        if self.settings.strategy == lit.MC_MODE_AND_CONFIDENCE:
            value, error = utils.find_mode_and_uncertainty(n, edges, self.settings.confidence)
            value_label = "mode = {:.2f}".format(value)
            plt.title("MC with {:.1f}% confidence".format(self.settings.confidence * 100))
        else:
            value, error = np.mean(samples), np.std(samples, ddof=1)
            value_label = "mean = {:.2f}".format(value)
            plt.title("MC highlighting mean and standard deviation")

        plt.plot([value, value], [0, max(n)], "r", label=value_label)

        low, high = value - error, value + error
        plt.plot([low, low], [0, max(n)], "r--", label="low bound = {:.2f}".format(low))
        plt.plot([high, high], [0, max(n)], "r--", label="high bound = {:.2f}".format(high))

        plt.legend()
        plt.show()

    def __compute_samples(self, formula: "dt.Formula") -> np.ndarray:
        """Executes an operation with propagated results using the Monte-Carlo method

        For each original measurement that the formula is derived from, generate a normally
        distributed random data set with the mean and standard deviation of that measurement.
        Evaluate the formula with each sample, and return the final sample set

        """

        sample_size = self.settings.sample_size

        # Find measurements that this formula is derived from
        source_meas_ids = _find_source_measurement_ids(formula)  # type: Set[UUID]
        source_measurements = list(dt.get_variable_by_id(_id) for _id in source_meas_ids)

        # Each source measurement is assigned a set of normally distributed values with the
        # mean and standard deviation of the measurement's center value and uncertainty.
        data_sets = {}  # type: Dict[UUID, np.ndarray]

        # Generate a sample matrix with 0 mean and unit variance, correlated if applicable
        sample_set = dut.generate_offset_matrix(source_measurements, sample_size)
        for _id, sample in zip(source_meas_ids, sample_set):
            # Apply each sample to the desired mean and standard deviation of the measurement
            data_sets[_id] = _generate_random_data_set(_id, sample)

        result_data_set = _evaluate_formula(formula, data_sets)

        # Check the quality of the result data
        assert isinstance(result_data_set, np.ndarray)

        # First remove undefined values
        result_data_set = result_data_set[np.isfinite(result_data_set)]

        if len(result_data_set) / sts.get_settings().monte_carlo_sample_size < 0.9:
            # If over 10% of the results calculated are invalid
            warnings.warn(
                "Over 10 percent of the random samples generated for the Monte Carlo "
                "simulation falls outside the domain on which the function is defined. "
                "Check the error or the standard deviation of the measurements passed in, "
                "it is possible that the domain of this function is too narrow compared to "
                "the standard deviation of the measurements.")

        # return the result data set
        return result_data_set


def differentiate(formula: "dt.Formula", variable: "dt.ExperimentalValue") -> float:
    """Find the derivative of a formula with respect to a variable"""
    return __differentiator(formula.operator)(variable, *formula.operands)


def propagate_units(formula: "dt.Formula") -> Dict[str, dict]:
    """Calculate the correct units for the formula"""

    operator = formula.operator
    operands = formula.operands

    # the power operator is different, treat separately
    if operator == lit.POW and isinstance(operands[1], dt.Constant):
        power = operands[1].value
        return OrderedDict([
            (unit, count * power) for unit, count in operands[0]._unit.items()])

    if all(operand._unit or isinstance(operand, dt.Constant) for operand in operands):
        return utils.operate_with_units(operator, *(operand._unit for operand in operands))

    # If there are non-constant values with unknown units, the units of the final result
    # should also stay unknown. This is to avoid getting non-physical units.
    return {}


@utils.vectorize
def sqrt(x):
    """square root"""
    return _execute(lit.SQRT, x)


@utils.vectorize
def exp(x):
    """e raised to the power of x"""
    return _execute(lit.EXP, x)


@utils.vectorize
def sin(x):
    """sine of x in rad"""
    return _execute(lit.SIN, x)


@utils.vectorize
def sind(x):
    """sine of x in degrees"""
    return sin(x / 180 * np.pi)


@utils.vectorize
def cos(x):
    """cosine of x in rad"""
    return _execute(lit.COS, x)


@utils.vectorize
def cosd(x):
    """cosine of x in degrees"""
    return cos(x / 180 * np.pi)


@utils.vectorize
def tan(x):
    """tan of x in rad"""
    return _execute(lit.TAN, x)


@utils.vectorize
def tand(x):
    """tan of x in degrees"""
    return tan(x / 180 * np.pi)


@utils.vectorize
def sec(x):
    """sec of x in rad"""
    return _execute(lit.SEC, x)


@utils.vectorize
def secd(x):
    """sec of x in degrees"""
    return sec(x / 180 * np.pi)


@utils.vectorize
def csc(x):
    """csc of x in rad"""
    return _execute(lit.CSC, x)


@utils.vectorize
def cscd(x):
    """csc of x in degrees"""
    return csc(x / 180 * np.pi)


@utils.vectorize
def cot(x):
    """cot of x in rad"""
    return _execute(lit.COT, x)


@utils.vectorize
def cotd(x):
    """cot of x in degrees"""
    return cot(x / 180 * np.pi)


@utils.vectorize
def asin(x):
    """arcsine of x"""
    return _execute(lit.ASIN, x)


@utils.vectorize
def acos(x):
    """arccos of x"""
    return _execute(lit.ACOS, x)


@utils.vectorize
def atan(x):
    """arctan of x"""
    return _execute(lit.ATAN, x)


@utils.vectorize
def log(*args):
    """log with a base and power

    If two arguments are provided, returns the log of the second with the first on base
    If only one argument is provided, returns the natural log of that argument

    """
    if len(args) == 2:
        return _execute(lit.LOG, args[0], args[1])
    if len(args) == 1:
        return _execute(lit.LN, args[0])
    raise TypeError("Invalid number of arguments for log().")


@utils.vectorize
def log10(x):
    """log with base 10 for a value"""
    return _execute(lit.LOG10, x)


def mean(array):
    """The mean of an array"""
    if isinstance(array, dts.ExperimentalValueArray):
        return array.mean()
    return np.mean(array)


def sum_(array):  # avoid built-in function "sum"
    """The sum of an array"""
    if isinstance(array, dts.ExperimentalValueArray):
        return array.sum()
    return np.sum(array)


def std(array, ddof=1):
    """The standard deviation of an array"""
    if isinstance(array, dts.ExperimentalValueArray):
        return array.std()
    return np.std(array, ddof=ddof)


def _evaluate_formula(formula, samples: Dict[UUID, np.ndarray] = None):
    """Evaluates a Formula with original values of measurements or sample values

    This function evaluates the formula with the original measurements by default. If a set
    of samples are passed in, the formula will be evaluated with the sample values.

    Args:
        formula (Union[dt.Formula, dt.ExperimentalValue]): the formula to be evaluated
        samples (Dict): an np.ndarray of samples assigned to each source measurements's ID.

    """
    np.seterr(all="ignore")  # ignore runtime warnings
    if samples and isinstance(formula, dt.MeasuredValue) and formula._id in samples:
        # Use the value in the sample instead of its original value if specified
        return samples[formula._id]
    if isinstance(formula, dt.DerivedValue):
        return _evaluate_formula(formula._formula, samples)
    if isinstance(formula, (dt.MeasuredValue, dt.Constant)):
        return formula.value

    operands = (_evaluate_formula(variable, samples) for variable in formula.operands)
    return OPERATIONS[formula.operator](*operands)


def _find_source_measurement_ids(formula) -> Set[UUID]:
    """Find IDs of all measurements that the given formula is derived from"""

    if isinstance(formula, dt.Formula):
        return set.union(
            *(_find_source_measurement_ids(operand) for operand in formula.operands))
    if isinstance(formula, dt.MeasuredValue):
        return {formula._id}
    if isinstance(formula, dt.DerivedValue):
        return _find_source_measurement_ids(formula._formula)
    return set()


def _generate_random_data_set(measurement_id: UUID, offsets: np.ndarray):
    """Generate random simulated measurements for each MeasuredValue

    This method simply applies the desired mean and standard deviation to the random
    sample set with 0 mean and unit variance

    """

    measurement = dt.get_variable_by_id(measurement_id)

    # The error is used here instead of std even in the case of repeatedly measured values,
    # because the value used is the mean of all measurements, not the value of any single
    # measurement, thus it is more accurate.
    _std = measurement.error

    center_value = measurement.value
    return offsets * _std + center_value


def _execute(operator: str, *operands) -> "dt.DerivedValue":
    """Execute a math function on numbers or ExperimentalValue instances"""

    # For functions such as sqrt, sin, cos, ..., if a simple real number is passed in, a
    # simple real number should be returned instead of a Constant object.
    if all(isinstance(x, Real) for x in operands):
        return OPERATIONS[operator](*operands)

    try:
        # wrap all operands in ExperimentalValue objects
        values = list(dut.wrap_in_experimental_value(x) for x in operands)
    except TypeError:
        raise UndefinedOperationError(operator, operands, "real numbers")

    # Construct a DerivedValue object with the operator and operands
    return dt.DerivedValue(dt.Formula(operator, list(values)))


def __differentiator(operator: str) -> Callable:
    """Gets the derivative formula for an operator

    The differentiator is a lambda function that calculates the derivative of an expression
    with respect to a target value. The first argument is a reference to the target value,
    and the rest of the arguments are the operands for this operation.

    Usage:
        differentiator("MUL")(x, a, b) would return the derivative of the expression "a * b"
        with respect to the value x

    Args:
        operator (str): the operator of this expression

    """
    return DIFFERENTIATORS[operator]


def __pow_differentiator(o, x, a):
    """The differentiator for a power function is a little more complicated

    This is added because the derivative of f(x)^g(x) includes a term that involves taking
    the log of f(x). This should not be necessary if g(x) is a constant. In some cases where
    f(x) is smaller than 0, the log would return nan, which makes the entire expression nan.
    This should not need to happen if d/dx of g(x) is 0, which should eliminate the nan term.
    Since in Python nan * 0 returns nan instead of 0, this helper function is written so that
    this can happen.

    """
    leading = x.value ** (a.value - 1)
    first = a.value * x.derivative(o)
    second = x.value * np.log(x.value) * a.derivative(o) if a.derivative(o) != 0 else 0
    return leading * (first + second)


OPERATIONS = {
    lit.NEG: lambda x: -x,
    lit.ADD: lambda a, b: a + b,
    lit.SUB: lambda a, b: a - b,
    lit.MUL: lambda a, b: a * b,
    lit.DIV: lambda a, b: a / b,
    lit.SQRT: np.sqrt,
    lit.EXP: np.exp,
    lit.SIN: np.sin,
    lit.COS: np.cos,
    lit.TAN: np.tan,
    lit.ASIN: np.arcsin,
    lit.ACOS: np.arccos,
    lit.ATAN: np.arctan,
    lit.SEC: lambda x: 1 / np.cos(x),
    lit.CSC: lambda x: 1 / np.sin(x),
    lit.COT: lambda x: 1 / np.tan(x),
    lit.POW: lambda x, a: x ** a,
    lit.LOG: lambda base, x: np.log(x) / np.log(base),
    lit.LOG10: np.log10,
    lit.LN: np.log
}

DIFFERENTIATORS = {
    lit.NEG: lambda o, x: -x.derivative(o),
    lit.ADD: lambda o, a, b: a.derivative(o) + b.derivative(o),
    lit.SUB: lambda o, a, b: a.derivative(o) - b.derivative(o),
    lit.MUL: lambda o, a, b: a.derivative(o) * b.value + b.derivative(o) * a.value,
    lit.DIV: lambda o, a, b: (b.value * a.derivative(o) - a.value * b.derivative(o)) / (
        b.value ** 2),
    lit.SQRT: lambda o, x: 1 / 2 / np.sqrt(x.value) * x.derivative(o),
    lit.EXP: lambda o, x: np.exp(x.value) * x.derivative(o),
    lit.SIN: lambda o, x: np.cos(x.value) * x.derivative(o),
    lit.COS: lambda o, x: -np.sin(x.value) * x.derivative(o),
    lit.TAN: lambda o, x: 1 / (np.cos(x.value)) ** 2 * x.derivative(o),
    lit.ASIN: lambda o, x: 1 / np.sqrt(1 - x.value ** 2) * x.derivative(o),
    lit.ACOS: lambda o, x: -1 / np.sqrt(1 - x.value ** 2) * x.derivative(o),
    lit.ATAN: lambda o, x: 1 / (1 + x.value ** 2) * x.derivative(o),
    lit.SEC: lambda o, x: np.tan(x.value) / np.cos(x.value) * x.derivative(o),
    lit.CSC: lambda o, x: -1 / (np.tan(x.value) * np.sin(x.value)) * x.derivative(o),
    lit.COT: lambda o, x: -1 / (np.sin(x.value) ** 2) * x.derivative(o),
    lit.POW: __pow_differentiator,  # see function definition and comment above
    lit.LOG: lambda o, b, x: ((np.log(b.value) * x.derivative(o) / x.value) - (
        b.derivative(o) * np.log(x.value) / b.value)) / (np.log(b.value) ** 2),
    lit.LOG10: lambda o, x: x.derivative(o) / (np.log(10) * x.value),
    lit.LN: lambda o, x: x.derivative(o) / x.value
}
