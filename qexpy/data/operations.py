"""Operations for ExperimentalValue objects

This module implements operations with ExperimentalValue objects. This includes basic arithmetic
operations and some basic math functions. Error propagation is implemented for all operations.
This file also contains helper methods for operations with ExperimentalValue objects

"""

import itertools
import warnings
import functools
from typing import Set, Dict, Union, List
from numbers import Real
from uuid import UUID

import numpy as np

import qexpy.utils.utils as utils
import qexpy.utils.units as units
import qexpy.settings.literals as lit
import qexpy.settings.settings as settings
import qexpy.data.data as data  # pylint: disable=cyclic-import

from qexpy.utils.exceptions import InvalidArgumentTypeError

pi, e = np.pi, np.e


def differentiate(formula: "data.Formula", variable: "data.ExperimentalValue") -> float:
    """Find the derivative of a formula with respect to a variable"""
    return _differentiator(formula.operator)(variable, *formula.operands)


def get_derivative_propagated_value_and_error(formula: "data.Formula") -> "data.ValueWithError":
    """Executes an operation with propagated results using the general method

    All error propagation is done with the raw measurements from which this formula is derived.
    This implementation avoids intermediate values, which improves speed.

    """

    # execute the operation
    result_value = evaluate_formula(formula)

    # find measurements that this formula is derived from
    source_measurement_ids = _find_source_measurements(formula)  # type: Set[UUID]
    source_measurements = list(data.get_variable_by_id(_id) for _id in source_measurement_ids)

    # add the quadrature terms
    quadratures = map(lambda x: (x.error * differentiate(formula, x)) ** 2, source_measurements)
    sum_of_quadratures = sum(quadratures)

    # handle covariance between measurements
    sum_of_covariance_terms = 0
    for combination in itertools.combinations(source_measurements, 2):
        var1, var2 = combination  # type: data.MeasuredValue
        covariance = data.get_covariance(var1, var2)
        if covariance != 0:
            # the covariance term is the covariance multiplied by partial derivatives of both operands
            sum_of_covariance_terms += 2 * covariance * differentiate(formula, var1) * differentiate(formula, var2)

    result_error = np.sqrt(sum_of_quadratures + sum_of_covariance_terms)
    if result_error < 0:
        warnings.warn("\nThe error propagated for the given operation is negative. This is likely to be incorrect!\n"
                      "Check your values, maybe you have unphysical covariance.")

    return data.ValueWithError(result_value, result_error)


def get_monte_carlo_propagated_value_and_error(formula: "data.Formula") -> "data.ValueWithError":
    """Calculates the value and error using the Monte Carlo method

    For each original measurement that the formula is derived from, a random normally distributed data
    set is generated with the mean and standard deviation of that measurement. For each sample, the given
    formula is evaluated with the set of simulated measurements. The mean and standard deviation of the
    set of calculated results is returned as the value/error pair.

    """

    sample_size = settings.get_monte_carlo_sample_size()

    def __generate_offset_matrix(measurements: List["data.MeasuredValue"]):
        """Generates offsets from mean for each measurement

        Each random data set generated has 0 mean and unit variance. They will be multiplied by the
        desired standard deviation plus the mean of each measurement to produce the final random data
        set. The reason for this procedure is to apply covariance if applicable.

        The covariance is applied to the set of samples using the Chelosky algorithm. The covariance
        matrix is constructed, and the Chelosky decomposition of the covariance matrix is calculated.
        The Chelosky decomposition can then be used to correlate the vector of random samples with 0
        mean and unit variance.

        Args:
            measurements (list): a set of the source measurements to simulate

        Returns:
            A N row times M column matrix where N is the number of measurements to simulate and M is
            the requested sample size for Monte Carlo simulations. Each row of this matrix is an array
            of random values with 0 mean and unit variance

        """
        raw_offset_matrix = np.vstack([np.random.normal(0, 1, sample_size) for _ in measurements])
        offset_matrix = _correlate_random_samples(measurements, raw_offset_matrix)
        return offset_matrix

    def __generate_random_data_set(measurement_id: UUID, offsets: np.ndarray):
        """Generate random simulated measurements for each MeasuredValue

        This method simply applies the desired mean and standard deviation to the random sample set with
        0 mean and unit variance generated by __generate_offset_matrix defined above.

        """
        measurement = data.get_variable_by_id(measurement_id)
        std = measurement.std if isinstance(measurement, data.RepeatedlyMeasuredValue) else measurement.error
        center_value = measurement.value
        return offsets * std + center_value

    source_measurement_ids = _find_source_measurements(formula)  # type: Set[UUID]
    source_measurements = list(
        data.get_variable_by_id(_id) for _id in source_measurement_ids)  # type: List[data.MeasuredValue]

    # a dictionary object where each source measurement is assigned a set of normally distributed values
    # with the mean and standard deviation of the measurement's center value and uncertainty.
    data_sets = {}  # type: Dict[UUID, utils.ARRAY_TYPES]

    # first generate a matrix of random samples with 0 mean and unit variance, correlated if applicable
    sample_set = __generate_offset_matrix(source_measurements)
    for _id, sample in zip(source_measurement_ids, sample_set):
        # then apply each sample to the desired mean and standard deviation of the measurement
        data_sets[_id] = __generate_random_data_set(_id, sample)

    result_data_set = evaluate_formula(formula, data_sets)

    # check the quality of the result data
    if isinstance(result_data_set, np.ndarray):
        result_data_set = result_data_set[np.isfinite(result_data_set)]  # remove undefined values
    if len(result_data_set) / settings.get_monte_carlo_sample_size() < 0.9:
        # if over 10% of the results calculated is invalid
        warnings.warn("\nOver 10 percent of the random data generated for the Monte Carlo simulation falls outside the"
                      "\ndomain on which the function is defined. Check the uncertainty or the standard deviation of "
                      "\nthe measurements passed in, it is possible that the domain of this function is too narrow "
                      "\ncompared to the standard deviation of the measurements. Consider choosing a different error "
                      "\nmethod for this value.")

    # use the standard deviation as the uncertainty and mean as the center value
    return data.ValueWithError(np.mean(result_data_set), np.std(result_data_set, ddof=1))


def execute(operator: str, *operands: Union[Real, "data.ExperimentalValue"]):
    """Execute an operation

    For functions such as sqrt, sin, cos, ..., if a regular number is passed in, a regular
    number should be returned instead of a Constant object.

    """
    if all(isinstance(x, Real) for x in operands):
        # if all operands involved are just numbers, return the value as a number
        return OPERATIONS[operator](*operands)

    try:
        # wrap all operands in ExperimentalValue objects
        values = list(data.wrap_operand(x) for x in operands)
    except TypeError:
        raise InvalidArgumentTypeError("{}()".format(operator), operands, "real numbers or QExPy defined values")

    # else construct a DerivedValue object
    return data.DerivedValue(data.Formula(operator, list(values)))


def propagate_units(formula: "data.Formula") -> Dict[str, int]:
    """Calculate the correct units for the formula"""

    operator = formula.operator
    operands = formula.operands

    return units.operate_with_units(operator, *(operand._units for operand in operands))


def vectorize(func):
    """vectorize a function if inputs are arrays"""

    @functools.wraps(func)
    def wrapper_vectorize(*args):
        if any(isinstance(arg, utils.ARRAY_TYPES) for arg in args):
            return np.vectorize(func)(*args)
        return func(*args)

    return wrapper_vectorize


@vectorize
def sqrt(x):
    """square root"""
    return execute(lit.SQRT, x)


@vectorize
def exp(x):
    """e raised to the power of x"""
    return execute(lit.EXP, x)


@vectorize
def sin(x):
    """sine of x in rad"""
    return execute(lit.SIN, x)


@vectorize
def sind(x):
    """sine of x in degrees"""
    return sin(x / 180 * np.pi)


@vectorize
def cos(x):
    """cosine of x in rad"""
    return execute(lit.COS, x)


@vectorize
def cosd(x):
    """cosine of x in degrees"""
    return cos(x / 180 * np.pi)


@vectorize
def tan(x):
    """tan of x in rad"""
    return execute(lit.TAN, x)


@vectorize
def tand(x):
    """tan of x in degrees"""
    return tan(x / 180 * np.pi)


@vectorize
def sec(x):
    """sec of x in rad"""
    return execute(lit.SEC, x)


@vectorize
def secd(x):
    """sec of x in degrees"""
    return sec(x / 180 * np.pi)


@vectorize
def csc(x):
    """csc of x in rad"""
    return execute(lit.CSC, x)


@vectorize
def cscd(x):
    """csc of x in degrees"""
    return csc(x / 180 * np.pi)


@vectorize
def cot(x):
    """cot of x in rad"""
    return execute(lit.COT, x)


@vectorize
def cotd(x):
    """cot of x in degrees"""
    return cotd(x / 180 * np.pi)


@vectorize
def asin(x):
    """arcsine of x"""
    return execute(lit.ASIN, x)


@vectorize
def acos(x):
    """arccos of x"""
    return execute(lit.ACOS, x)


@vectorize
def atan(x):
    """arctan of x"""
    return execute(lit.ATAN, x)


@vectorize
def log(*args):
    """log with a base and power

    If two arguments are provided, returns the log of the second with the first on base
    If only one argument is provided, returns the natural log of that argument

    """
    if len(args) == 2:
        return execute(lit.LOG, args[0], args[1])
    if len(args) == 1:
        return execute(lit.LN, args[0])
    raise TypeError("Invalid number of arguments for log().")


@vectorize
def log10(x):
    """log with base 10 for a value"""
    return execute(lit.LOG10, x)


def evaluate_formula(formula, samples: Dict[UUID, np.ndarray] = None) -> Union[np.ndarray, float]:
    """Evaluates a Formula with original values of measurements or sample values

    This method has an option of passing in a "samples" parameter where the value for each
    MeasuredValue can be specified. The keys of this dictionary object are the unique IDs of
    the values. This feature is used for Monte Carlo simulations where a formula needs to be
    evaluated for a large set of values

    The "sample" of a measurement can either be a single number or an np.ndarray. In the latter
    case, the return value will also be an np.ndarray. This improves the computing speed
    significantly because it takes advantage of numpy's vectorization of the arrays

    Args:
        formula (Union[data.Formula, data.ExperimentalValue])
        samples (Dict[UUID, np.ndarray])

    """
    np.seterr(all="ignore")  # ignore runtime warnings
    if samples is not None and isinstance(formula, data.MeasuredValue) and formula._id in samples:
        # use the value in the sample instead of its original value if specified
        return samples[formula._id]
    if isinstance(formula, data.DerivedValue):
        return evaluate_formula(formula._formula, samples)
    if isinstance(formula, (data.MeasuredValue, data.Constant)):
        return formula.value
    if isinstance(formula, Real):
        return float(formula)
    if isinstance(formula, data.Formula):
        operands = (evaluate_formula(variable, samples) for variable in formula.operands)
        result = OPERATIONS[formula.operator](*operands)
        return result
    return 0


def _correlate_random_samples(variables: List["data.MeasuredValue"], sample_vector: np.ndarray):
    """Uses the Chelosky Decomposition algorithm to correlate samples for the Monte Carlo method

    This method finds the Chelosky decomposition of the correlation matrix of the given list of measurements,
    then applies it to the sample vector. This adds correlation to the samples.

    The sample vector is a list of random samples, each entry correspond to each variable passed in. Each
    random sample corresponding each entry is a list of random numbers with unit variance and 0 mean. The
    result will later be applied the desired mean and std of the variables in a Monte Carlo simulation

    """
    correlation_matrix = np.array([[data.get_correlation(row, col) for col in variables] for row in variables])
    try:
        chelosky_decomposition = np.linalg.cholesky(correlation_matrix)
        result_vector = np.dot(chelosky_decomposition, sample_vector)
        return result_vector
    except np.linalg.linalg.LinAlgError:
        warnings.warn("\nFail to generate a physical correlation matrix for the values provided, using uncorrelated "
                      "\nsamples for this simulation. Please check that the correlation or covariance factors you"
                      "\nhave given to the measurements are physical. ")
        return sample_vector


def _find_source_measurements(formula: Union["data.Formula", "data.ExperimentalValue"]) -> Set[UUID]:
    """Returns a set of IDs for MeasuredValue objects that the given formula is derived from"""

    if isinstance(formula, data.Formula):
        source_measurements = set()
        for operand in formula.operands:
            source_measurements.update(_find_source_measurements(operand))
        return source_measurements
    if isinstance(formula, data.MeasuredValue):
        return {formula._id}
    if isinstance(formula, data.DerivedValue):
        return _find_source_measurements(formula._formula)
    return set()


def _differentiator(operator: str):
    """Gets the derivative formula for an operator

    The differentiator is a lambda function calculates the derivative of an expression with
    respect to a target value. The first argument is a reference to the target value, and the
    rest of the arguments are the operands for this operation

    Usage:
        differentiator("MUL")(x, a, b) would return the derivative of the expression "a * b"
        with respect to the value x

    Args:
        operator (str): the operator of this expression (use string literals to avoid typos)

    """
    return DIFFERENTIATORS[operator]


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

# the usage of the differentiators are documented under the method differentiator
DIFFERENTIATORS = {
    lit.NEG: lambda other, x: -x.derivative(other),
    lit.ADD: lambda other, a, b: a.derivative(other) + b.derivative(other),
    lit.SUB: lambda other, a, b: a.derivative(other) - b.derivative(other),
    lit.MUL: lambda other, a, b: a.derivative(other) * b.value + b.derivative(other) * a.value,
    lit.DIV: lambda other, a, b: (b.value * a.derivative(other) - a.value * b.derivative(other)) / (b.value ** 2),
    lit.SQRT: lambda other, x: 1 / 2 / np.sqrt(x.value) * x.derivative(other),
    lit.EXP: lambda other, x: np.exp(x.value) * x.derivative(other),
    lit.SIN: lambda other, x: np.cos(x.value) * x.derivative(other),
    lit.COS: lambda other, x: -np.sin(x.value) * x.derivative(other),
    lit.TAN: lambda other, x: 1 / (np.cos(x.value)) ** 2 * x.derivative(other),
    lit.ASIN: lambda other, x: 1 / np.sqrt(1 - x.value ** 2) * x.derivative(other),
    lit.ACOS: lambda other, x: -1 / np.sqrt(1 - x.value ** 2) * x.derivative(other),
    lit.ATAN: lambda other, x: 1 / (1 + x.value ** 2) * x.derivative(other),
    lit.SEC: lambda other, x: np.tan(x.value) / np.cos(x.value) * x.derivative(other),
    lit.CSC: lambda other, x: -1 / (np.tan(x.value) * np.sin(x.value)) * x.derivative(other),
    lit.COT: lambda other, x: -1 / (np.sin(x.value) ** 2) * x.derivative(other),
    lit.POW: lambda other, x, a: x.value ** (a.value - 1) * (
        a.value * x.derivative(other) + x.value * np.log(x.value) * a.derivative(other)),
    lit.LOG: lambda other, base, x: ((np.log(base.value) * x.derivative(other) / x.value) - (
        base.derivative(other) * np.log(x.value) / base.value)) / (np.log(base.value) ** 2),
    lit.LOG10: lambda other, x: x.derivative(other) / (np.log(10) * x.value),
    lit.LN: lambda other, x: x.derivative(other) / x.value
}
