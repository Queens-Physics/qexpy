"""Defines arithmetic and math operations with ExperimentalValue objects"""

import itertools
import warnings
import numpy as np

from typing import Dict, Callable, List, Set, Generator
from numbers import Real
from qexpy.utils import UndefinedOperationError
from uuid import UUID

import qexpy.utils as utils
import qexpy.settings as sts
import qexpy.settings.literals as lit

import qexpy.data.data as dt  # pylint: disable=cyclic-import

pi, e = np.pi, np.e


def differentiate(formula: "dt.Formula", variable: "dt.ExperimentalValue") -> float:
    """Find the derivative of a formula with respect to a variable"""
    return __differentiator(formula.operator)(variable, *formula.operands)


def get_derivative_propagated_value_and_error(formula: "dt.Formula") -> "dt.ValueWithError":
    """Executes an operation with propagated results using the derivative method

    This is also known as the method of adding quadratures. It also takes into account the
    covariance between measurements if they are specified. It is only valid when the relative
    uncertainties in the quantities are small (less than ~10%)

    """

    # Execute the operation
    result_value = __evaluate_formula(formula)

    # Find measurements that this formula is derived from
    source_meas_ids = __find_source_measurement_ids(formula)  # type: Set[UUID]
    source_measurements = list(dt.get_variable_by_id(_id) for _id in source_meas_ids)

    # Find the quadrature terms
    quads = map(lambda x: (x.error * differentiate(formula, x)) ** 2, source_measurements)

    # Handle covariance between measurements
    covariance_terms = __find_cov_terms(formula, source_measurements)

    # Calculate the result
    result_error = np.sqrt(sum(quads) + sum(covariance_terms))
    if result_error < 0:
        warnings.warn(
            "The error propagated for the given operation is negative. This is likely to be "
            "incorrect! Check your values, maybe you have unphysical covariance.")

    return dt.ValueWithError(result_value, result_error)


def get_monte_carlo_propagated_value_and_error(formula: "dt.Formula") -> "dt.ValueWithError":
    """Executes an operation with propagated results using the Monte-Carlo method

    For each original measurement that the formula is derived from, generate a normally
    distributed random data set with the mean and standard deviation of that measurement.
    Evaluate the formula with each sample. The mean and standard deviation of the results
    are returned as the value and propagated error.

    """

    sample_size = sts.get_settings().monte_carlo_sample_size

    def __generate_random_data_set(measurement_id: UUID, offsets: np.ndarray):
        """Generate random simulated measurements for each MeasuredValue

        This method simply applies the desired mean and standard deviation to the random
        sample set with 0 mean and unit variance

        """
        measurement = dt.get_variable_by_id(measurement_id)

        # The error is used here instead of std even in the case of repeatedly measured
        # values, because the value used is the mean of all measurements, not the value
        # of any single measurement, thus it is more accurate.
        std = measurement.error

        center_value = measurement.value
        return offsets * std + center_value

    def __generate_offset_matrix(measurements):
        """Generates offsets from mean for each measurement

        Each sample set generated has 0 mean and unit variance. Then covariance is applied
        to the set of samples using the Chelosky algorithm.

        Args:
            measurements (List[dt.ExperimentalValue]): a set of measurements to simulate

        Returns:
            A N row times M column matrix where N is the number of measurements to simulate
            and M is the requested sample size for Monte Carlo simulations. Each row of this
            matrix is an array of random values with 0 mean and unit variance

        """
        offset_matrix = np.vstack([np.random.normal(0, 1, sample_size) for _ in measurements])
        offset_matrix = __correlate_random_samples(measurements, offset_matrix)
        return offset_matrix

    def __correlate_random_samples(variables, sample_vector):
        """Uses the Chelosky algorithm to add correlation to random samples

        This method finds the Chelosky decomposition of the correlation matrix of the given
        list of measurements, then applies it to the sample vector.

        The sample vector is a list of random samples, each entry correspond to each variable
        passed in. Each random sample, corresponding to each entry, is an array of random
        numbers with 0 mean and unit variance.

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
                "Fail to generate a physical correlation matrix for the values provided, "
                "using uncorrelated samples instead. Please check that the covariance or "
                "correlation factors assigned to the measurements are physical.")
            return sample_vector

    # Find measurements that this formula is derived from
    source_meas_ids = __find_source_measurement_ids(formula)  # type: Set[UUID]
    source_measurements = list(dt.get_variable_by_id(_id) for _id in source_meas_ids)

    # Each source measurement is assigned a set of normally distributed values with the mean
    # and standard deviation of the measurement's center value and uncertainty.
    data_sets = {}  # type: Dict[UUID, np.ndarray]

    # First generate a sample matrix with 0 mean and unit variance, correlated if applicable
    sample_set = __generate_offset_matrix(source_measurements)
    for _id, sample in zip(source_meas_ids, sample_set):
        # Apply each sample to the desired mean and standard deviation of the measurement
        data_sets[_id] = __generate_random_data_set(_id, sample)

    result_data_set = __evaluate_formula(formula, data_sets)

    # Check the quality of the result data
    if isinstance(result_data_set, np.ndarray):
        # First remove undefined values
        result_data_set = result_data_set[np.isfinite(result_data_set)]

    if len(result_data_set) / sts.get_settings().monte_carlo_sample_size < 0.9:
        # If over 10% of the results calculated are invalid
        warnings.warn(
            "Over 10 percent of the random samples generated for the Monte Carlo simulation "
            "falls outside the domain on which the function is defined. Check the error or "
            "the standard deviation of the measurements passed in, it is possible that the "
            "domain of this function is too narrow compared to the standard deviation of "
            "the measurements. Consider choosing a different error method for this value.")

    # use the standard deviation as the uncertainty and mean as the center value
    return dt.ValueWithError(np.mean(result_data_set), np.std(result_data_set, ddof=1))


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


def propagate_units(formula: "dt.Formula") -> Dict[str, dict]:
    """Calculate the correct units for the formula"""

    operator = formula.operator
    operands = formula.operands

    if all(operand._unit or isinstance(operand, dt.Constant) for operand in operands):
        return utils.operate_with_units(operator, *(operand._unit for operand in operands))

    # If there are non-constant values with unknown units, the units of the final result
    # should also stay unknown. This is to avoid getting non-physical units.
    return {}


@utils.vectorize
def sqrt(x):
    """square root"""
    return __execute(lit.SQRT, x)


@utils.vectorize
def exp(x):
    """e raised to the power of x"""
    return __execute(lit.EXP, x)


@utils.vectorize
def sin(x):
    """sine of x in rad"""
    return __execute(lit.SIN, x)


@utils.vectorize
def sind(x):
    """sine of x in degrees"""
    return sin(x / 180 * np.pi)


@utils.vectorize
def cos(x):
    """cosine of x in rad"""
    return __execute(lit.COS, x)


@utils.vectorize
def cosd(x):
    """cosine of x in degrees"""
    return cos(x / 180 * np.pi)


@utils.vectorize
def tan(x):
    """tan of x in rad"""
    return __execute(lit.TAN, x)


@utils.vectorize
def tand(x):
    """tan of x in degrees"""
    return tan(x / 180 * np.pi)


@utils.vectorize
def sec(x):
    """sec of x in rad"""
    return __execute(lit.SEC, x)


@utils.vectorize
def secd(x):
    """sec of x in degrees"""
    return sec(x / 180 * np.pi)


@utils.vectorize
def csc(x):
    """csc of x in rad"""
    return __execute(lit.CSC, x)


@utils.vectorize
def cscd(x):
    """csc of x in degrees"""
    return csc(x / 180 * np.pi)


@utils.vectorize
def cot(x):
    """cot of x in rad"""
    return __execute(lit.COT, x)


@utils.vectorize
def cotd(x):
    """cot of x in degrees"""
    return cot(x / 180 * np.pi)


@utils.vectorize
def asin(x):
    """arcsine of x"""
    return __execute(lit.ASIN, x)


@utils.vectorize
def acos(x):
    """arccos of x"""
    return __execute(lit.ACOS, x)


@utils.vectorize
def atan(x):
    """arctan of x"""
    return __execute(lit.ATAN, x)


@utils.vectorize
def log(*args):
    """log with a base and power

    If two arguments are provided, returns the log of the second with the first on base
    If only one argument is provided, returns the natural log of that argument

    """
    if len(args) == 2:
        return __execute(lit.LOG, args[0], args[1])
    if len(args) == 1:
        return __execute(lit.LN, args[0])
    raise TypeError("Invalid number of arguments for log().")


@utils.vectorize
def log10(x):
    """log with base 10 for a value"""
    return __execute(lit.LOG10, x)


def __evaluate_formula(formula, samples: Dict[UUID, np.ndarray] = None):
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
        return __evaluate_formula(formula._formula, samples)
    if isinstance(formula, (dt.MeasuredValue, dt.Constant)):
        return formula.value
    if isinstance(formula, Real):
        return float(formula)
    if isinstance(formula, dt.Formula):
        operands = (__evaluate_formula(variable, samples) for variable in formula.operands)
        result = OPERATIONS[formula.operator](*operands)
        return result
    return 0


def __find_source_measurement_ids(formula) -> Set[UUID]:
    """Find IDs of all measurements that the given formula is derived from"""

    if isinstance(formula, dt.Formula):
        return set.union(
            *(__find_source_measurement_ids(operand) for operand in formula.operands))
    if isinstance(formula, dt.MeasuredValue):
        return {formula._id}
    if isinstance(formula, dt.DerivedValue):
        return __find_source_measurement_ids(formula._formula)
    return set()


def __find_cov_terms(formula: "dt.Formula", source_measurements: List) -> Generator:
    """Finds the contributing covariance terms for the quadrature method"""
    for var1, var2 in itertools.combinations(source_measurements, 2):
        cov = dt.get_covariance(var1, var2)
        if cov != 0:
            yield 2 * cov * differentiate(formula, var1) * differentiate(formula, var2)


def __execute(operator: str, *operands) -> "dt.DerivedValue":
    """Execute a math function on numbers or ExperimentalValue instances"""

    # For functions such as sqrt, sin, cos, ..., if a simple real number is passed in, a
    # simple real number should be returned instead of a Constant object.
    if all(isinstance(x, Real) for x in operands):
        return OPERATIONS[operator](*operands)

    try:
        # wrap all operands in ExperimentalValue objects
        values = list(wrap_in_experimental_value(x) for x in operands)
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
    lit.POW: lambda o, x, a: x.value ** (a.value - 1) * (
        a.value * x.derivative(o) + x.value * np.log(x.value) * a.derivative(o)),
    lit.LOG: lambda o, b, x: ((np.log(b.value) * x.derivative(o) / x.value) - (
        b.derivative(o) * np.log(x.value) / b.value)) / (np.log(b.value) ** 2),
    lit.LOG10: lambda o, x: x.derivative(o) / (np.log(10) * x.value),
    lit.LN: lambda o, x: x.derivative(o) / x.value
}
