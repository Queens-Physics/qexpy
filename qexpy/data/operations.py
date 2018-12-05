"""Operations for ExperimentalValue objects

This file implements the different operation overloads on ExperimentalValue objects. The
operations will return a DerivedValue object, with errors propagated properly. They are helper
methods which will be used in data.py

The operator methods all return the error propagated results used to construct DerivedValue
objects defined in data.py. For more details please refer to the default constructor of the
DerivedValue class

"""

import itertools
import numbers
import warnings
from typing import Set, Dict, Union
from uuid import UUID

import numpy as np

import qexpy.settings.literals as lit
import qexpy.settings.settings as settings
from . import data


def differentiator(operator):
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
    return differentiators[operator]


def execute(operator, operands):
    """Executes an operation with propagated results"""
    # finds the lambda function for this operation
    operation = operations[operator]

    # extract the values of the operands
    values = map(lambda x: x.value, operands)

    # execute the operation and propagate values
    result_value = operation(*values)
    result_error = propagate_error_derivative(operator, operands)

    # By default only the value-error pair propagated with the derivative method is calculated.
    # The Monte Carlo method will be called when the user specifically sets the error method
    # to Monte Carlo, and only when the user requests it from a value
    return {
        lit.DERIVATIVE_PROPAGATED: data.ValueWithError(result_value, result_error)
    }


def execute_without_wrapping(operator, *operands):
    """Execute an operation without wrapping real values into Constant objects

    For functions such as sqrt, sin, cos, ..., if a regular number is passed in, a regular
    number should be returned instead of a Constant object.

    """
    if all(isinstance(x, numbers.Real) for x in operands):
        # if all operands involved are just numbers, return the value as a number
        return operations[operator](*operands)

    # technically at this point, since the operand passed in is definitely not a number,
    # there is no need for this "_wrap_operand" call. However, this call is added as an
    # type check
    values = map(data._wrap_operand, operands)

    # else construct a DerivedValue object
    return data.DerivedValue({
        lit.OPERATOR: operator,
        lit.OPERANDS: list(values)
    })


def propagate_error_derivative(operator, operands):
    # the function for calculating derivatives, see docstring under the method differentiator
    diff = differentiator(operator)

    # calculate the derivative of this expression with respect to each operand, multiplied
    # by the error of that operand, raised to the power of two, and summed
    quadratures = map(lambda operand: (operand.error * diff(operand, *operands)) ** 2, operands)

    # traverse the formula tree of this expression and find all MeasuredValue objects at the leaves
    # of this formula tree.
    source_measurements = set.union(*(_find_source_measurements(value) for value in operands))  # type: Set[UUID]
    covariance_term_sum = 0

    # For each pair of MeasuredValue objects, the covariance times partial derivative of the
    # formula with respect to each MeasuredValue is calculated and will be added to the final
    # sum. This implementation avoids propagating covariance for DerivedValue objects since
    # everything is calculated with the raw measurements associated with this formula
    for combination in itertools.combinations(source_measurements, 2):
        var1 = data.ExperimentalValue._register[combination[0]]  # type: data.ExperimentalValue
        var2 = data.ExperimentalValue._register[combination[1]]  # type: data.ExperimentalValue
        covariance = data.get_covariance(var1, var2)
        if covariance != 0:
            # the covariance term is the covariance multiplied by partial derivatives of both operands
            covariance_term_sum += covariance * diff(var1, *operands) * diff(var2, *operands)

    error = np.sqrt(sum(quadratures) + covariance_term_sum)
    if error >= 0:
        return error  # make sure that the uncertainty is positive
    else:
        warnings.warn("The error propagated for the given operation is negative. This is likely to "
                      "be incorrect! Check your values, maybe you have an unphysical covariance.")


def get_monte_carlo_propagated_value(value):
    """Calculates the value and error using the Monte Carlo method

    For each original measurement that the value is derived from, a random data set is generated with
    the mean and standard deviation of that measurement. For each sample, the formula of this value
    is evaluated with that set of simulated measurements. This generates an array of calculated values
    with the mean and standard deviation representing the value and propagated uncertainty of the
    final result.

    Args:
        value (DerivedValue): the derived value to find Monte Carlo propagated values for

    """

    sample_size = settings.get_monte_carlo_sample_size()

    def __generate_offset_matrix(measurement_ids: Set[UUID]):
        """Generates random offsets from mean for each measurement

        Each random data set generated has 0 mean and unit variance. They will be multiplied by the
        desired standard deviation plus the mean of each measurement to produce the final random data
        set. The reason for this procedure is to apply covariance if applicable.

        The covariance is applied to the set of samples using the Chelosky algorithm. The covariance
        matrix is constructed, and the Chelosky decomposition of the covariance matrix is calculated.
        The Chelosky decomposition can then be used to correlate the vector of random samples with 0
        mean and unit variance.

        Args:
            measurement_ids (set): a set of the source measurements to simulate (their IDs)

        Returns:
            A N row times M columns matrix where N is the number of measurements to simulate
            and M is the requested sample size for Monte Carlo simulations. Each row of this
            matrix is an array of random values with mean of 0 and unit variance

        """
        raw_offset_matrix = np.vstack([np.random.normal(0, 1, sample_size) for _ in measurement_ids])
        offset_matrix = _correlate_random_sample_set_with_chelosky_decomposition(measurement_ids, raw_offset_matrix)
        return offset_matrix

    def __generate_random_data_set(measurement, offsets):
        """Generate random simulated measurements for each MeasuredValue

        This method simply applies the desired mean and standard deviation to the random sample set with
        0 mean and unit variance

        """
        std = measurement.std if isinstance(measurement, data.RepeatedlyMeasuredValue) else measurement.error
        center_value = measurement.value
        return offsets * std + center_value

    source_measurements = _find_source_measurements(value)  # type: Set[UUID]

    # a dictionary object where the key is the unique ID of a MeasuredValue, and the values are the lists
    # of randomly generated numbers with a normal distribution around the value of this MeasuredValue, with
    # the standard deviation being the error of this value
    data_sets = {}

    sample_set = __generate_offset_matrix(source_measurements)
    for operand_id, sample in zip(source_measurements, sample_set):
        data_sets[operand_id] = __generate_random_data_set(data.ExperimentalValue._register[operand_id], sample)

    result_data_set = _evaluate_formula_tree_for_value(value, data_sets)

    # check the quality of the result data
    if isinstance(result_data_set, np.ndarray):
        result_data_set = result_data_set[~np.isnan(result_data_set)]  # remove undefined values
    if len(result_data_set) / settings.get_monte_carlo_sample_size() < 0.9:
        # if over 10% of the results calculated is invalid
        warnings.warn("More than 10 percent of the random data generated for the Monte Carlo simulation falls "
                      "outside of the domain on which the function is defined. Check the uncertainty or the "
                      "standard deviation of the measurements passed in, it is possible that the domain of this "
                      "function is too narrow compared to the standard deviation of the measurements. Consider "
                      "choosing a different error method for this value.")

    # use the standard deviation as the uncertainty and mean as the center value
    return data.ValueWithError(np.mean(result_data_set), np.std(result_data_set, ddof=1))


def propagate_units(operator, operands):
    units = {}
    if operator == lit.MUL:
        for operand in operands:
            for unit, exponent in operand.get_units().items():
                units[unit] = (0 if unit not in units else units[unit]) + exponent
        return units
    elif operator == lit.DIV:
        for unit, exponent in operands[0].get_units().items():
            units[unit] = (0 if unit not in units else units[unit]) + exponent
        for unit, exponent in operands[1].get_units().items():
            units[unit] = (0 if unit not in units else units[unit]) - exponent
        return units
    elif operator in [lit.ADD, lit.SUB]:
        if len(operands[0].get_units()) == 0 and len(operands[1].get_units()) > 0:
            return operands[1].get_units()
        elif len(operands[1].get_units()) == 0 and len(operands[0].get_units()) > 0:
            return operands[0].get_units()
        elif operands[1].get_units() != operands[0].get_units():
            warnings.warn("You're trying to add/subtract two values with mismatching units, returning empty unit")
            return units
        else:
            return operands[0].get_units()
    elif operator == lit.NEG:
        return operands[0].get_units()
    # TODO: implement unit propagation for non-linear operations
    return units


def sqrt(x):
    return execute_without_wrapping(lit.SQRT, x)


def exp(x):
    return execute_without_wrapping(lit.EXP, x)


def sin(x):
    return execute_without_wrapping(lit.SIN, x)


def sind(x):
    return sin(x / 180 * np.pi)


def cos(x):
    return execute_without_wrapping(lit.COS, x)


def cosd(x):
    return cos(x / 180 * np.pi)


def tan(x):
    return execute_without_wrapping(lit.TAN, x)


def tand(x):
    return tan(x / 180 * np.pi)


def sec(x):
    return execute_without_wrapping(lit.SEC, x)


def secd(x):
    return sec(x / 180 * np.pi)


def csc(x):
    return execute_without_wrapping(lit.CSC, x)


def cscd(x):
    return csc(x / 180 * np.pi)


def cot(x):
    return execute_without_wrapping(lit.COT, x)


def cotd(x):
    return cotd(x / 180 * np.pi)


def asin(x):
    return execute_without_wrapping(lit.ASIN, x)


def acos(x):
    return execute_without_wrapping(lit.ACOS, x)


def atan(x):
    return execute_without_wrapping(lit.ATAN, x)


def _evaluate_formula_tree_for_value(root, samples) -> Union[np.ndarray, float]:
    """Evaluate the value of a formula tree

    This method has an option of passing in a "samples" parameter where the value for each
    MeasuredValue can be specified. The keys of this dictionary object are the unique IDs of
    the values. This feature is ued for Monte Carlo simulations where a formula needs to be
    evaluated for a large set of values

    The "sample" of a measurement can either be a single number or an np.ndarray. In the latter
    case, the return value will also be an np.ndarray. This improves the computing speed
    significantly because it takes advantage of numpy's vectorization of the arrays

    Args:
        root (Union[data.ExperimentalValue, float])
        samples (Dict[UUID, Union[np.ndarray, numbers.Real]])

    """
    if isinstance(root, data.MeasuredValue) and root._id in samples:
        # use the value in the sample instead of its original value if specified
        return samples[root._id]
    elif isinstance(root, data.MeasuredValue) or isinstance(root, data.Constant):
        return root.value
    elif isinstance(root, numbers.Real):
        return float(root)
    elif isinstance(root, data.DerivedValue):
        operator = root._formula[lit.OPERATOR]
        operands = (_evaluate_formula_tree_for_value(value, samples) for value in root._formula[lit.OPERANDS])
        result = operations[operator](*operands)
        return result


def _correlate_random_sample_set_with_chelosky_decomposition(variables, sample_vector):
    """Uses the Chelosky Decomposition algorithm to correlate samples for the Monte Carlo method"""
    variables = list(map(lambda key: data.ExperimentalValue._register[key], variables))
    correlation_matrix = np.array([[data.get_correlation(row, col) for col in variables] for row in variables])
    try:
        chelosky_decomposition = np.linalg.cholesky(correlation_matrix)
        result_vector = np.dot(chelosky_decomposition, sample_vector)
        return result_vector
    except np.linalg.linalg.LinAlgError:
        warnings.warn("Fail to generate a physical correlation matrix for the values provided, using uncorrelated "
                      "samples for this simulation. Please check that the correlation or covariance factors you "
                      "have given to the measurements are physical. ")
        return sample_vector


def _find_source_measurements(value) -> Set[UUID]:
    """Returns a set of IDs for MeasuredValue objects that the given value was derived from

    Args:
        value (data.ExperimentalValue)

    """
    if isinstance(value, data.MeasuredValue):
        return {value._id}
    elif isinstance(value, data.DerivedValue):
        source_measurements = set()
        for operand in value._formula[lit.OPERANDS]:
            if isinstance(operand, data.MeasuredValue):
                source_measurements.add(operand._id)
            elif isinstance(operand, data.DerivedValue):
                source_measurements.update(_find_source_measurements(operand))
        return source_measurements
    else:
        return set()


operations = {
    lit.NEG: lambda x: -x,
    lit.ADD: lambda a, b: a + b,
    lit.SUB: lambda a, b: a - b,
    lit.MUL: lambda a, b: a * b,
    lit.DIV: lambda a, b: a / b,
    lit.SQRT: lambda x: np.sqrt(x),
    lit.EXP: lambda x: np.exp(x),
    lit.SIN: lambda x: np.sin(x),
    lit.COS: lambda x: np.cos(x),
    lit.TAN: lambda x: np.tan(x),
    lit.ASIN: lambda x: np.arcsin(x),
    lit.ACOS: lambda x: np.arccos(x),
    lit.ATAN: lambda x: np.arctan(x),
    lit.SEC: lambda x: 1 / np.cos(x),
    lit.CSC: lambda x: 1 / np.sin(x),
    lit.COT: lambda x: 1 / np.tan(x)
}

# the usage of the differentiators are documented under the method differentiator
differentiators = {
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
    lit.COT: lambda other, x: -1 / (np.sin(x.value) ** 2) * x.derivative(other)
}
