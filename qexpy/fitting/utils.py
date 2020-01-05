"""Utility functions for the fit module"""
import functools
import inspect
import warnings

from collections import namedtuple

# Contains the name, callable fit function, and the constraints on the fit parameters
from enum import Enum
from numbers import Real
from inspect import Parameter
from typing import Callable, List

from qexpy.data import operations as op
from qexpy.settings import literals as lit
from qexpy.utils import IllegalArgumentError

# Contains the name, callable function, and parameter constraints on a fit model
FitModelInfo = namedtuple("FitModelInfo", "name, func, param_constraints")

# Contains constraints on fit parameters, including the number of params required, a flag
# indicating if there is a variable position argument in the fit function, which indicates
# that the actual number of fit parameters might be higher than what it looks, and a flag
# stating if guesses are required to execute this fit
FitParamConstraints = namedtuple("FitParamConstraints", "length, var_len, guess_required")

# Contains the parameter information used in a fit
FitParamInfo = namedtuple("FitParamInfo", "parguess, parnames, parunits")


class FitModel(Enum):
    """QExPy supported pre-set fit models"""
    LINEAR = lit.LIN
    QUADRATIC = lit.QUAD
    POLYNOMIAL = lit.POLY
    GAUSSIAN = lit.GAUSS
    EXPONENTIAL = lit.EXPO


def prepare_fit_model(model) -> FitModelInfo:
    """Prepares the fit model and fit function for a fit

    Args:
        model: the fit model as is passed into the fit function

    Returns:
        model (FitModelInfo): the fit model and all information related to it

    """

    # First find the name and the callable fit function for the model
    if isinstance(model, str) and model in FITTERS:
        name, func = model, FITTERS[model]
    elif isinstance(model, FitModel):
        name, func = model.value, FITTERS[model.value]
    elif callable(model):
        name, func = "custom", model
    else:
        raise ValueError(
            "Invalid fit model specified! The fit model can be one of the following: "
            "one of the pre-set fit models in the form of a string or chosen from the "
            "q.FitModel enum, or a custom callable fit function")

    # Now find the number of parameters this fit function has
    params = list(inspect.signature(func).parameters.values())

    if any(arg.kind in [Parameter.KEYWORD_ONLY, Parameter.VAR_KEYWORD] for arg in params):
        raise ValueError("The fit function should not have keyword arguments")

    # If the last param is variable positional, the actual number of params may be higher
    var_pos_present = params[-1].kind == Parameter.VAR_POSITIONAL

    # The first argument of the fit function is the variable, only the rest are parameters
    nr_of_params = len(params) - 1

    if nr_of_params == 0:
        raise ValueError("The number of parameters in the given fit model is 0!")

    guess_required = name not in [lit.LIN, lit.QUAD, lit.POLY]
    constraints = FitParamConstraints(nr_of_params, var_pos_present, guess_required)

    return FitModelInfo(name, func, constraints)


def prepare_param_info(model: FitModelInfo, **kwargs) -> (FitParamInfo, FitModelInfo):
    """Prepares the parameter information for a fit function

    Args:
        model (FitModelInfo): the fit model used for this fit

    Keyword Args:
        parguess: the vector of parameter guesses
        parnames: the vector of parameter names
        parunits: the vector of parameter units

    Returns:
        The first return is a FitParamInfo which includes: parguess, parnames, parunits,
        all read from kwargs and validated against the fit model and constraints. The last
        return value would be the updated FitModelInfo based on all the parameter info.

    """

    constraints = model.param_constraints

    # check if guess parameters are provided
    parguess = kwargs.get("parguess", None)
    if constraints.guess_required and parguess is None:
        warnings.warn(
            "You have not provided any guesses of parameters for a {} fit. For this type "
            "of fitting, it is recommended to specify parguess".format(model.name))

    validate_param_info(parguess, "parguess", constraints)

    if parguess is not None:
        # The length of the parguess vector dictates the number of parameters
        constraints = FitParamConstraints(len(parguess), False, True)
        model = FitModelInfo(model.name, model.func, constraints)

    if parguess and any(not isinstance(guess, Real) for guess in parguess):
        raise TypeError("The guess parameters provided are not real numbers!")

    parnames = kwargs.get("parnames", prepare_param_names(model))
    validate_param_info(parnames, "parnames", constraints)
    if parnames and any(not isinstance(name, str) for name in parnames):
        raise TypeError("The parameter names provided are not strings!")

    parunits = kwargs.get("parunits", [""] * constraints.length)
    validate_param_info(parunits, "parunits", constraints)
    if parunits and any(not isinstance(unit, str) for unit in parunits):
        raise TypeError("The parameter units provided are not strings!")

    return FitParamInfo(parguess, parnames, parunits), model


def validate_param_info(info, info_name: str, constraints: FitParamConstraints):
    """Validates the param information is valid and matches the fit model"""

    if not info:
        return  # skip if there's nothing to check

    if not isinstance(info, (list, tuple)):
        raise IllegalArgumentError("\"{}\" has to be a list or a tuple.".format(info_name))

    if constraints.var_len and len(info) < constraints.length:
        raise ValueError(
            "The length of \"{}\" ({}) doesn't match the number of parameters in the fit "
            "function ({} or higher)".format(info_name, len(info), constraints.length))

    if not constraints.var_len and len(info) != constraints.length:
        raise ValueError(
            "The length of \"{}\" ({}) doesn't match the number of parameters in the fit "
            "function (expecting {})".format(info_name, len(info), constraints.length))


def prepare_param_names(model: FitModelInfo):
    """Finds the default param names for pre-set fit models"""

    if model.name in DEFAULT_PARNAMES:
        return DEFAULT_PARNAMES.get(model.name)

    nr_of_params = model.param_constraints.length

    # check the function signature for custom functions
    par_names = get_param_names_from_signature(model.func, nr_of_params)

    return par_names if par_names else [""] * nr_of_params


def get_param_names_from_signature(func: Callable, nr_of_params: int) -> List:
    """Inspect the signature of the custom function for parameter names"""

    # get all arguments to the function except for the first one (the variable)
    params = list(inspect.signature(func).parameters.values())[1:]

    # the last parameter could be variable, so we process the rest of the parameters first
    param_names = list(param.name for param in params[:-1])

    # now process the last parameter
    if params[-1].kind == Parameter.VAR_POSITIONAL:
        left_overs = nr_of_params - len(param_names)  # how many params left to be filled
        last_params = list("{}_{}".format(params[-1].name, idx) for idx in range(left_overs))
    else:
        last_params = [params[-1].name]

    param_names.extend(last_params)

    return param_names


FITTERS = {
    lit.LIN: lambda x, a, b: a * x + b,
    lit.QUAD: lambda x, a, b, c: a * x ** 2 + b * x + c,
    lit.POLY: lambda x, *coeffs: functools.reduce(lambda a, b: a * x + b, reversed(coeffs)),
    lit.EXPO: lambda x, c, a: c * op.exp(-a * x),
    lit.GAUSS: lambda x, norm, mean, std: norm / op.sqrt(
        2 * op.pi * std ** 2) * op.exp(-1 / 2 * (x - mean) ** 2 / std ** 2)
}

DEFAULT_PARNAMES = {
    lit.LIN: ["slope", "intercept"],
    lit.EXPO: ["amplitude", "decay constant"],
    lit.GAUSS: ["normalization", "mean", "std"]
}
