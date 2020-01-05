"""This module contains curve fitting functions"""

import inspect
import numpy as np
import scipy.optimize as opt

from typing import Callable
from inspect import Parameter
from collections import namedtuple
from qexpy.utils.exceptions import IllegalArgumentError
from .utils import FitModelInfo, FitParamConstraints

import qexpy.data.data as dt
import qexpy.data.datasets as dts
import qexpy.settings.literals as lit
import qexpy.utils as utils

from . import utils as fut

# container for the raw outputs of a fit
RawFitResults = namedtuple("RawFitResults", "popt, perr, pcov")

# container for fit results
FitResults = namedtuple("FitResults", "func, params, residuals, chi2, pcorr")

ARRAY_TYPES = np.ndarray, list


class XYFitResult:
    """Stores the results of a curve fit"""

    def __init__(self, **kwargs):
        """Constructor for an XYFitResult object"""

        self._dataset = kwargs.pop("dataset")
        self._model = kwargs.pop("model")
        self._xrange = kwargs.pop("xrange")

        result_func = kwargs.pop("res_func")
        result_params = kwargs.pop("res_params")
        pcorr = kwargs.pop("pcorr")

        y_fit_res = result_func(self._dataset.xdata)
        self._ndof = len(y_fit_res) - len(result_params) - 1

        y_err = self._dataset.ydata - y_fit_res

        chi2 = sum(
            (res.value / err) ** 2 for res, err in zip(y_err, self._dataset.yerr) if err != 0)

        self._result = FitResults(result_func, result_params, y_err, chi2, pcorr)

    def __getitem__(self, index):
        return self._result.params[index]

    def __str__(self):
        header = "----------------- Fit Results -------------------"
        fit_type = "Fit of {} to {}\n".format(self._dataset.name, self._model.name)
        res_params = map(str, self._result.params)
        res_param_str = "Result Parameter List: \n{}\n".format(",\n".join(res_params))
        corr_matrix = np.array_str(self._result.pcorr, precision=3)
        corr_matrix_str = "Correlation Matrix: \n{}\n".format(corr_matrix)
        chi2_ndof = "chi2/ndof = {:.2f}/{}\n".format(self._result.chi2, self._ndof)
        ending = "--------------- End Fit Results -----------------"
        return "\n".join(
            [header, fit_type, res_param_str, corr_matrix_str, chi2_ndof, ending])

    @property
    def dataset(self):
        """dts.XYDataSet: The dataset used for this fit"""
        return self._dataset

    @property
    def fit_function(self):
        """Callable: The function that fits to this data set"""
        return self._result.func

    @property
    def params(self):
        """List[dt.ExperimentalValue]: The fit parameters of the fit function"""
        return self._result.params

    @property
    def residuals(self):
        """dts.ExperimentalValueArray: The residuals of the fit"""
        return self._result.residuals

    @property
    def chi_squared(self):
        """dt.ExperimentalValue: The goodness of fit represented as chi^2"""
        return self._result.chi2

    @property
    def ndof(self):
        """int: The degree of freedom of this fit function"""
        return self._ndof

    @property
    def xrange(self):
        """tuple: The xrange of the fit"""
        return self._xrange


def fit(*args, **kwargs) -> XYFitResult:
    """Perform a fit to a data set

    The fit function can be called on an XYDataSet object, or two arrays or MeasurementArray
    objects. QExPy provides 5 builtin fit models, which includes linear fit, quadratic fit,
    general polynomial fit, gaussian fit, and exponential fit. The user can also pass in a
    custom function they wish to fit their dataset on. For non-polynomial fit functions, the
    user would usually need to pass in an array of guesses for the parameters.

    Args:
        *args: An XYDataSet object or two arrays to be fitted.

    Keyword Args:
        model: the fit model given as the string or enum representation of a pre-set model
            or a custom callable function with parameters. Available pre-set models include:
            "linear", "quadratic", "polynomial", "exponential", "gaussian"
        xrange (tuple|list): a pair of numbers indicating the domain of the function
        degrees (int): the degree of the polynomial if polynomial fit were chosen
        parguess (list): initial guess for the parameters
        parnames (list): the names of each parameter
        parunits (list): the units for each parameter
        dataset: the XYDataSet instance to fit on
        xdata : the x-data of the fit
        ydata: the y-data of the fit
        xerr: the uncertainty on the xdata
        yerr: the uncertainty on the ydata

    Returns:
        XYFitResult: the result of the fit

    See Also:
        :py:class:`~qexpy.data.XYDataSet`

    """

    result = __try_fit_to_xy_dataset(*args, **kwargs)
    if result:
        return result

    result = __try_fit_to_xdata_and_ydata(*args, **kwargs)
    if result:
        return result

    raise IllegalArgumentError(
        "Unable to execute fit. Please make sure the arguments provided are correct.")


def fit_to_xy_dataset(dataset: dts.XYDataSet, model, **kwargs) -> XYFitResult:
    """Perform a fit on an XYDataSet object"""

    fit_model = fut.prepare_fit_model(model)

    if fit_model.name == lit.POLY:
        # By default, the degree of a polynomial fit model is 3, because if it were 2, the
        # quadratic fit model would've been chosen. The number of parameters is the degree
        # of the fit model plus one. (e.g. a degree-1, or linear fit, has 2 params)
        new_constraints = FitParamConstraints(kwargs.get("degrees", 3) + 1, False, False)
        fit_model = FitModelInfo(fit_model.name, fit_model.func, new_constraints)

    param_info, fit_model = fut.prepare_param_info(fit_model, **kwargs)

    xrange = kwargs.get("xrange", None)
    if xrange and utils.validate_xrange(xrange):
        x_to_fit = dataset.xdata[(xrange[0] <= dataset.xdata) & (dataset.xdata < xrange[1])]
        y_to_fit = dataset.ydata[(xrange[0] <= dataset.xdata) & (dataset.xdata < xrange[1])]
    else:
        x_to_fit = dataset.xdata
        y_to_fit = dataset.ydata

    yerr = y_to_fit.errors if any(err > 0 for err in y_to_fit.errors) else None

    if fit_model.name in [lit.POLY, lit.LIN, lit.QUAD]:
        raw_res = __polynomial_fit(
            x_to_fit, y_to_fit, fit_model.param_constraints.length - 1, yerr)
    else:
        raw_res = __curve_fit(
            fit_model.func, x_to_fit, y_to_fit, param_info.parguess, yerr)

    # wrap the parameters in MeasuredValue objects
    def wrap_param_in_measurements():
        par_res = zip(raw_res.popt, raw_res.perr, param_info.parunits, param_info.parnames)
        for param, err, unit, name in par_res:
            yield dt.MeasuredValue(param, err, unit=unit, name=name)

    params = list(wrap_param_in_measurements())

    pcorr = utils.cov2corr(raw_res.pcov)
    __correlate_fit_params(params, raw_res.pcov)

    # wrap the result function with the params
    result_func = __combine_fit_func_and_fit_params(fit_model.func, params)

    return XYFitResult(dataset=dataset, model=fit_model, res_func=result_func,
                       res_params=params, pcorr=pcorr, xrange=xrange)


def __try_fit_to_xy_dataset(*args, **kwargs):
    """Helper function to parse the inputs to a call to fit() for a single XYDataSet"""

    dataset = kwargs.pop("dataset", args[0] if args else None)
    model = kwargs.pop("model", args[1] if len(args) > 1 else None)

    if isinstance(dataset, dts.XYDataSet) and model:
        return fit_to_xy_dataset(dataset, model, **kwargs)

    return None


def __try_fit_to_xdata_and_ydata(*args, **kwargs):
    """Helper function to parse the inputs to a call to fit() for separate xdata and ydata"""

    xdata = kwargs.pop("xdata", args[0] if args else None)
    ydata = kwargs.pop("ydata", args[1] if len(args) > 1 else None)
    model = kwargs.pop("model", args[2] if len(args) > 2 else None)

    if not isinstance(xdata, dts.ExperimentalValueArray):
        xdata = np.asarray(xdata) if isinstance(xdata, ARRAY_TYPES) else np.empty(0)

    if not isinstance(ydata, dts.ExperimentalValueArray):
        ydata = np.asarray(ydata) if isinstance(ydata, ARRAY_TYPES) else np.empty(0)

    if xdata.size and ydata.size and model:
        return fit_to_xy_dataset(dts.XYDataSet(xdata, ydata, **kwargs), model, **kwargs)

    return None


def __polynomial_fit(xdata, ydata, degrees, yerr) -> RawFitResults:
    """perform a polynomial fit with numpy.polyfit"""

    weights = 1 / yerr if yerr is not None else None
    popt, pcov = np.polyfit(xdata.values, ydata.values, degrees, cov=True, w=weights)
    perr = np.sqrt(np.diag(pcov))
    return RawFitResults(popt, perr, pcov)


def __curve_fit(fit_func, xdata, ydata, parguess, yerr) -> RawFitResults:
    """perform a regular curve fit with scipy.optimize.curve_fit"""

    try:
        popt, pcov = opt.curve_fit(
            fit_func, xdata.values, ydata.values, p0=parguess, sigma=yerr)

        # adjust the fit by factoring in the uncertainty on x
        if any(err > 0 for err in xdata.errors):
            func = __combine_fit_func_and_fit_params(fit_func, popt)
            yerr = 0 if yerr is None else yerr
            adjusted_yerr = np.sqrt(
                yerr ** 2 + xdata.errors * utils.numerical_derivative(func, xdata.errors))

            # re-calculate the fit with adjusted uncertainties for ydata
            popt, pcov = opt.curve_fit(
                fit_func, xdata.values, ydata.values, p0=parguess, sigma=adjusted_yerr)

    except RuntimeError:  # pragma: no cover

        # Re-write the error message so that it can be more easily understood by the user
        raise RuntimeError(
            "Fit could not converge. Please check that the fit model is well defined, and "
            "that the parameter guess as well as the y-errors are appropriate.")

    # The error on the parameters
    perr = np.sqrt(np.diag(pcov))

    return RawFitResults(popt, perr, pcov)


def __combine_fit_func_and_fit_params(func: Callable, params) -> Callable:
    """wraps a function with params to a function of x"""

    result_func = utils.vectorize(lambda x: func(x, *params))

    # Change signature of the function to match the actual signature
    sig = inspect.signature(result_func)
    new_sig = sig.replace(parameters=[Parameter("x", Parameter.POSITIONAL_ONLY)])
    result_func.__signature__ = new_sig

    return result_func


def __correlate_fit_params(params, corr):
    """Apply correlation to the list of parameters with the covariance matrix"""

    for index1, param1 in enumerate(params):
        for index2, param2 in enumerate(params[index1 + 1:]):
            if param1.error == 0 or param2.error == 0:  # pragma: no cover
                continue
            param1.set_covariance(param2, corr[index1][index2 + index1 + 1])
