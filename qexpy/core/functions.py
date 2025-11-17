"""Module for various functions."""

from functools import singledispatch

import numpy as np

from qexpy.typing import ArrayLike


@singledispatch
def correlation(var1, var2) -> float:
    r"""Find the correlation coefficient between two values or arrays.

    The correlation coefficient is the normalized covariance, defined as

    .. math::
        \rho_{xy} = \frac{cov_{xy}}{\sigma_x\sigma_y}

    where :math:`\sigma_x` and :math:`sigma_y` are the standard deviations.
    It measures the joint variability of two variables.

    See Also
    --------
    func:`~.qexpy.core.functions.covariance`

    """
    raise TypeError(
        "The correlation is undefined between variables of type "
        f"{type(var1)} and {type(var2)}."
    )


@correlation.register
def _(var1: ArrayLike, var2: ArrayLike) -> float:
    if not isinstance(var2, ArrayLike):
        raise TypeError(
            "The correlation is undefined between variables of type "
            f"{type(var1)} and {type(var2)}."
        )
    return float(np.corrcoef(var1, var2)[0][1])


@singledispatch
def covariance(var1, var2) -> float:
    r"""Find the covariance between two values or arrays.

    The covariance is defined as

    .. math::
        cov_{xy} = \frac{\sum_{i}(x_i-\bar{x})(y_i-\bar{y})}{N-1}

    It measures the joint variability of two variables.

    See Also
    --------
    func:`~.qexpy.core.functions.correlation`

    """
    raise TypeError(
        "The covariance is undefined between variables of type "
        f"{type(var1)} and {type(var2)}."
    )


@covariance.register
def _(var1: ArrayLike, var2: ArrayLike):
    if not isinstance(var2, ArrayLike):
        raise TypeError(
            "The covariance is undefined between variables of type "
            f"{type(var1)} and {type(var2)}."
        )
    return float(np.cov(var1, var2)[0][1])
