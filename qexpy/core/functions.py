"""General functions."""

from __future__ import annotations

import qexpy as q
from qexpy._typing import ArrayLike


def correlation(
    var1: q.core.Measurement | ArrayLike, var2: q.core.Measurement | ArrayLike
) -> float:
    """Gets the correlation between two measurements or two arrays

    The correlation coefficient is the normalized covariance, defined as

    .. math::
        \\rho_{xy} = \\frac{cov_{xy}}{\\sigma_x\\sigma_y}

    where :math:`\\sigma_x` and :math:`\\sigma_y` are the standard deviations.

    """

    if isinstance(var1, q.core.Measurement) and isinstance(var2, q.core.Measurement):
        return var1.get_correlation(var2)


def covariance(var1: q.core.Measurement | ArrayLike, var2: q.core.Measurement | ArrayLike) -> float:
    """Gets the covariance between two measurements or two arrays

    The covariance is defined as

    .. math::
        cov_{xy} = \\frac{\\sum_{i}(x_i-\\bar{x})(y_i-\\bar{y})}{N-1}

    """

    if isinstance(var1, q.core.Measurement) and isinstance(var2, q.core.Measurement):
        return var1.get_covariance(var2)
