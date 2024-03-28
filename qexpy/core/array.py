"""Defines a wrapper of numpy.ndarray for experimental values."""

from __future__ import annotations

from numbers import Real

import numpy as np

from qexpy._typing import ArrayLikeT, ArrayLike


def pack_data_arrays(
    data: ArrayLike, error: ArrayLike | Real, relative_error: ArrayLike | Real | None
):
    """Packs data and error into arrays of data and errors"""

    data = np.asarray(data)

    # First deal with relative error if present
    if isinstance(relative_error, Real):
        error = data * float(relative_error)
    elif isinstance(relative_error, ArrayLikeT):
        if len(data) != len(relative_error):
            raise TypeError("The data and error arrays must have the same length!")
        error = data * np.asarray(relative_error)

    # Then deal with the error
    elif isinstance(error, ArrayLikeT):
        error = np.asarray(error)
    elif isinstance(error, Real):
        error = np.full(len(data), float(error))

    # Raise error if packing fails
    else:
        raise TypeError(
            "The error must be a non-negative real number or an array of non-"
            "negative real numbers that match the length of the data array!"
        )

    # Check that the data and error arrays have the same length
    if len(data) != len(error):
        raise TypeError("The data and error arrays must have the same length!")

    # Return the packed array
    return data, error
