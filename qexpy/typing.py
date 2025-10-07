"""Internal module used for typing."""

from fractions import Fraction

import numpy as np

Number = int | float | Fraction
ArrayLike = list | tuple | np.ndarray
