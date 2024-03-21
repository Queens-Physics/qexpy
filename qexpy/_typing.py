"""Type definitions for easy type checking"""

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, list]
ArrayLikeT = (np.ndarray, list)

__all__ = ["ArrayLikeT", "ArrayLike"]
