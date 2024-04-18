"""Defines the DerivedValue class"""

from functools import cached_property

import qexpy as q
from qexpy.core.experimental_value import ExperimentalValue
from qexpy.core.formula import _Formula
from qexpy.utils import Unit


class DerivedValue(ExperimentalValue):
    """A calculated value with a propagated uncertainty"""

    def __init__(self, formula: _Formula):
        self._formula = formula
        self._error_method = "auto"
        super().__init__("", None)

    def __copy__(self):
        obj = object.__new__(DerivedValue)
        obj._formula = self._formula
        obj._error_method = self._error_method
        obj._name = self._name
        return obj

    @property
    def value(self) -> float:
        return self._value

    @property
    def error(self) -> float:
        return self._error

    @property
    def unit(self) -> Unit:
        return self._unit

    @cached_property
    def _value(self):
        return self._formula.value

    @cached_property
    def _error(self):
        return self._formula.error

    @cached_property
    def _unit(self):  # pylint: disable=method-hidden
        return self._formula.unit

    @property
    def error_method(self) -> str:
        """The method of error propagation used for this value

        QExPy supports error propagation with partial derivatives (`"derivative"`) and by using
        a Monte Carlo simulation (`"monte-carlo"`). By default, the global preference for the
        error method will be used, but it is also possible to configure the error method for a
        single derived value. To simply use the global option, set this to `"auto"`.

        Examples
        --------

        >>> import qexpy as q
        >>> a = q.Measurement(5, 0.1)
        >>> b = q.Measurement(6, 0.1)
        >>> res = a * b
        >>> res.error_method = "monte-carlo"
        >>> print(res)
        30.0 +/- 0.8

        """
        if self._error_method == "auto":
            return q.options.error.method
        return self._error_method

    @error_method.setter
    def error_method(self, method: str):
        if method not in ("derivative", "monte-carlo", "auto"):
            raise ValueError("The error method can only be 'derivative', 'monte-carlo', or 'auto'")
        self._error_method = method
