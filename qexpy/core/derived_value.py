"""Defines the DerivedValue class"""

from functools import cached_property

import qexpy as q
from qexpy.core.experimental_value import ExperimentalValue
from qexpy.core.formula import _Formula
from qexpy.utils import Unit


class DerivedValue(ExperimentalValue):
    """A calculated value with a propagated uncertainty

    When :py:class:`~qexpy.core.ExperimentalValue` objects are used in calculations, the results
    are wrapped in instances of this class. Internally, the ``DerivedValue`` stores the expression
    tree for how it was calculated, the leaf nodes of which are the constants and measurements.
    This allows the value and uncertainty to be calculated more flexibly. Different error methods
    can be chosen to propagate the uncertainty.

    Attributes
    ----------

    value
    error
    relative_error
    name
    unit

    """

    def __init__(self, formula: _Formula):
        self._formula = formula
        self._mc = MonteCarloConfig(self._formula)
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
        if self.error_method == "monte-carlo":
            return self._mc.value
        return self._value

    @property
    def error(self) -> float:
        if self.error_method == "monte-carlo":
            return self._mc.error
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


class MonteCarloConfig:
    """Stores all data and configurations of a Monte Carlo simulation."""

    def __init__(self, formula: _Formula):
        self._formula = formula
        self._sample_size = 0
        self._samples = None

    @property
    def value(self):
        """The value computed from the simulated samples"""
        return self.samples.mean()

    @property
    def error(self):
        """The error computed from the simulated samples"""
        return self.samples.std()

    @property
    def sample_size(self) -> int:
        """The number of samples to use in a Monte Carlo simulation"""
        if not self._sample_size:
            return q.options.error.mc.sample_size
        return self._sample_size

    @sample_size.setter
    def sample_size(self, size: int):
        if size < 0:
            raise ValueError("The sample size must be a positive integer!")
        self._sample_size = size
        self._samples = None

    @property
    def samples(self):
        """The array of simulated samples"""
        if self._samples is None:
            self._samples = q.core.monte_carlo(self._formula, self.sample_size)
        return self._samples
