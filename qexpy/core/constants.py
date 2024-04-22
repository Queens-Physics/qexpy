"""Defines constants"""

from numbers import Real

from .experimental_value import ExperimentalValue
from .formula import _Formula


class Constant(ExperimentalValue, _Formula):
    """A numerical value."""

    def __new__(cls, value: Real, error: Real = 0, name: str = "", unit: str = ""):
        return object.__new__(Constant)

    def __init__(self, value: Real, error: Real = 0, name: str = "", unit: str = ""):
        self._value = float(value)
        self._error = float(error)
        super().__init__(name, unit)

    @property
    def value(self) -> float:
        return self._value

    @property
    def error(self) -> float:
        return self._error

    def _derivative(self, x: _Formula) -> float:
        return 0


# Physical constants
e = Constant(1.602176634e-19, unit="C")  # elementary charge (electron charge)
G = Constant(6.6743e-11, 0.00015e-11, unit="m^3*kg^-1*s^-2")  # gravitational constant
me = Constant(9.1093837015e-31, 0.0000000028e-31, unit="kg")  # mass of electron
c = Constant(299792458, unit="m*s^-1")  # speed of light
eps0 = Constant(8.8541878128e-12, 0.0000000013e-12, unit="F*m^-1")  # electric permittivity
mu0 = Constant(1.25663706212e-6, 0.00000000019e-6, unit="N*A^-2")  # magnetic permeability
h = Constant(6.62607015e-34, unit="J*Hz^-1")  # Planck's constant
hbar = Constant(1.05457181e-34, unit="J*s")  # reduced Planck's constant
kb = Constant(1.380649e-23, unit="J*K^-1")  # Boltzmann's constant

# Mathematical constants
pi = 3.141592653589793
