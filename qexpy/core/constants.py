"""Defines common physical constants."""

from qexpy.typing import Number
from qexpy.units import UnitLike

from .quantity import Quantity


class Constant(Quantity):
    """Represents a constant numerical value."""

    def __init__(
        self,
        value: Number,
        error: Number = 0.0,
        name: str = "",
        unit: UnitLike = "",
    ):
        super().__init__(name, unit)
        self._value = float(value)
        self._error = float(error)

    @property
    def value(self) -> float:
        """The value of this constant.

        :type: float

        """
        return self._value

    @property
    def error(self) -> float:
        """The uncertainty of this constant.

        :type: float

        Although constants are allowed to have uncertainties, they do not
        participate in error propagation.

        """
        return self._error


######################
# Physical Constants #
######################

# Elementary charge (electron charge)
e = Constant(1.602176634e-19, name="e", unit="C")
# Gravitational constant
G = Constant(6.6743e-11, 0.00015e-11, name="G", unit="m^3*kg^-1*s^-2")
# Mass of the electron
me = Constant(9.1093837015e-31, 0.0000000028e-31, name="me", unit="kg")
# Speed of light
c = Constant(299792458, name="c", unit="m*s^-1")
# Electric permittivity of free space
eps0 = Constant(8.8541878128e-12, 0.0000000013e-12, name="eps0", unit="F*m^-1")
# Magnetic permeability of free space
mu0 = Constant(1.25663706212e-6, 0.00000000019e-6, name="mu0", unit="N*A^-2")
# Planck's constant
h = Constant(6.62607015e-34, name="h", unit="J*Hz^-1")
# Reduced Plank's constant
hbar = Constant(1.05457181e-34, name="hbar", unit="J*s")
# Boltzmann's constant
kb = Constant(1.380649e-23, name="kb", unit="J*K^-1")

# Mathematical constants
pi = 3.141592653589793
