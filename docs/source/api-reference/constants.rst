.. _api.constants:

Constants
=========

QExPy makes available the following list of physical constants:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Name
     - Description
   * - ``qexpy.e``
     - `Elementary charge <https://en.wikipedia.org/wiki/Elementary_charge>`_ (electron charge)
   * - ``qexpy.G``
     - `Gravitational constant <https://en.wikipedia.org/wiki/Gravitational_constant>`_
   * - ``qexpy.me``
     - `Electron mass <https://en.wikipedia.org/wiki/Electron>`_
   * - ``qexpy.c``
     - `Speed of light in vacuum <https://en.wikipedia.org/wiki/Speed_of_light>`_
   * - ``qexpy.eps0``
     - `Vacuum electric permittivity <https://en.wikipedia.org/wiki/Vacuum_permittivity>`_ (permittivity of free space) in :math:`\text{F}\cdot\text{m}^{-1}`
   * - ``qexpy.mu0``
     - `Vacuum magnetic permeability <https://en.wikipedia.org/wiki/Vacuum_permeability>`_ (permeability of free space) in :math:`\text{N}\cdot\text{A}^{-2}`
   * - ``qexpy.h``
     - `Planck constant <https://en.wikipedia.org/wiki/Planck_constant>`_ in :math:`\text{J}\cdot \text{Hz}^{-1}`
   * - ``qexpy.hbar``
     - `Reduced Planck constant <https://en.wikipedia.org/wiki/Planck_constant>`_ in :math:`\text{J}\cdot \text{s}`
   * - ``qexpy.kb``
     - `Boltzmann constant <https://en.wikipedia.org/wiki/Boltzmann_constant>`_ in :math:`\text{J}\cdot \text{K}^{-1}`
   * - ``qexpy.pi``
     - The ratio of a circle's circumference to its diameter, written as :math:`\pi`

Compound Units
~~~~~~~~~~~~~~

Note that some of the constants are defined in terms of compound units such as :math:`\text{N}`,
:math:`\text{J}`, and :math:`\text{F}`. By default, they are not expanded into their full form
consisting only of base units. Depending on the use case, you might wish to unpack these compound
units differently. For example, :math:`\text{F}` can be expressed in many different ways. You may
choose to express it in terms of :math:`\text{N}`, :math:`\text{m}`, and :math:`\text{V}`:

.. ipython:: python

   import qexpy as q
   q.options.format.style.value = "scientific"
   q.define_unit("F", "N*m/V^2")
   q.eps0
   res = q.eps0 * q.Measurement(4, 0.1, unit="V") ** 2
   res

Define your own constants
~~~~~~~~~~~~~~~~~~~~~~~~~

QExPy only pre-defines some of the most common physical constants. You can easily define
a constant for yourself if needed:

.. ipython:: python

   mp = q.Constant(1.67262192369e-27, 0.00000000051e-27, unit="kg")
   mp
