.. _user_guide.constants:

=========
Constants
=========

The following physical constants are defined in qexpy:

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

Define your own constants
~~~~~~~~~~~~~~~~~~~~~~~~~

qexpy only pre-defines some of the most common physical constants. You can easily define
a constant for yourself if needed:

.. ipython:: python

   import qexpy as q
   q.options.format.value = 'scientific'
   mp = q.Constant(1.67262192369e-27, 0.00000000051e-27, unit="kg")
   mp
