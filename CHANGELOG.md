# Changelog

## Release 3.0.1

### New Features

* Units now support non-integer exponents.
* You can now define a custom alias for a compound unit. e.g., define `"N"` as `"kg*m/s^2"`, and
  it will be treated as its expanded form during calculations.

### Breaking Changes

* Support for Python 3.5, 3.6, 3.7, 3.8 are discontinued.
* Switching to dotted-style options like in `pandas`. The old way of setting package-wide options
  are removed. See documentation for more details.
