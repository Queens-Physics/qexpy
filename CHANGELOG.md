# Changelog

## Release 3.0.0

### New Features

* QExPy now adopts a pandas-style options API for working with configurations.

* Users can now define custom aliases for compound units using `define_unit`,
  e.g., define ``"N"`` as an alias for ``"kg*m/s^2"``.

### Bug Fixes

* Fixed various bugs with parsing a unit string, e.g., when nested brackets are
  present or with non-integer exponents.

### Breaking Changes

* The ``Measurement`` is now immutable. A measurement can no longer be modified
  post instantiation, as it sometimes leads to ambiguous behaviour.

### Deprecations and Removals

* Support for Python versions lower than 3.11 is discontinued.

### Internal Changes

* The QExPy project has migrated to using `uv` as the package manager.
