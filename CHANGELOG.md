# Changelog

## Release 3.0.0

### New Features

* QExPy now adopts a pandas-style options API for working with configurations.

### Bug Fixes

* Fixes a bug where unit expressions containing nested brackets are not parsed
  correctly, allowing for more flexibility in how a unit can be specified.

### Breaking Changes

### Deprecations and Removals

* Support for Python versions lower than 3.11 is discontinued.

### Internal Changes

* The QExPy project has migrated to using `uv` as the package manager.
