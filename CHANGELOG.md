# Changelog

## Release 3.0.1

### New Features

* Units now support non-integer exponents, such as `"kg^(1/2)"`

* Users can now define a custom alias for a compound unit. e.g., define `"N"` as `"kg*m/s^2"`, and
  it will be treated as its expanded form during calculations.

* Common physical constants are added and can be access from `qexpy`.
  ```pycon
  >>> q.hbar
  (1.05 +/- 0.00) * 10^-34 [J⋅s]
  ```

### Deprecations

* Support for Python 3.5, 3.6, 3.7, 3.8 are discontinued.

* Switching to dotted-style options like in `pandas`. The old methods of setting package-wide
  options are removed. See documentation for more details.

* Measurements are now semi-immutable. Users will no longer be able to override the value or 
  uncertainty of a recorded measurement.

* Mathematical functions are removed from QExPy. Instead, experimental values and arrays defined 
  in QExPy are now compatible with standard `numpy` functions.
  ```pycon
  >>> m = q.Measurement(5, 0.1)
  >>> q.sqrt(m)  # this has now been removed
  ```
  Instead, simply use functions from `numpy`.
  ```pycon
  >>> np.sqrt(m)
  2.00 +/- 0.02
  ```