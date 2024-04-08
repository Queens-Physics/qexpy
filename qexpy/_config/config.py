"""Defines the data structure that stores package-wide configurations

References
----------
https://github.com/pandas-dev/pandas/blob/main/pandas/_config/config.py

"""

from __future__ import annotations

import re
from contextlib import contextmanager
import keyword
from functools import wraps
from numbers import Real
from typing import NamedTuple, Callable, Any, Iterable, Dict

_registered_options: Dict[str, Option] = {}

_global_config: Dict[str, Any] = {}


class Option(NamedTuple):
    """The data structure that represents an option"""

    key: str
    default: object
    doc: str
    validator: Callable[[object], Any] | None


class DictWrapper:
    """Provide attribute-style access to a nested dictionary."""

    def __init__(self, d: dict[str, Any], prefix: str = "") -> None:
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        if key in self.d and not isinstance(self.d[key], dict):
            set_option(prefix, val)
        else:
            raise KeyError("You can only set the value of existing options")

    def __getattr__(self, key: str):
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        try:
            v = object.__getattribute__(self, "d")[key]
        except KeyError as err:
            raise KeyError("No such option") from err
        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return get_option(prefix)


options = DictWrapper(_global_config)


def register_option(
    key: str,
    default: object,
    doc: str = "",
    validator: Callable[[object], Any] | None = None,
) -> None:
    """Register an option in the package-wide qexpy config object

    Parameters
    ----------

    key : str
        The name of the option
    default : object
        The default value of the option
    doc: str, optional
        A description of the option
    validator : Callable, optional
        A function that should raise an exception if the value is invalid

    """

    # Make sure the default value is valid
    if validator:
        validator(default)

    # Walk the nested dict, creating dicts as needed along the path
    path = key.split(".")

    # Validate keys
    for p in path:
        if keyword.iskeyword(p):
            raise ValueError(f"{p} is a python keyword")

    cursor = _global_config
    for p in path[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError("Path prefix is already an option")
        if p not in cursor:
            cursor[p] = {}
        cursor = cursor[p]

    if not isinstance(cursor, dict):
        raise ValueError("Path prefix is already an option")

    if path[-1] in cursor and isinstance(cursor[path[-1]], dict):
        raise ValueError("Path is already a prefix to other options")

    cursor[path[-1]] = default

    _registered_options[key] = Option(key, default, doc, validator)


def validate_key(accessor):
    """Decorator to validate the key passed to an option accessor"""

    @wraps(accessor)
    def wrapper(*args) -> None:
        try:
            return accessor(*args)
        except KeyError as err:
            raise KeyError(f"No such option: {args[0]}") from err

    return wrapper


@validate_key
def get_option(key: str) -> Any:
    """Gets the value of an option

    Parameters
    ----------

    key : str
        The name of the option

    Returns
    -------

    value
        The current setting for the requested option

    Examples
    --------

    The option can be accessed using a string:

    >>> import qexpy as q
    >>> q.get_option('error.mc.sample_size')
    100000

    Or using the dotted-style, as an attribute:

    >>> q.options.error.mc.sample_size
    100000

    """
    root, k = _get_root(key)
    return root[k]


@validate_key
def set_option(key: str, value) -> None:
    """Sets the value of an option

    Parameters
    ----------

    key : str
        The name of the option
    value : Any
        The value to set

    Examples
    --------

    >>> import qexpy as q
    >>> q.get_option('error.mc.sample_size')
    100000
    >>> q.set_option('error.mc.sample_size', 200000)
    >>> q.get_option('error.mc.sample_size')
    200000

    Options can also be modified using the dotted-style, as an attribute:

    >>> q.options.error.mc.sample_size = 500000
    >>> q.options.error.mc.sample_size
    500000

    """
    o = _registered_options[key]

    if o and o.validator:
        o.validator(value)

    root, k = _get_root(key)
    root[k] = value


def reset_option(key: str = "") -> None:
    """Resets an option or all options to their default values

    Parameters
    ----------

    key : str, optional
        The name of the option to reset, if empty or ``"all"``, resets all options

    Examples
    --------

    >>> import qexpy as q
    >>> q.set_option('error.mc.sample_size', 200000)
    >>> q.set_option('error.method', 'monte-carlo')
    >>> q.set_option('format.style.value','scientific')

    After changing options, you can reset one of them to their default values:

    >>> q.reset_option('error.mc.sample_size')
    >>> q.get_option('error.mc.sample_size')
    100000

    Or reset everything to their default values:

    >>> q.reset_option()
    >>> q.get_option('error.method')
    'derivative'
    >>> q.get_option('format.style.value')
    "default"

    """

    keys = _select_keys(key)

    if len(keys) == 0:
        raise KeyError(f"No such option(s) matching {key=}")

    for k in keys:
        set_option(k, _registered_options[k].default)


def describe_option(key: str = ""):
    """Prints the description of one or more options

    Parameters
    ----------

    key: str, optional
        The key of the option, or a path prefix that matches multiple options.
        If not provided or empty, all options will be listed.

    Examples
    --------

    >>> import qexpy as q
    >>> q.describe_option()
    error.mc.sample_size : int
        The sample size used in Monte Carlo error propagation.
        [default: 100000] [currently: 100000]
    error.method : {"derivative", "monte-carlo"}
        The preferred method of error propagation.
        [default: derivative] [currently: derivative]
    format.precision.mode : {"value", "error"}
        Specifies whether the number of significant figures is imposed on the value
        or the error.
        [default: error] [currently: error]
    format.precision.sig_fig : int
        The number of significant figures keep when displaying values.
        [default: 1] [currently: 1]
    format.style.latex : bool
        Whether values are formatted with LaTeX grammar.
        [default: False] [currently: False]
    format.style.unit : {"fraction", "exponent"}
        How units are formatted, as a fraction, e.g., "m/s^2", or with exponents,
        e.g., "m⋅s^-2".
        [default: fraction] [currently: fraction]
    format.style.value : {"default", "scientific"}
        How value strings are formatted, by default, e.g., "2.1 +/- 0.5", or using
        the scientific notation, e.g., "(1.200 +/- 0.004) * 10^5".
        [default: default] [currently: default]

    """

    keys = _select_keys(key)

    if len(keys) == 0:
        raise KeyError(f"No such option(s) matching {key=}")

    print("\n".join([_build_option_description(k) for k in keys]))


@contextmanager
def option_context(*args):
    """
    Context manager to set options in a ``with`` statement.

    Parameters
    ----------
    *args
        An even amount of arguments which will be interpreted as (key, value) pairs.

    Examples
    --------

    Use this method to temporarily set options in a ``with`` statement:

    >>> import qexpy as q
    >>> with q.option_context(
    ...     "format.style.value",
    ...     "scientific",
    ...     "format.precision.sig_fig",
    ...     2,
    ...     "format.precision.mode",
    ...     "error",
    ... ):
    ...    m = q.Measurement(2123, 13)
    ...    print(m)
    (2.123 +/- 0.013) * 10^3

    You can see that outside of the context, the global options have not changed:

    >>> print(m)
    2120 +/- 10

    """
    if len(args) % 2 != 0 or len(args) < 2:
        raise ValueError(
            "Provide an even amount of arguments as option_context(key, val, key, val...)."
        )

    ops = tuple(zip(args[::2], args[1::2]))
    try:
        undo = tuple((pat, get_option(pat)) for pat, val in ops)
        for pat, val in ops:
            set_option(pat, val)
        yield
    finally:
        for pat, val in undo:  # pylint: disable=used-before-assignment
            set_option(pat, val)


def _select_keys(pattern: str) -> list[str]:
    """Selects one or more keys matching a pattern"""

    if pattern in _registered_options:
        return [pattern]

    keys = sorted(_registered_options.keys())
    if pattern in ("", "all"):
        return keys

    return [k for k in keys if re.search(rf"^{pattern}\.", k, re.I)]


def _get_root(key: str) -> tuple[dict[str, Any], str]:
    """Gets the root of the sub-dictionary"""

    path = key.split(".")
    cursor = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _build_option_description(k: str) -> str:
    """Builds a formatted description of a registered option and prints it"""

    option = _registered_options[k]

    s = f"{k} "
    s += "\n".join(option.doc.strip().split("\n"))
    s += f"\n    [default: {option.default}] [currently: {get_option(k)}]"

    return s


def is_one_of_factory(legal_values: Iterable) -> Callable[[Any], None]:
    """Produce a function that checks if a value is one of the legal values"""

    def inner(x) -> None:
        if x not in legal_values:
            vals = [str(val) for val in legal_values]
            msg = f"Value must be one of {'|'.join(vals)}"
            raise ValueError(msg)

    return inner


def is_positive_integer(x) -> None:
    """Checks if a value is a positive integer"""

    if not isinstance(x, int) or x <= 0:
        raise ValueError(f"Value must be a positive integer: {x}")


def is_boolean(x) -> None:
    """Checks if a value is a boolean"""

    if not isinstance(x, bool):
        raise TypeError(f"Value must be a boolean: {x}")


def is_tuple_of_floats(x) -> None:
    """Checks if a value is a tuple of floats"""

    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise TypeError(f"Value must be a tuple of length 2: {x}")

    if not all(isinstance(v, Real) for v in x):
        raise ValueError(f"Value must be a tuple of floats: {x}")
