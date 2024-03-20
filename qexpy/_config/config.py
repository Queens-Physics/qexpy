"""Defines the data structure that stores package-wide configurations

References
----------
https://github.com/pandas-dev/pandas/blob/main/pandas/_config/config.py

"""

from __future__ import annotations

import re
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
        The name of the option to reset, if empty or ``all``, resets all options

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
        If empty, all options will be listed.

    """

    keys = _select_keys(key)

    if len(keys) == 0:
        raise KeyError(f"No such option(s) matching {key=}")

    print("\n".join([_build_option_description(k) for k in keys]))


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
