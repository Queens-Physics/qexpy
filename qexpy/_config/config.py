"""
The config module holds package-wide configurables and a uniform API
for working with them.

This module is adapted from the pandas source code:
https://github.com/pandas-dev/pandas/blob/main/pandas/_config/config.py

Original license: BSD 3-Clause License

Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc.
and PyData Development Team

All rights reserved.

Copyright (c) 2011-2025, Open source contributors.

"""

import re
from collections.abc import Callable
from functools import wraps
from typing import Any, NamedTuple


class Option(NamedTuple):
    """A data structure that represents a configurable option."""

    key: str
    default: object
    doc: str
    validator: Callable[[object], Any] | None


_registered_options: dict[str, Option] = {}

_global_options: dict[str, Any] = {}


def validate_key(accessor):
    """Validate the key passed to an option accessor."""

    @wraps(accessor)
    def wrapper(*args) -> None:
        try:
            return accessor(*args)
        except KeyError as err:
            raise KeyError(f"No such option: {args[0]}") from err

    return wrapper


@validate_key
def get_option(key: str) -> Any:
    """Retrieve the current value of the specified option.

    Available options can be queried using :func:`~.qexpy.describe_option`.

    Parameters
    ----------
    key : str
        The name of the option.

    Returns
    -------
    Any
        The current value of the specified options.

    Raises
    ------
    KeyError
        If the requested option does not exist.

    """
    root, k = _get_root(key)
    return root[k]


class _OptionsContext:
    """A context manager that restores updated options upon exit."""

    def __init__(self, originals: dict[str, Any]):
        self.originals = originals

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        for k, v in self.originals.items():
            root, k = _get_root(k)
            root[k] = v


@validate_key
def set_option(*args) -> _OptionsContext:
    """Set the value(s) of the specified option(s).

    Parameters
    ----------
    key1, value1, key2, value2, ...
        An even number of arguments to represent (key, value) pairs, or a
        single dictionary with all the option values to be updated.

    Raises
    ------
    KeyError
        If the requested option does not exist

    """

    if len(args) == 1 and isinstance(args[0], dict):
        args = tuple(kv for item in args[0].items() for kv in item)

    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    originals = {}

    for k, v in zip(args[::2], args[1::2], strict=False):
        o = _registered_options[k]
        if o and o.validator:
            o.validator(v)
        full_path = k
        root, k = _get_root(k)
        originals[full_path] = root[k]
        root[k] = v

    return _OptionsContext(originals)


def reset_option(key: str = "") -> None:
    """Reset one or more options to their default values.

    Parameters
    ----------
    key : str, optional
        The name of the option, if empty or ``"all"``, reset all options

    Raises
    ------
    KeyError
        If the requested option does not exist

    """
    keys = _select_keys(key)

    if len(keys) == 0:
        raise KeyError(f"No such option(s) matching {key=}")

    for k in keys:
        set_option(k, _registered_options[k].default)


def describe_option(key: str = ""):
    """Display the description of one or more options in the command line.

    Parameters
    ----------
    key : str, optional
        The key of the option, or a path prefix that matches multiple options.
        If not provided or empty, all options will be listed.

    """
    keys = _select_keys(key)

    if len(keys) == 0:
        raise KeyError(f"No such option(s) matching {key=}")

    print("\n".join([_build_option_description(k) for k in keys]))


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
        set_option(prefix, val)

    def __getattr__(self, key: str) -> Any:
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
        return get_option(prefix)


options = DictWrapper(_global_options)


def register_option(
    key: str,
    default: Any,
    doc: str = "",
    validator: Callable[[object], Any] | None = None,
) -> None:
    """Register a configurable option."""

    key = key.lower()

    if key in _registered_options:
        raise ValueError(f"Option '{key}' has already been registered")

    # Make sure the default value is valid
    if validator:
        validator(default)

    # Walk the nested dict, creating dicts as needed along the path
    path = key.split(".")

    cursor = _global_options
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


def _get_root(key: str) -> tuple[dict[str, Any], str]:
    path = key.split(".")
    cursor = _global_options
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _select_keys(pattern: str) -> list[str]:
    if pattern in _registered_options:
        return [pattern]
    keys = list(_registered_options.keys())
    if pattern in ("", "all"):
        return keys
    return [k for k in keys if re.search(rf"^{pattern}\.", k, re.I)]


def _build_option_description(k: str) -> str:
    option = _registered_options[k]
    s = f"{k} "
    s += "\n".join(option.doc.strip().split("\n"))
    s += f"\n    [default: {option.default}] [currently: {get_option(k)}]"
    return s
