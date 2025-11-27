"""Defines the dispatcher for performing operations."""

# ruff: noqa: D103

from collections.abc import Callable
from functools import singledispatch
from typing import Any


@singledispatch
def add(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def subtract(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def rsubtract(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def multiply(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def divide(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def rdivide(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def power(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def rpower(var1, var2) -> Any:
    raise NotImplementedError


@singledispatch
def negate(var) -> Any:
    raise NotImplementedError


@singledispatch
def array_ufunc(var, ufunc: Callable, *inputs):
    raise NotImplementedError
