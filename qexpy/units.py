"""Defines the data structure used to store units."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from warnings import warn

from qexpy.format import format_unit_as_fraction, format_unit_as_product

from ._config import options
from .typing import Number

_DOT_STRING = "\N{DOT OPERATOR}"
_LEFT_BRACKET = "("
_RIGHT_BRACKET = ")"
_MULTIPLIER = "*"
_IMPLICIT_MULTIPLIER = "_*"
_DIVISOR = "/"
_CARET = "^"


# Keeps track of user-defined aliases for compound units.
_unit_aliases: dict[str, Unit] = {}


class Unit:
    """A data structure that represents a unit."""

    _unit: dict[str, Number]

    def __new__(cls, unit):  # noqa: D102
        if isinstance(unit, Unit):
            return unit
        if isinstance(unit, (str, dict)):
            return super().__new__(cls)
        raise TypeError("The unit must be specified with a str or a dict.")

    def __init__(self, unit: str | dict[str, Number] | Unit):
        if isinstance(unit, dict):
            self._unit = unit
        else:  # safe to assume that the unit is a string at this point.
            assert isinstance(unit, str)
            self._unit = _parse_unit_str(unit)

    def __str__(self) -> str:
        if not self:
            return ""

        _unit = self._unit
        for name, unit in _unit_aliases.items():
            if exp := _unit_log(unit._unit, self._unit):
                _unit = {name: exp}
                break

        if options.format.unit == "fraction":
            return format_unit_as_fraction(_unit)

        return format_unit_as_product(_unit)

    __repr__ = __str__

    def __getitem__(self, key) -> Number:
        return self._unit[key]

    def __bool__(self) -> bool:
        return bool(self._unit)

    def __eq__(self, other) -> bool:
        if isinstance(other, Unit):
            return self._resolve()._unit == other._resolve()._unit
        if isinstance(other, dict):
            return self._unit == other
        return False

    def __add__(self, other) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        if self and other and self != other:
            warn("Adding two quantities with mismatching units!", stacklevel=2)
            return Unit({})
        return self if self else other

    def items(self):
        """Return the individual units and their exponents."""
        return self._unit.items()

    __radd__ = __add__

    def __sub__(self, other) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        if self and other and self != other:
            warn("Subtracting quantities with mismatching units!", stacklevel=2)
            return Unit({})
        return self if self else other

    __rsub__ = __sub__

    def __mul__(self, other) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        if not self:
            return other
        if not other:
            return self
        result = defaultdict(Fraction)
        for u, p in self._resolve().items():
            result[u] += Fraction(p)
        for u, p in other._resolve().items():
            result[u] += Fraction(p)
        return Unit({k: v for k, v in result.items() if v != 0})

    __rmul__ = __mul__

    def __truediv__(self, other) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        if not self:
            return Unit({k: -v for k, v in other.items()})
        if not other:
            return self
        result = defaultdict(Fraction)
        for u, p in self._resolve().items():
            result[u] += Fraction(p)
        for u, p in other._resolve().items():
            result[u] -= Fraction(p)
        return Unit({k: v for k, v in result.items() if v != 0})

    def __rtruediv__(self, other):
        if other == 1:
            return Unit({k: -v for k, v in self.items()})
        return NotImplemented

    def __pow__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        return Unit({k: Fraction(v) * Fraction(other) for k, v in self.items()})

    def __rpow__(self, other):
        return NotImplemented

    def _resolve(self) -> Unit:
        """Resolve all unit aliases into fundamental units."""

        try:
            result = {}
            for name, exp in self.items():
                if name in _unit_aliases:
                    unpacked = _unit_aliases[name]._resolve()
                    for tok, val in unpacked.items():
                        result[tok] = result.get(tok, 0) + val * exp
                else:
                    result[name] = result.get(name, 0) + exp
            return Unit(result)
        except RecursionError as e:
            raise RecursionError(
                "Recursion limit reached when attempting to resolve units. "
                "There is likely circular reference in your unit aliases. "
            ) from e


UnitLike = str | dict | Unit


def define_unit(name: str, unit: UnitLike):
    """Define an alias for a compound unit.

    Parameters
    ----------
    name: str
        The name of the compound unit.
    unit: UnitLike
        The full definition of the unit.

    """
    if not re.fullmatch(r"[a-zA-Z]+", name):
        raise ValueError("The alias can only contain letters!")
    _unit_aliases[name] = Unit(unit)


def clear_unit_aliases():
    """Delete all unit aliases.

    See Also
    --------
    :py:func:`define_unit`

    """
    _unit_aliases.clear()


def _parse_unit_str(unit_str: str) -> dict[str, Number]:
    """Parse the string representation of a unit to a dictionary.

    Parameters
    ----------
    unit_str : str
        The string representation of a unit.

    Returns
    -------
    unit : dict
        A dictionary mapping individual units to their powers.
    """

    if not unit_str:
        return {}

    tokens = _tokenize_unit(unit_str)
    tree = _construct_expression_tree(tokens)
    unit = tree.evaluate()
    assert isinstance(unit, Unit)
    return unit._unit  # We can safely assume that unit is a Unit here.


# Matches an individual token in a unit expression.
_TOKEN_PATTERN = re.compile(
    r"""
    (?P<UNIT>[a-zA-Z]+) |  # a single unit
    (?P<EXPONENT>\^(?:  # an exponent of a unit
        (?P<NUMBER>
            (-?\d+/\d+) |  # expressed in terms of a fraction
            (-?\d+(\.\d+)?)  # expressed as a decimal number
        ) | 
        (?:\(
            (?P<NUMBER_IN_BRACKETS>  # expression enclosed in brackets
                (-?\d+/\d+) |  # a fraction
                (-?\d+(\.\d+)?)  # a decimal number
            )
        \))
    )) |
    (?P<OP>[\*/]) |  # the only valid operators are '*', '/'
    (?P<BRACKET>[\(\)]) |  # matches brackets
    (?P<ONE>1/)  # to support syntax like 1/s
    """,
    re.X,
)


def _tokenize_unit(unit_str: str) -> list:
    """Parse the unit string into a nested list of tokens."""

    unit_str = unit_str.replace(_DOT_STRING, _MULTIPLIER)

    # Check if the input only consists of valid token strings
    if not re.fullmatch(rf"({_TOKEN_PATTERN.pattern})+", unit_str, re.X):
        raise ValueError(f'"{unit_str}" is not a valid unit expression.')

    # Tracks whether the previous token was a unit expression. This includes
    # inidividual units (e.g., "kg"), a unit with an exponent (e.g., "m^2"),
    # or bracket-enclosed expressions in general. If a unit expression is
    # encountered following another unit expression, they are considered to
    # be connected via implicit muliplication. For example "kg^1m^2" is
    # interpreted as "(kg^1*m^2)". This implicit muliplication is assumed to
    # have higher precedence over explicit muliplication and division.
    has_preceding_unit = False

    tokens = []
    stack = [tokens]  # used to track bracket-enclosed expressions
    for m in _TOKEN_PATTERN.finditer(unit_str):
        # Individual units can be added directly to the list of tokens.
        if token := m.group("UNIT"):
            if has_preceding_unit:
                stack[-1].append(_IMPLICIT_MULTIPLIER)
            stack[-1].append(token)
            has_preceding_unit = True

        elif token := m.group("OP"):
            # An operator must follow a unit expression.
            if not has_preceding_unit:
                raise ValueError(f"{unit_str} is not a valid unit expression.")
            stack[-1].append(token)
            has_preceding_unit = False

        elif m.group("EXPONENT"):
            if not has_preceding_unit:
                # An exponent expression only makes sense following a unit.
                raise ValueError(f"{unit_str} is not a valid unit expression.")
            exponent = m.group("NUMBER") or m.group("NUMBER_IN_BRACKETS")
            preceding_unit = stack[-1].pop()
            stack[-1].append([preceding_unit, _CARET, Fraction(exponent)])

        # A left bracket marks the start of a sublist of tokens
        elif (token := m.group("BRACKET")) == _LEFT_BRACKET:
            if has_preceding_unit:
                stack[-1].append(_IMPLICIT_MULTIPLIER)
            stack.append([])
            has_preceding_unit = False

        # A right bracket closes a sublist of tokens
        elif token == _RIGHT_BRACKET:
            # The stack initially has only one list, which is the list of all
            # tokens at the top level. If a right bracket is reached and there
            # is no sublist available on the top of the stack, raise an error.
            if len(stack) == 1:
                raise ValueError(f"{unit_str} contains unmatched brackets.")
            # An expression enclosed in a bracket cannot end with an operator.
            if not has_preceding_unit:
                raise ValueError(f"{unit_str} is not a valid unit expression.")
            expression_in_bracket = stack.pop()
            stack[-1].append(expression_in_bracket)
            has_preceding_unit = True

        elif token := m.group("ONE"):
            if len(stack[-1]) > 0:
                raise ValueError(f"{unit_str} is not a valid unit expression.")
            stack[-1].append(1)
            stack[-1].append(_DIVISOR)

    # If there is an unclosed bracket left in the stack.
    if len(stack) > 1:
        raise ValueError(f"{unit_str} contains unmatched brackets.")

    # The overall unit expression cannot end on an operator.
    if not has_preceding_unit:
        raise ValueError(f"{unit_str} is not a valid unit expression.")

    return tokens


class _Expression:
    """An expression tree that represents a unit."""

    def evaluate(self) -> Unit | Number:
        """Evaluate the expression."""
        return Unit({})


@dataclass(frozen=True)
class _BinaryOp(_Expression):
    """A binary operator."""

    op: str
    left: _Expression
    right: _Expression

    def __post_init__(self):
        assert self.op in (_MULTIPLIER, _IMPLICIT_MULTIPLIER, _DIVISOR, _CARET)

    def evaluate(self) -> Unit:
        left, right = self.left.evaluate(), self.right.evaluate()
        if self.op in (_MULTIPLIER, _IMPLICIT_MULTIPLIER):
            assert isinstance(left, Unit) and isinstance(right, Unit)
            return left * right
        if self.op == _DIVISOR:
            return left / right
        return left**right


@dataclass(frozen=True)
class _Number(_Expression):
    """A numerical value."""

    value: Number

    def evaluate(self) -> Number:
        return self.value


@dataclass(frozen=True)
class _Unit(_Expression):
    """A single unit."""

    name: str

    def evaluate(self) -> Unit:
        return Unit({self.name: 1})


def _unit_log(base: dict[str, Number], unit: dict[str, Number]) -> Number:
    """Find n such that unit is represented as base raised a power of n."""

    base = {k: v for k, v in base.items() if v != 0}
    unit = {k: v for k, v in unit.items() if v != 0}

    if base.keys() != unit.keys():
        return 0

    scale = 0
    for k, v in base.items():
        new_scale = unit[k] / v
        if scale and scale != new_scale:
            return 0
        scale = new_scale
    return scale


def _construct_expression_tree(tokens: list) -> _Expression:
    """Construct an expression tree with a list of tokens."""

    operand_stack: list[_Expression] = []
    operator_stack: list[str] = []
    precedence = {
        _MULTIPLIER: 1,
        _DIVISOR: 1,
        _IMPLICIT_MULTIPLIER: 2,
        _CARET: 3,
    }

    def _construct_sub_tree() -> _Expression:
        right = operand_stack.pop()
        left = operand_stack.pop()
        operator = operator_stack.pop()
        return _BinaryOp(operator, left, right)

    def _process_new_operator(op: str):
        last_op = operator_stack[-1] if operator_stack else ""
        last_precedence = precedence.get(last_op, 0)
        if precedence[op] < last_precedence:
            operand_stack.append(_construct_sub_tree())
        operator_stack.append(op)

    for token in tokens:
        if isinstance(token, list):
            # Recursively make subtrees with grouped expressions
            operand_stack.append(_construct_expression_tree(token))
        elif token in precedence:
            _process_new_operator(token)
        elif isinstance(token, Number):
            operand_stack.append(_Number(token))
        else:  # the token must be a unit expression here.
            operand_stack.append(_Unit(token))

    # Build the final tree from all tokens and subtrees left in the stacks.
    while len(operator_stack) > 0:
        operand_stack.append(_construct_sub_tree())

    return operand_stack[0] if operand_stack else _Expression()
