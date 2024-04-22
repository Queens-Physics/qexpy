"""Functions for parsing and constructing unit strings"""

# pylint: disable=too-few-public-methods

from __future__ import annotations

import re
import warnings
from copy import copy
from fractions import Fraction
from numbers import Real
from typing import Dict, List

import math as m

import qexpy as q

_DOT_STRING = "\N{Dot Operator}"

# Keeps track of user-defined units
_registered_units: Dict[str, Unit] = {}


class Unit(dict):
    """A dictionary of individual units and their exponents"""

    def __init__(self, _dict: Dict):
        super().__init__(_dict.items())

    @classmethod
    def from_string(cls, string: str) -> Unit:
        """Parse a unit string into a Unit object.

        This function converts the string representation of a unit into a binary expression tree,
        then evaluates the tree. The units are parsed to the following rules:

        1. Expressions enclosed in brackets are evaluated first
        2. A unit and its exponent (e.g. "m^2") are always evaluated together
        3. Units connected with implicit multiplication are evaluated together

        For example, with "kg*m^2/s^2A^2". Units and their exponents are automatically grouped
        together, as well as concatenated expressions. Therefore, this expression is read as
        "kg*(m^2)/((s^2)*(A^2))" and ultimately converted to {"kg": 1, "m": 2, "s": -2, "A": -2}.

        Parameters
        ----------

        string : str
            The string representation of a unit

        Returns
        -------

        unit : Unit
            The unit object.

        """
        if not string:
            return Unit({})

        tokens = _split_unit_string(string)
        tree = _construct_expression_tree(tokens)
        return tree.evaluate()

    def __str__(self) -> str:

        if not self:
            return ""

        # Replace part of the unit expression with user-defined aliases if applicable. To avoid
        # making unwanted substitutions, the unit expression is only simplified when the entire
        # expression is a power of a pre-defined unit. For example, "kg^2*m^2/s^4" is simplified
        # to "N^2", but "kg^2*m^2/s^3" will not be converted to "N^2*s"
        unit_dict = copy(self)
        for unit, expression in _registered_units.items():
            exp = _try_pack(unit_dict, expression)
            if exp:
                unit_dict = {unit: exp}
                break

        if q.options.format.style.unit == "fraction":
            return _unit_to_fraction_string(unit_dict)

        return _unit_to_exponent_string(unit_dict)

    def __repr__(self) -> str:
        return str(self)

    def __setitem__(self, key, value):
        raise TypeError("Unit does not support item assignment.")

    def update(self, __m, **_):
        raise TypeError("Unit does not support item assignment.")

    def _unpack(self):  # pylint: disable=protected-access
        """Recursively unpacks user-defined aliases for compound units"""

        try:
            result = {}
            for unit, exp in self.items():
                if unit in _registered_units:
                    unpacked = _registered_units[unit]._unpack()
                    for tok, val in unpacked.items():
                        result[tok] = result.get(tok, 0) + val * exp
                else:
                    result[unit] = result.get(unit, 0) + exp
            return Unit(result)
        except RecursionError as e:
            raise RecursionError(
                "Unable to derive units for the result of this operation, there is likely "
                "circular reference in your custom unit definitions."
            ) from e

    def __add__(self, other: dict) -> Unit:
        assert isinstance(other, Unit)
        _self, _other = self._unpack(), other._unpack()
        if _self and _other and _self != _other:
            warnings.warn("Adding two quantities with mismatching units!")
            return Unit({})
        return Unit(dict(self.items())) if self else Unit(dict(other.items()))

    __radd__ = __add__

    def __sub__(self, other):
        assert isinstance(other, Unit)
        _self, _other = self._unpack(), other._unpack()
        if _self and _other and _self != _other:
            warnings.warn("Subtracting two quantities with mismatching units!")
            return Unit({})
        return Unit(dict(self.items())) if self else Unit(dict(other.items()))

    __rsub__ = __sub__

    def __mul__(self, other):
        assert isinstance(other, Unit)
        if self and not other:
            return Unit(dict(self.items()))
        if not self and other:
            return Unit(dict(other.items()))
        result = {}
        _self, _other = self._unpack(), other._unpack()
        for unit, exp in _self.items():
            result[unit] = exp
        for unit, exp in _other.items():
            result[unit] = result.get(unit, 0) + exp
        result = {name: exp for name, exp in result.items() if exp != 0}
        return Unit(result)

    __rmul__ = __mul__

    def __truediv__(self, other):
        assert isinstance(other, Unit)
        if self and not other:
            return Unit(dict(self.items()))
        if not self and other:
            return Unit({k: -v for k, v in other.items()})
        result = {}
        _self, _other = self._unpack(), other._unpack()
        for unit, exp in _self.items():
            result[unit] = exp
        for unit, exp in _other.items():
            result[unit] = result.get(unit, 0) - exp
        result = {name: exp for name, exp in result.items() if exp != 0}
        return Unit(result)

    def __pow__(self, power):
        assert isinstance(power, Real)
        return Unit({name: exp * power for name, exp in self.items()})

    def __copy__(self):
        return Unit(dict(self.items()))


def define_unit(name: str, unit_str: str):
    """Define an alias for a compound unit

    Assign a unit expression to a name. For example, define ``"N"`` as ``"kg*m/s^2"``. Once a
    unit is defined, it will be treated as its expanded form when performing calculations.

    Parameters
    ----------

    name: str
        The name of the compound unit
    unit_str: str
        The expanded unit expression

    Examples
    --------

    >>> import qexpy as q
    >>> q.define_unit("N", "kg*m/s^2")

    Once a unit is defined, we can assign it to a measurement:

    >>> force = q.Measurement(4, unit="N")
    >>> mass = q.Measurement(2, unit="kg")
    >>> acceleration = force / mass
    >>> print(acceleration)
    2 +/- 0 [m/s^2]

    See Also
    --------

    :py:func:`clear_unit_definitions`

    """
    if not re.match(r"^[a-zA-Z]+$", name):
        raise ValueError("The name of a unit can only contain letters!")
    _registered_units[name] = Unit.from_string(unit_str)


def clear_unit_definitions():
    """Deletes all user-defined unit aliases

    See Also
    --------

    :py:func:`define_unit`

    """
    _registered_units.clear()


def _split_unit_string(unit_string: str) -> List:
    """Split the unit string into a list of tokens

    A token can be a single unit, an operator such as "*", "/" or "^", a number representing the
    exponent of a unit, or a list of grouped tokens.

    For example, the unit "kg*m/s^2A^2" would be parsed into:
    ["kg", "*", "m", "/", [["s", "^", "2"], "*", ["A", "^", "2"]]]

    """

    unit_string = unit_string.replace(_DOT_STRING, "*")

    raw_tokens = []  # The raw list of tokens
    tokens = []  # The final list of tokens

    p_unit = re.compile(r"[a-zA-Z]+")  # A unit must only consist of letters
    p_unit_with_exponent = re.compile(
        r"[a-zA-Z]+"  # The name of the unit
        r"((\^-?\d+/\d+)|"  # The exponent in fraction form
        r"(\^-?\d+(\.\d+)?)|"  # The exponent in decimal form
        r"(\^\(-?\d+(/\d+)?\))|"  # The exponent written inside brackets (fraction)
        r"(\^\(-?\d+(\.\d+)?\)))"  # The exponent written inside brackets (decimal)
    )
    p_operator = re.compile(r"[/*]")  # Only acceptable operators are '*' and '/'
    p_bracket_enclosed_expression = re.compile(r"\(.*?\)")
    p_token = re.compile(
        rf"{p_unit_with_exponent.pattern}|"
        rf"{p_unit.pattern}|"
        rf"{p_operator.pattern}|"
        rf"{p_bracket_enclosed_expression.pattern}"
    )  # Regular expression for all acceptable tokens

    # Check if the input only consists of valid token strings
    if not re.fullmatch(rf"({p_token.pattern})+", unit_string):
        raise ValueError(f'"{unit_string}" is not a valid unit expression.')

    # For every token found, process it and append it to the list
    for result in p_token.finditer(unit_string):
        token = result.group()
        if p_bracket_enclosed_expression.fullmatch(token):
            # If the token is a bracket enclosed expression, recursively parse its content and
            # append them to the list of tokens as a list
            raw_tokens.append(_split_unit_string(token[1:-1]))
        elif p_unit_with_exponent.fullmatch(token):
            # Group a unit with its exponent and append to the list as a whole
            exp = token.split("^")
            if p_bracket_enclosed_expression.fullmatch(exp[1]):
                exp[1] = exp[1][1:-1]
            power = Fraction(exp[1])
            raw_tokens.append([exp[0], "^", float(power)])
        else:
            raw_tokens.append(token)

    # At this stage, except for when explicit brackets are present, no grouping of tokens has
    # taken place yet. The following code checks for expressions concatenated with implicit
    # multiplication, and groups them together (also adding a multiplication operator). The
    # preceding_operator flag keeps track of whether there is an operator present between the
    # current token being processed and its predecessor.
    has_preceding_operator = True

    for token in raw_tokens:
        if has_preceding_operator:
            # If there is an operator preceding, add the current token to the list of tokens
            tokens.append(token)
            has_preceding_operator = False
        elif isinstance(token, str) and p_operator.fullmatch(token):
            # If there is no operator preceding, that is, the last token was a unit expression,
            # and the current token is an operator, add the current token to the list of tokens
            tokens.append(token)
            has_preceding_operator = True
        else:
            # Apply implicit multiplication when there is no preceding operator, that is, the
            # current token is a unit expression closely following another unit expression.
            last_token = tokens.pop()
            tokens.append([last_token, "*", token])
            has_preceding_operator = False

    return tokens


class _Expression:
    """A node in a binary expression tree representing a unit expression"""

    def evaluate(self) -> Unit | float:
        """Evaluates the expression to a Unit object."""
        return Unit({})  # pragma: no cover


class _BinOp(_Expression):
    """A node in an expression tree representing a binary operator"""

    op: str
    left: _Expression
    right: _Expression

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self) -> Unit:

        if self.op == "*":
            return self.left.evaluate() * self.right.evaluate()
        elif self.op == "/":
            return self.left.evaluate() / self.right.evaluate()
        elif self.op == "^":
            return self.left.evaluate() ** self.right.evaluate()

        raise TypeError(f"Unsupported operator in unit string: {self.op}")  # pragma: no cover


class _Number(_Expression):
    """A node in an expression tree representing a number"""

    value: Real

    def __init__(self, value):
        self.value = value

    def evaluate(self) -> Real:
        return self.value


class _Unit(_Expression):
    """A node in an expression tree representing an individual unit"""

    name: str

    def __init__(self, name):
        self.name = name

    def evaluate(self) -> Unit:
        return Unit({self.name: 1})


def _construct_expression_tree(tokens: List) -> _Expression:
    """Construct a binary expression tree with a list of tokens

    The algorithm that builds the expression tree is called recursive descent, which makes use of
    two stacks: the operator stack and the operand stack. For each new token, if the token is an
    operator, it is compared with the current top of the operator stack. The operator stack is
    maintained so that the top of the stack has the highest priority in order of operations. If
    the current top has higher priority than the operator being processed, it is popped from the
    stack, used as the root node to build an expression tree with the top two operands in the
    operand stack, which is then pushed back into the operand stack.

    For details regarding this algorithm, see the reference below.

    Reference
    ---------

    Parsing Expressions by Recursive Descent - Theodore Norvell (C) 1999
    https://www.engr.mun.ca/~theo/Misc/exp_parsing.htm

    Parameters
    ----------

    tokens : str
        The list of tokens to process.

    Returns
    -------

    exp : _Expression
        The root node of a binary expression tree representing the set of units.

    """

    # Initialize the two stacks
    operand_stack: List = []
    operator_stack: List[_Expression | str] = ["base"]

    # Define operator precedence, a higher value corresponds to a higher precedence.
    precedence = {"base": 0, "*": 1, "/": 1, "^": 2}

    def __construct_sub_tree() -> _Expression:
        right = operand_stack.pop()
        left = operand_stack.pop()
        operator = operator_stack.pop()
        return _BinOp(operator, left, right)

    # Push all tokens into the two stacks, make subtrees if necessary
    for token in tokens:
        top = operator_stack[-1]
        if isinstance(token, list):
            # Recursively make subtrees with grouped expressions
            operand_stack.append(_construct_expression_tree(token))
        elif token in precedence and precedence[token] > precedence[top]:
            operator_stack.append(token)
        elif token in precedence and precedence[token] <= precedence[top]:
            operand_stack.append(__construct_sub_tree())
            operator_stack.append(token)
        elif isinstance(token, Real):
            operand_stack.append(_Number(token))
        else:
            operand_stack.append(_Unit(token))

    # Build the final tree from all tokens and subtrees left in the stacks
    while len(operator_stack) > 1:
        operand_stack.append(__construct_sub_tree())

    return operand_stack[0] if operand_stack else _Expression()


def _number_of_decimals(value: float) -> int:
    """Calculates the correct number of decimal places to show"""

    order = m.floor(m.log10(abs(value) % 1))
    number_of_decimals = -order + 2 - 1
    return number_of_decimals if number_of_decimals > 0 else 0


def _construct_exponent(exponent: float) -> str:
    """Construct the string representation of a unit exponent"""

    # Gets rid of the error caused by the machine epsilon
    fraction = Fraction(exponent).limit_denominator(int(1e10))

    # Construct the string representation of the exponent
    if fraction.numerator == 1 and fraction.denominator == 1:
        return ""  # do not print power of 1 as it's implied
    if fraction.denominator == 1:
        return f"^{str(fraction.numerator)}"

    # When the fraction form is too complicated, keep decimal form, but only keep two
    # significant figures after the decimal point for simplicity.
    if fraction.denominator > 10:
        return f"^{exponent:.{_number_of_decimals(exponent)}f}"

    return f"^({str(fraction)})"


def _unit_to_fraction_string(units: Dict[str, float]) -> str:
    """Construct the string representation of a unit in the fraction style

    Parameters
    ----------

    units : Dict
        A dictionary of units and their exponents

    Returns
    -------
    The string representation of the unit in the fraction style

    """

    numerator = [f"{unit}{_construct_exponent(exp)}" for unit, exp in units.items() if exp > 0]
    denominator = [f"{unit}{_construct_exponent(-exp)}" for unit, exp in units.items() if exp < 0]
    numerator_string = _DOT_STRING.join(numerator) if numerator else "1"
    denominator_string = _DOT_STRING.join(denominator)

    if not denominator:
        return numerator_string

    if len(denominator) > 1:
        # For multiple units in the denominator, use bracket to avoid ambiguity
        denominator_string = f"({denominator_string})"

    # Combine numerator and denominator
    return f"{numerator_string}/{denominator_string}"


def _unit_to_exponent_string(units: Dict[str, float]) -> str:
    """Construct the string representation of a unit in the exponent style

    Parameters
    ----------
    units : Dict
        A dictionary of units and their exponents

    Returns
    -------

    The string representation of the unit in the exponent style

    """
    return _DOT_STRING.join(f"{unit}{_construct_exponent(exp)}" for unit, exp in units.items())


def _try_pack(unit: Dict[str, float], pre_defined: Dict) -> float:
    """Try packing a unit into some power of a predefined compound unit"""

    exponent = 0
    for name, exp in unit.items():
        pre_exp = pre_defined.get(name, 0)
        if not pre_exp:
            return 0  # unit non-existing in predefined units
        if exponent and exponent != exp / pre_exp:
            return 0  # exponent different from previous
        if not exponent:
            exponent = exp / pre_exp
    for name, exp in pre_defined.items():
        if not unit.get(name, 0):
            return 0
    return exponent
