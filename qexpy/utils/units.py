"""Functions for parsing and constructing unit strings"""

from __future__ import annotations

import re
import warnings
from copy import copy
from fractions import Fraction
from numbers import Real
from typing import Dict, List

_DOT_STRING = "\N{Dot Operator}"


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

    def __setitem__(self, key, value):
        return TypeError("Unit does not support item assignment.")

    def update(self, __m, **kwargs):
        return TypeError("Unit does not support item assignment.")

    def __hash__(self):
        return hash(frozenset(self.items()))

    def __str__(self) -> str:
        pass

    def __add__(self, other: dict) -> Unit:
        if not isinstance(other, Unit):
            raise TypeError("Can only add units with other units.")
        if self and other and self != other:
            warnings.warn("Adding two quantities with mismatching units!")
        return Unit(dict(self.items())) if self else Unit(dict(other.items()))

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Unit):
            raise TypeError("Can only subtract units with other units.")
        if self and other and self != other:
            warnings.warn("Subtracting two quantities with mismatching units!")
        return Unit(dict(self.items())) if self else Unit(dict(other.items()))

    __rsub__ = __sub__

    def __mul__(self, other):
        if not isinstance(other, Unit):
            raise TypeError("Can only multiply units with other units.")
        result = {}
        for unit, exp in self.items():
            result[unit] = exp
        for unit, exp in other.items():
            result[unit] = result.get(unit, 0) + exp
        return Unit(result)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Unit):
            raise TypeError("Can only divide units with other units.")
        result = {}
        for unit, exp in self.items():
            result[unit] = exp
        for unit, exp in other.items():
            result[unit] = result.get(unit, 0) - exp
        return Unit(result)

    def __rtruediv__(self, other):
        if not isinstance(other, Unit):
            raise TypeError("Can only divide units with other units.")
        result = {}
        for unit, exp in other.items():
            result[unit] = exp
        for unit, exp in self.items():
            result[unit] = result.get(unit, 0) + exp
        return Unit(result)

    def __pow__(self, power):
        if not isinstance(power, Real):
            raise TypeError("Can only take a unit to a numerical exponent.")
        return Unit({name: exp * power for name, exp in self.items()})

    def __copy__(self):
        return Unit(dict(self.items()))


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
        return Unit({})


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

        raise TypeError(f"Unsupported operator in unit string: {self.op}")


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
