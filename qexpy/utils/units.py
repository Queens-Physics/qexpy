"""Internal module used for unit parsing and propagation"""

import re
import warnings

from typing import Dict, List, Union
from collections import namedtuple, OrderedDict
from qexpy.settings import UnitStyle
from copy import deepcopy
from fractions import Fraction

import qexpy.settings as sts
import qexpy.settings.literals as lit

# The standard character used in a dot multiply expression
DOT_STRING = "⋅"

# A sub-tree in a binary expression tree representing a unit expression. The "operator" is
# the root node of the sub-tree, and the "left" and "right" points to the two branches. The
# leaf nodes of a unit expression tree are either unit strings or their powers.
Expression = namedtuple("Expression", "operator, left, right")


def parse_unit_string(unit_string: str) -> Dict[str, int]:
    """Decodes the string representation of a set of units

    This function parses the unit string into a binary expression tree, evaluate the tree to
    find all units present in the string and their powers, which is then stored in a Python
    dictionary object.

    The units are parsed to the following rules:
        1. Expressions enclosed in brackets are evaluated first
        2. A unit with its power (e.g. "m^2") are always evaluated together
        3. Expressions connected with implicit multiplication are evaluated together

    For example, "kg*m^2/s^2A^2" would be decoded to: {"kg": 1, "m": 2, "s": -2, "A": -2}

    Args:
        unit_string (str): The string to be parsed

    Returns:
        A dictionary object that stores the power of each unit in the expression

    """
    tokens = __parse_unit_string_to_list(unit_string)
    ast = __construct_expression_tree_with_list(tokens)
    return __evaluate_unit_tree(ast)


def construct_unit_string(units: Dict[str, int]) -> str:
    """Constructs the string representation of a set of units

    Units can be displayed in two different formats: Fraction and Exponents. The function
    retrieves the global settings for unit styles and construct the string accordingly.

    Args:
        units (dict): A dictionary object representing a set of units

    Returns:
        The string representation of the units

    """
    unit_string = ""
    if sts.get_settings().unit_style == UnitStyle.FRACTION:
        unit_string = __construct_unit_string_as_fraction(units)
    if sts.get_settings().unit_style == UnitStyle.EXPONENTS:
        unit_string = __construct_unit_string_with_exponents(units)
    return unit_string


def operate_with_units(operator, *operands):
    """perform an operation with two sets of units"""

    result = UNIT_OPERATIONS[operator](*operands) if operator in UNIT_OPERATIONS else {}
    # filter for non-zero values
    return OrderedDict([(unit, count) for unit, count in result.items() if count != 0])


def __parse_unit_string_to_list(unit_string: str) -> List[Union[str, List]]:
    """Parse a unit string into a list of tokens

    A token can be a single unit, an operator such as "*" or "/" or "^", a number indicating
    the power of a unit, or a list of tokens grouped together. For example, kg*m/s^2A^2 would
    be parsed into: ["kg", "*", "m", "/", [["s", "^", "2"], "*", ["A", "^", "2"]]]

    """

    unit_string = unit_string.replace("⋅", "*")  # replace dots with multiplication sign

    raw_tokens_list = []  # The raw list of tokens
    tokens_list = []  # The final list of tokens

    token_pattern = re.compile(r"[a-zA-Z]+(\^-?[0-9]+)?|/|\*|\(.*?\)")
    bracket_enclosed_expression_pattern = re.compile(r"\(.*?\)")
    unit_with_exponent_pattern = re.compile(r"[a-zA-Z]+\^-?[0-9]+")
    operator_pattern = re.compile(r"[/*]")

    # Check if the input only consists of valid token strings
    if not re.fullmatch(r"({})+".format(token_pattern.pattern), unit_string):
        raise ValueError("\"{}\" is not a valid unit".format(unit_string))

    # For every token found, process it and append it to the list
    for result in token_pattern.finditer(unit_string):
        token = result.group()
        if bracket_enclosed_expression_pattern.fullmatch(token):
            # If the token is a bracket enclosed expression, recursively parse the content of
            # that bracket and append it to the tokens list as a list
            raw_tokens_list.append(__parse_unit_string_to_list(token[1:-1]))
        elif unit_with_exponent_pattern.fullmatch(token):
            # Group a unit with exponent together and append to the list as a whole
            unit_and_exponent = token.split("^")
            raw_tokens_list.append([unit_and_exponent[0], "^", unit_and_exponent[1]])
        else:
            raw_tokens_list.append(token)

    # At this stage, except for when an explicit bracket is present, no grouping of tokens
    # has occurred yet. The following code checks for expressions connected with implicit
    # multiplication, and groups them together (also adding a multiplication operator). The
    # following flag keeps track of if there is an operator present between the current token
    # and the last expression being processed, if not, assume implicit multiplication.
    preceding_operator_exists = True

    for token in raw_tokens_list:
        if preceding_operator_exists:
            tokens_list.append(token)
            preceding_operator_exists = False
        elif isinstance(token, str) and operator_pattern.fullmatch(token):
            tokens_list.append(token)
            preceding_operator_exists = True
        else:
            # When there is no preceding operator, and the current token is not an operator,
            # add multiplication sign, and group this item with the previous one.
            last_token = tokens_list.pop()
            tokens_list.append([last_token, "*", token])
            preceding_operator_exists = False

    return tokens_list


def __construct_expression_tree_with_list(tokens: List[Union[str, List]]) -> Expression:
    """Build a binary expression tree with a list of tokens

    The algorithm to construct the tree is called recursive descent, which made use of two
    stacks. The operator stack and the operand stack. For each new token, if the token is an
    operator, it is compared with the current top of the operator stack. The operator stack
    is maintained so that the top of the stack has higher priority in order of operations
    compared to the rest of the stack. If the current top has higher priority compared to the
    operator being processed, it is popped from the stack, used to build a sub-tree with the
    top two operands in the operand stack, and pushed into the operand stack.

    For details regarding this algorithm, see the reference below.

    Reference:
        Parsing Expressions by Recursive Descent - Theodore Norvell (C) 1999
        https://www.engr.mun.ca/~theo/Misc/exp_parsing.htm

    Args:
        tokens (list): The list of tokens to process.

    Returns:
        The expression tree representing the set of units. For more details regarding the
        structure of the tree, see top of this file where the Expression type is defined.

    """

    # Initialize the two stacks
    operand_stack = []  # type: List[Union[Expression, str]]
    operator_stack = ["base"]  # type: List[str]

    # Define the order of operations
    precedence = {
        "base": 0,
        "*": 1,
        "/": 1,
        "^": 2
    }

    def __construct_sub_tree_and_push_to_operand_stack():
        right = operand_stack.pop()
        left = operand_stack.pop()
        operator = operator_stack.pop()
        operand_stack.append(Expression(operator, left, right))

    # Push all tokens into the two stacks, make sub-trees if necessary
    for token in tokens:
        top_of_operators = operator_stack[-1]
        if isinstance(token, list):
            # Recursively make sub-tree with grouped expressions
            operand_stack.append(__construct_expression_tree_with_list(token))
        elif token in precedence and precedence[token] > precedence[top_of_operators]:
            operator_stack.append(token)  # Push the higher priority operator on top
        elif token in precedence and precedence[token] <= precedence[top_of_operators]:
            # If an operator with lower precedence is being processed, make a sub-tree
            # with the current top of the operator stack and push it to the operands.
            __construct_sub_tree_and_push_to_operand_stack()
            operator_stack.append(token)  # This operator becomes the new top
        else:
            operand_stack.append(token)

    # Create the final tree from all the tokens and sub-trees left in the stacks
    while len(operator_stack) > 1:
        __construct_sub_tree_and_push_to_operand_stack()

    return operand_stack[0] if operand_stack else Expression("", "", "")


def __evaluate_unit_tree(tree: Expression) -> Dict[str, int]:
    """Construct a unit dictionary object from an expression tree

    Args:
        tree (Expression): the expression tree to be evaluated

    Returns:
        All units in the tree and their powers stored in a dictionary object

    """
    units = OrderedDict()
    if isinstance(tree, Expression) and tree.operator == "^":
        # When a unit with an exponent is found, add it to the dictionary object
        units[tree.left] = int(tree.right)
    elif isinstance(tree, Expression) and tree.operator in ["*", "/"]:
        for unit, exponent in __evaluate_unit_tree(tree.left).items():
            units[unit] = exponent
        for unit, exponent in __evaluate_unit_tree(tree.right).items():
            start_exponent_from = units[unit] if unit in units else 0
            plus_or_minus = 1 if tree.operator == "*" else -1
            units[unit] = start_exponent_from + plus_or_minus * exponent
    else:  # just a string then count it
        units[tree] = 1
    return units


def __construct_unit_string_as_fraction(units: Dict[str, int]) -> str:
    """Construct a unit string in the fraction format"""

    numerator_units = ["{}{}".format(
        unit, __power_num2str(power)) for unit, power in units.items() if power > 0]
    denominator_units = ["{}{}".format(
        unit, __power_num2str(-power)) for unit, power in units.items() if power < 0]

    numerator_string = DOT_STRING.join(numerator_units) if numerator_units else "1"
    denominator_string = DOT_STRING.join(denominator_units)

    if not denominator_units:
        return numerator_string if numerator_units else ""
    if len(denominator_units) > 1:
        # For multiple units in the denominator, use brackets to avoid ambiguity
        return "{}/({})".format(numerator_string, denominator_string)

    return "{}/{}".format(numerator_string, denominator_string)


def __construct_unit_string_with_exponents(units: Dict[str, int]) -> str:
    """Construct a unit string in the exponent format"""
    unit_strings = ["{}{}".format(
        unit, __power_num2str(power)) for unit, power in units.items()]
    return DOT_STRING.join(unit_strings)


def __power_num2str(power) -> str:
    """Construct a string for the power of a unit"""

    fraction = Fraction(power).limit_denominator(10)
    if fraction.numerator == 1 and fraction.denominator == 1:
        return ""  # do not print power of 1 as it's implied
    if fraction.denominator == 1:
        return "^{}".format(str(fraction.numerator))
    return "^({})".format(str(fraction))


def __neg(units):
    return deepcopy(units)


def __add_and_sub(units_var1, units_var2):
    if units_var1 and units_var2 and units_var1 != units_var2:
        warnings.warn("You're trying to add/subtract two values with mismatching units.")
        return OrderedDict()
    if not units_var1:  # If any of the two units are empty, use the other one
        return deepcopy(units_var2)
    return deepcopy(units_var1)


def __mul(units_var1, units_var2):
    units = OrderedDict()
    for unit, exponent in units_var1.items():
        __update_unit_exponent_count_in_dict(units, unit, exponent)
    for unit, exponent in units_var2.items():
        __update_unit_exponent_count_in_dict(units, unit, exponent)
    return units


def __div(units_var1, units_var2):
    units = OrderedDict()
    for unit, exponent in units_var1.items():
        __update_unit_exponent_count_in_dict(units, unit, exponent)
    for unit, exponent in units_var2.items():
        __update_unit_exponent_count_in_dict(units, unit, -exponent)
    return units


def __sqrt(units):
    new_units = OrderedDict()
    for unit, exponent in units.items():
        new_units[unit] = exponent / 2
    return new_units


def __update_unit_exponent_count_in_dict(unit_dict, unit_string, change):
    current_count = 0 if unit_string not in unit_dict else unit_dict[unit_string]
    unit_dict[unit_string] = current_count + change


UNIT_OPERATIONS = {
    lit.NEG: __neg,
    lit.ADD: __add_and_sub,
    lit.SUB: __add_and_sub,
    lit.MUL: __mul,
    lit.DIV: __div,
    lit.SQRT: __sqrt
}
