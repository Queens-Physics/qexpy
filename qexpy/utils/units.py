"""Internal module used to parse units"""

import re
import collections
from typing import Dict, List, Union
import qexpy.settings.settings as settings

DOT_STRING = "⋅"

# structure for an expression tree used to parse units
Expression = collections.namedtuple("Expression", "operator, left, right")


def construct_unit_string(units: Dict[str, int]) -> str:
    """Constructs a string from a dictionary object of units

    Args:
        units (dict): the unit dictionary object where the powers are mapped to their
            corresponding unit strings

    Returns:
        the string representation of the units

    """

    unit_string = ""
    if settings.get_unit_style() == settings.UnitStyle.EXPONENTS:
        unit_string = __construct_unit_string_as_fraction(units)
    elif settings.get_unit_style() == settings.UnitStyle.FRACTION:
        unit_string = __construct_unit_string_with_exponents(units)
    return unit_string


def __construct_unit_string_as_fraction(units: Dict[str, int]) -> str:
    """Prints out units like kg⋅m^2⋅s^-2"""
    unit_strings = []
    for unit, power in units.items():
        if power == 1:
            # for power of 1 or -1, do not print the exponent as it's implied
            unit_strings.append("{}".format(unit))
        elif power != 0:
            unit_strings.append("{}^{}".format(unit, power))
    return DOT_STRING.join(unit_strings)


def __construct_unit_string_with_exponents(units: Dict[str, int]) -> str:
    """Prints out units like kg⋅m^2/s^3"""
    numerator_units = []
    denominator_units = []
    for unit, power in units.items():
        if power == 1:
            numerator_units.append("{}".format(unit))
        elif power == -1:
            denominator_units.append("{}".format(unit))
        elif power < 0:
            denominator_units.append("{}^{}".format(unit, -power))
        elif power > 0:
            numerator_units.append("{}^{}".format(unit, power))
    numerator_string = DOT_STRING.join(numerator_units)
    denominator_string = DOT_STRING.join(denominator_units)
    if not numerator_units:
        # if fraction style is enforced, print as 1/... if all exponents are negative
        numerator_string = "1"
    if not denominator_units:
        return numerator_string
    if len(denominator_units) > 1:
        # for multiple units in the denominator, use brackets to avoid ambiguity
        return "{}/({})".format(numerator_string, denominator_string)
    return "{}/{}".format(numerator_string, denominator_string)


def parse_units(unit_string: str) -> Dict[str, int]:
    """Parses a unit string into a dictionary object

    The keys of the return object will be the separate unit strings, and the values
    for each entry is the exponent on that unit. This method is implemented with an
    abstract syntax tree (AST).

    Args:
        unit_string (str): the unit string to be parsed

    Returns:
        the dictionary object that stores information about the units

    TODO:
        add support for non-integer exponents on units? maybe?

    """
    tokens = __parse_unit_string_to_list(unit_string)
    ast = __construct_expression_tree_with_list(tokens)
    return __evaluate_unit_tree(ast)


def __parse_unit_string_to_list(unit_string: str) -> List[Union[str, List]]:
    """Transforms a raw unit string into a list of tokens

    For example, kg*m/s^2A^2 would be converted into the following list of tokens:
    ["kg", "*", "m", "/", ["s", "^", "2", "*", "A", "^", "2"]]

    This list is later on used to construct the abstract syntax tree

    """
    tokens_list = []  # the final list of tokens
    bracket_buffer = ""  # buffer for processing brackets
    bracket_on = False  # flag indicating tokens inside of a bracket is being processed

    def __process_char_for_list(char: str, tokens: list):
        nonlocal bracket_on, bracket_buffer
        # access the last character in the existing token list
        top = tokens[len(tokens) - 1] if tokens else ""
        if char == "(":
            # set flag to true, all of the following characters will be stored in temp_string
            bracket_on = True
        elif char == ")":
            bracket_on = False
            # once the end bracket is found, process the temp_string buffer into a separate list
            # and then append the list as one single token to the list of tokens, as everything
            # inside a bracket should be considered as one
            tokens.append(__parse_unit_string_to_list(bracket_buffer))
            tokens.append("*")  # append implicit multiplication sign
            # empty out the buffer
            bracket_buffer = ""
        elif bracket_on:
            # load character into buffer if the current character is inside of a bracket
            bracket_buffer += char
        elif not tokens and re.match(r"[a-zA-Z]", char):
            # validate the first entry and push it into the stack
            tokens.append(char)
        elif not tokens and not re.match(r"[a-zA-Z]", char):
            raise ValueError("The unit string must start with a unit")
        elif re.match(r"[*/]", char):
            if not isinstance(top, list) and top == "*":
                tokens.pop()
            tokens.append(char)
        elif isinstance(top, list):
            # if the last element of the tokens array is a list
            __process_char_for_list(char, top)
        elif re.fullmatch(r"[a-zA-Z]+", top) and re.match(r"[a-zA-Z]", char):
            # if both the last character and this character are alphabetical, combine them into
            # one single unit string
            tokens.pop()
            tokens.append(top + char)
        elif re.fullmatch(r"-?[1-9]*", top) and re.match(r"[1-9]", char):
            # combine characters that belong to the same exponent expression
            tokens.pop()
            tokens.append(top + char)
        elif re.fullmatch(r"-?[1-9]*", top) and re.match(r"[a-zA-Z]", char):
            # for units entered in formats such as s^2A^2, add implicit multiplication sign and
            # consider them together. It will practically be processed like (s^2*A^2)
            exponent = tokens.pop()
            exp_sign = tokens.pop()
            unit = tokens.pop()
            tokens.append([unit, exp_sign, exponent, "*", char])
        else:
            tokens.append(char)

    for character in unit_string:
        # read character by character
        __process_char_for_list(character, tokens_list)
    # delete trailing operators
    if tokens_list[len(tokens_list) - 1] == "*":
        tokens_list.pop()
    return tokens_list


def __construct_expression_tree_with_list(tokens: List[Union[str, List]]) -> Expression:
    """Construct an abstract syntax tree from a list of tokens

    This method will return a named tuple object that represents an expression tree.
    The "operator" in the named tuple object is the node, and the "left" and "right"
    operands are the two branches stemming from the node. The operands can be a simple
    unit string, which represents a leaf, or a dictionary object, which would be the
    root of a sub-tree. This tree can be traversed using in-order to recreate the unit
    expression.

    The algorithm to construct the tree is called recursive descent, which made use of
    two stacks. The operator stack and the operand stack. For each new token, if the
    token is an operator, it is compared with the current top of the operator stack.
    The operator stack is maintained so that the top of the stack always has higher
    priority compared to the rest of the stack. If the current top has higher priority
    compared to the operator being processed, it is popped from the stack, used to build
    a sub-tree with the top two operands in the operand stack, and re-pushed into the
    operand stack.

    For details regarding this algorithm, see the reference below.

    Reference:
        Parsing Expressions by Recursive Descent - Theodore Norvell (C) 1999
        https://www.engr.mun.ca/~theo/Misc/exp_parsing.htm

    """
    operand_stack = []  # type: List[Union[Expression, str]]
    operator_stack = ["base"]  # type: List[str]
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
        expression = Expression(operator, left, right)
        operand_stack.append(expression)

    # push all tokens into the two stacks, make sub-trees if necessary
    for token in tokens:
        top_of_operators = operator_stack[len(operator_stack) - 1]
        if isinstance(token, list):
            operand_stack.append(__construct_expression_tree_with_list(token))
        elif token in precedence and precedence[token] > precedence[top_of_operators]:
            operator_stack.append(token)
        elif token in precedence and precedence[token] <= precedence[top_of_operators]:
            __construct_sub_tree_and_push_to_operand_stack()
            operator_stack.append(token)
        else:
            operand_stack.append(token)

    # create the final tree from all the tokens and sub-trees
    while len(operator_stack) > 1:
        __construct_sub_tree_and_push_to_operand_stack()

    if len(operand_stack) > 1:
        raise RuntimeError("Fail to construct AST. Please report the bug to the package github page")
    return operand_stack[0] if operand_stack else Expression("", "", "")


def __evaluate_unit_tree(tree: Union[Expression, str]) -> dict:
    """Construct unit dictionary object from an expression tree"""
    units = {}
    if isinstance(tree, Expression) and tree.operator == "^":
        # when a unit with an exponent is found, add it to the dictionary object
        units[tree.left] = int(tree.right)
    elif isinstance(tree, Expression) and (tree.operator == "/" or tree.operator == "*"):
        for unit, exponent in __evaluate_unit_tree(tree.left).items():
            units[unit] = exponent
        for unit, exponent in __evaluate_unit_tree(tree.right).items():
            start_exponent_from = units[unit] if unit in units else 0
            plus_or_minus = 1 if tree.operator == "*" else -1
            units[unit] = start_exponent_from + plus_or_minus * exponent
    elif not isinstance(tree, Expression):
        units[tree] = 1
    return units
