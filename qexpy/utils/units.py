"""Internal module used to parse units"""

import re
import collections
import warnings
from typing import Dict, List, Union
import qexpy.settings.settings as settings
import qexpy.settings.literals as lit

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
    ["kg", "*", "m", "/", [["s", "^", "2"], "*", ["A", "^", "2"]]]

    This list is later on used to construct the abstract syntax tree

    """

    raw_tokens_list = []  # the raw list of tokens
    tokens_list = []  # the final list of tokens

    # The following regex patterns matches the individual tokens in a unit string
    # "[a-zA-Z]+(\^[0-9]+)*" matches any individual unit strings or ones with exponents
    # "\).*?\)" matches any bracket enclosed expressions
    # "/" and "\*" are the division and multiplication signs
    token_pattern = re.compile(r"[a-zA-Z]+(\^-?[0-9]+)?|/|\*|\(.*?\)")
    bracket_pattern = re.compile(r"\(.*?\)")
    unit_with_exponent_pattern = re.compile(r"[a-zA-Z]+\^-?[0-9]+")
    operator_pattern = re.compile(r"[/*]")

    # check if the input is valid using regex
    if not re.fullmatch(r"({})+".format(token_pattern.pattern), unit_string):
        raise ValueError("The given unit string \"{}\" is invalid!".format(unit_string))

    # for every token found, process it and append it to the list
    for result in token_pattern.finditer(unit_string):
        token = result.group()
        if bracket_pattern.fullmatch(token):
            # if the token is a bracket enclosed expression, recursively parse the content of
            # that bracket and append it to the tokens list as a list
            raw_tokens_list.append(__parse_unit_string_to_list(token[1:len(token) - 1]))
        elif unit_with_exponent_pattern.fullmatch(token):
            # append a unit with exponent pattern as a list because it should be seen as a whole
            exponent_sign_index = token.find("^")
            raw_tokens_list.append([token[:exponent_sign_index], "^", token[exponent_sign_index + 1:]])
        else:
            # during this stage, all tokens are appended in order, no grouping has happened yet
            # unless there is an explicit bracket, for expressions such as "s^2A^2", implicit
            # multiplications signs still need to be added, and this expression should also be
            # seen as enclosed by brackets so that it's considered as one
            raw_tokens_list.append(token)

    # the following flag indicates if there is a preceding operator in the list of tokens, if
    # not, an implicit multiplication sign needs to be added, and this token should be grouped
    # with the previous one. This value starts as True because the first item in the list does
    # not need any preceding operators
    preceding_operator_exists = True

    # process the raw token list, add implicit brackets or multiplication signs if needed
    for token in raw_tokens_list:
        if preceding_operator_exists:
            tokens_list.append(token)
            preceding_operator_exists = False
        elif isinstance(token, str) and operator_pattern.fullmatch(token):
            # if an actual operator is found
            tokens_list.append(token)
            preceding_operator_exists = True
        else:
            last_token = tokens_list.pop()
            # add implicit multiplication sign and group this item with the previous one
            tokens_list.append([last_token, "*", token])
            preceding_operator_exists = False

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
    elif isinstance(tree, Expression) and tree.operator in ["*", "/"]:
        for unit, exponent in __evaluate_unit_tree(tree.left).items():
            units[unit] = exponent
        for unit, exponent in __evaluate_unit_tree(tree.right).items():
            start_exponent_from = units[unit] if unit in units else 0
            plus_or_minus = 1 if tree.operator == "*" else -1
            units[unit] = start_exponent_from + plus_or_minus * exponent
    elif isinstance(tree, str):
        units[tree] = 1
    return units


def operate_with_units(operator, *operands):
    # TODO: implement unit propagation for non-linear operations
    return UNIT_OPERATIONS[operator](*operands) if operator in UNIT_OPERATIONS else {}


def _add_and_sub(units_var1, units_var2):
    from copy import deepcopy
    if units_var1 and units_var2 and units_var1 != units_var2:
        warnings.warn("You're trying to add/subtract two values with mismatching units, returning empty unit")
        return {}
    elif not units_var1:
        return deepcopy(units_var2)
    else:
        return deepcopy(units_var1)


def _mul(units_var1, units_var2):
    units = {}
    for unit, exponent in units_var1.items():
        __update_unit_exponent_count_in_dict(units, unit, exponent)
    for unit, exponent in units_var2.items():
        __update_unit_exponent_count_in_dict(units, unit, exponent)
    return units


def _div(units_var1, units_var2):
    units = {}
    for unit, exponent in units_var1.items():
        __update_unit_exponent_count_in_dict(units, unit, exponent)
    for unit, exponent in units_var2.items():
        __update_unit_exponent_count_in_dict(units, unit, -exponent)
    return units


def __update_unit_exponent_count_in_dict(unit_dict, unit_string, count):
    unit_dict[unit_string] = (0 if unit_string not in unit_dict else unit_dict[unit_string]) + count


UNIT_OPERATIONS = {
    lit.ADD: _add_and_sub,
    lit.SUB: _add_and_sub,
    lit.MUL: _mul,
    lit.DIV: _div
}
