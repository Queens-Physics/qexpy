"""Internal module used to parse units"""

import re

DOT_STRING = "⋅"
OPERATOR = "operator"
LEFT = "left"
RIGHT = "right"


def construct_unit_string(units) -> str:
    """Constructs a string from a dictionary object

    Args:
        units (dict): the unit dictionary object

    Returns:
        the string representation of the units

    """
    from qexpy.settings.settings import UnitStyle, get_unit_style
    if get_unit_style() == UnitStyle.EXPONENTS:
        unit_strings = []
        for unit, power in units.items():
            if power == 1:
                unit_strings.append("{}".format(unit))
            elif power != 0:
                unit_strings.append("{}^{}".format(unit, power))
        return DOT_STRING.join(unit_strings)
    else:
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
        if len(numerator_units) == 0:
            numerator_string = "1"
        if len(denominator_units) == 0:
            return numerator_string
        elif len(denominator_units) > 1:
            return "{}/({})".format(numerator_string, denominator_string)
        else:
            return "{}/{}".format(numerator_string, denominator_string)


def parse_units(unit_string) -> dict:
    """Parses a unit string into a dictionary object

    The keys of the return object will be the separate unit strings, and the values
    for each entry is the power on that unit. This method is implemented with an
    abstract syntax tree (AST).

    Args:
        unit_string (str): the unit string to be parsed

    Returns:
        the dictionary object that stores information about the units

    TODO:
        add support for non-integer exponents on units? maybe?

    """
    tokens = __parse_unit_string_to_list(unit_string)
    ast = __parse_unit_list_into_tree(tokens)
    return __evaluate_unit_tree(ast)


def __parse_unit_string_to_list(unit_string) -> list:
    """Transforms a raw unit string into a token list

    For example, kg*m/(s^2A^2) would be converted into the following list of tokens:
    ["kg", "*", "m", "/", ["s", "^", "2", "*", "A", "^", "2"]]

    This list is later on used to construct the abstract syntax tree

    """
    tokens = []
    temp_string = ""
    bracket_on = False
    for char in unit_string:
        top = tokens[len(tokens) - 1] if len(tokens) > 0 else ""
        if char == "(":
            bracket_on = True
        elif char == ")":
            bracket_on = False
            tokens.append(__parse_unit_string_to_list(temp_string))
            temp_string = ""
        elif bracket_on:
            temp_string += char
        elif len(tokens) == 0 and re.match(r"[a-zA-Z]", char):
            # validate the first entry and push it into the stack
            tokens.append(char)
        elif len(tokens) == 0 and not re.match(r"[a-zA-Z]", char):
            raise ValueError("The unit string must start with a unit")
        elif isinstance(top, list):
            # do not try to combine a bracket expression with anything
            tokens.append(char)
        elif re.fullmatch(r"[a-zA-Z]+", top) and re.match(r"[a-zA-Z]", char):
            tokens.pop()
            tokens.append(top + char)  # combining letters of the same unit
        elif re.fullmatch(r"-?[1-9]*", top) and re.match(r"[1-9]", char):
            tokens.pop()
            tokens.append(top + char)  # combining the negative sign and the exponent number
        elif re.fullmatch(r"-?[1-9]*", top) and re.match(r"[a-zA-Z]", char):
            tokens.append("*")
            tokens.append(char)  # append implicit multiplication signs
        else:
            tokens.append(char)
    return tokens


def __parse_unit_list_into_tree(tokens) -> dict:
    """Construct an abstract syntax tree from a list of tokens

    The algorithm is modified from the following paper:
    Parsing Expressions by Recursive Descent - Theodore Norvell (C) 1999
    Link: https://www.engr.mun.ca/~theo/Misc/exp_parsing.htm

    """
    operands = []
    operators = ["base"]
    precedence = {
        "base": 0,
        "*": 1,
        "/": 1,
        "^": 2
    }

    def __construct_sub_tree():
        right = operands.pop()
        left = operands.pop()
        operator = operators.pop()
        expression = {
            "operator": operator,
            "left": left,
            "right": right
        }
        operands.append(expression)

    # push all tokens into the two stacks, make sub-trees if necessary
    for token in tokens:
        top_of_operators = operators[len(operators) - 1]
        if isinstance(token, list):
            operands.append(__parse_unit_list_into_tree(token))
        elif token in precedence and precedence[token] > precedence[top_of_operators]:
            operators.append(token)
        elif token in precedence and precedence[token] <= precedence[top_of_operators]:
            __construct_sub_tree()
            operators.append(token)
        else:
            operands.append(token)

    # create the final tree from all the tokens and sub-trees
    while len(operators) > 1:
        __construct_sub_tree()

    if len(operands) > 1:
        raise RuntimeError("Fail to construct AST. Please report the bug to the package github page")
    return operands[0]


def __evaluate_unit_tree(tree) -> dict:
    """Construct unit dictionary object from an AST"""
    units = {}
    if isinstance(tree, dict) and tree[OPERATOR] == "^":
        units[tree[LEFT]] = int(tree[RIGHT])
    elif isinstance(tree, dict) and (tree[OPERATOR] == "/" or tree[OPERATOR] == "*"):
        for unit, exponent in __evaluate_unit_tree(tree[LEFT]).items():
            units[unit] = exponent
        for unit, exponent in __evaluate_unit_tree(tree[RIGHT]).items():
            start_exponent_from = units[unit] if unit in units else 0
            plus_or_minus = 1 if tree[OPERATOR] == "*" else -1
            units[unit] = start_exponent_from + plus_or_minus * exponent
    elif not isinstance(tree, dict):
        units[tree] = 1
    return units
