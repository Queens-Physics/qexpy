"""This file contains definitions of some internal exceptions for QExPy"""


class QExPyBaseError(Exception):
    pass


class InvalidArgumentTypeError(QExPyBaseError):

    def __init__(self, objective, got, expected=""):
        message = "Invalid argument type for {}: \"{}\"".format(objective, type(got))
        if expected:
            message += ", expected: {}".format(expected)
        super().__init__(message)


class UndefinedOperationError(QExPyBaseError):

    def __init__(self, operation, got, expected=""):
        if isinstance(got, list):
            got = " and ".join("\"{}\"".format(type(x)) for x in got)
        message = "Operation \"{}\" is undefined with operands of type(s) {}".format(operation, got)
        if expected:
            message += ", expected: {}".format(expected)
        super().__init__(message)
