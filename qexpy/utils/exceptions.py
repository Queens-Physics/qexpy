"""This file contains definitions of some internal exceptions for QExPy"""


class QExPyBaseError(Exception):
    """This is the base error type for QExPy"""


class InvalidArgumentTypeError(QExPyBaseError):
    """Exception thrown for invalid function arguments"""

    def __init__(self, objective, got, expected=""):
        message = "Invalid argument type for {}: \"{}\"".format(objective, type(got))
        if expected:
            message += ", expected: {}".format(expected)
        super().__init__(message)


class UndefinedOperationError(QExPyBaseError):
    """Exception thrown for undefined operations"""

    def __init__(self, operation, got, expected=""):
        if isinstance(got, list):
            got = " and ".join("\"{}\"".format(type(x)) for x in got)
        message = "Operation \"{}\" is undefined with operands of type(s) {}".format(operation, got)
        if expected:
            message += ", expected: {}".format(expected)
        super().__init__(message)


class IllegalArgumentError(QExPyBaseError):
    """Exception for general invalid arguments"""

    def __init__(self, message):
        super().__init__(message)
