"""Definitions for internal exceptions in QExPy"""


class QExPyBaseError(Exception):
    """The base error type for QExPy"""


class IllegalArgumentError(QExPyBaseError):
    """Exception for invalid arguments"""


class UndefinedActionError(QExPyBaseError):
    """Exception for undefined system states or function calls"""


class UndefinedOperationError(UndefinedActionError):
    """Exception for undefined arithmetic operations between values"""

    def __init__(self, operation, got, expected=""):
        """Defines the standard format for the error message"""

        if isinstance(got, list):
            got_type = " and ".join("\"{}\"".format(type(x)) for x in got)
        else:
            got_type = type(got)
        message = "Operation \"{}\" is undefined with operands of type(s) {}."
        if expected:
            message += " Expected: {}".format(expected)
        super().__init__(message.format(operation, got_type))
