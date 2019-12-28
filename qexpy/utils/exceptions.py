"""Definitions for internal exceptions in QExPy"""


class QExPyBaseError(Exception):
    """The base error type for QExPy"""


class IllegalArgumentError(QExPyBaseError):
    """Exception for invalid arguments"""


class UndefinedActionError(QExPyBaseError):
    """Exception for undefined system states or function calls"""


class UndefinedOperationError(UndefinedActionError):
    """Exception for undefined arithmetic operations between values"""

    def __init__(self, op, got, expected=""):
        """Defines the standard format for the error message"""

        if not isinstance(got, (list, tuple)):
            got = (got,)

        got_types = " and ".join("\'{}\'".format(type(x).__name__) for x in got)
        message = "\"{}\" is undefined with operands of type(s) {}.".format(op, got_types)
        if expected:
            message += " Expected: {}".format(expected)

        super().__init__(message)
