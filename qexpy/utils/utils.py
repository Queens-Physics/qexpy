import numpy as np
import re

# Global variable to keep track of whether the output_notebook command was run
_mpl_output_notebook_called = False


def count_significant_figures(value) -> int:
    """Counts the number of significant figures for a number

    The input can be either a number or the string representation of a number

    Args:
        value: the number to be counted

    """
    try:
        float(value)
        # first remove the decimal point
        str_repr_of_value = str(value).replace(".", "")
        # then strip the leading 0s
        str_repr_of_value = re.sub(r"^0*", "", str_repr_of_value)
        return len(str_repr_of_value)
    except (ValueError, TypeError):
        print("Error: invalid input! You can only count the significant figures of a number "
              "or the string representation of a number.")


def load_data_from_file(path, delimiter=',') -> np.ndarray:
    """Reads data from a file

    Retrieves data from a file, separated with the given delimiter.

    Args:
        path: the file path
        delimiter: the delimiter with which the lines are split

    Returns:
        A 2-dimensional numpy array where each column in the file
        becomes an array of numbers
    """
    import csv
    with open(path, newline='') as openfile:
        reader = csv.reader(openfile, delimiter=delimiter)
        number_pattern = r'\-?[0-9]+(\.[0-9]+)?'
        # find all rows with only valid entries (entries that matches the number format)
        valid_rows = filter(lambda row: all(re.match(number_pattern, entry) for entry in row), reader)
        # for each row convert the entries to numbers
        data = map(lambda row: map(lambda entry: float(entry), row), valid_rows)
        # TODO: add code to check if all rows are of the same length (or else the transpose won't work properly)
        # transpose to arrays that correspond to the columns
        ret = np.transpose(np.array(data, dtype=float))
    return ret


def _in_notebook() -> bool:
    """Simple function to check if module is loaded in a notebook"""
    return hasattr(__builtins__, '__IPYTHON__')


def _mpl_output_notebook():
    from IPython import get_ipython
    ipython = get_ipython()
    # ipython.magic('matplotlib inline')
    global _mpl_output_notebook_called
    _mpl_output_notebook_called = True
