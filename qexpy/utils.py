import numpy as np
import re

# Global variable to keep track of whether the output_notebook command was run
bokeh_output_notebook_called = False
mpl_output_notebook_called = False


def in_notebook():
    """Simple function to check if module is loaded in a notebook"""
    return hasattr(__builtins__, '__IPYTHON__')


def mpl_output_notebook():
    from IPython import get_ipython
    ipython = get_ipython()
    # ipython.magic('matplotlib inline')
    global mpl_output_notebook_called
    mpl_output_notebook_called = True


def load_data_from_file(path, delimiter=','):
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


# These are used for checking whether something is an instance of an array or a number.
number_types = (
    int, float, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
    np.float32, np.float64)
int_types = (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
array_types = (tuple, list, np.ndarray)
