"""This module contains the data structure and operations for experimental values"""

from .data import MeasuredValue as Measurement
from .datasets import ExperimentalValueArray as MeasurementArray
from .datasets import XYDataSet
from .data import set_covariance, get_covariance, set_correlation, get_correlation
from .operations import sqrt, exp, sin, cos, tan, asin, acos, atan, csc, sec, cot, log, log10, sind, cosd, tand, pi, e
