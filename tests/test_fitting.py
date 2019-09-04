"""Unit tests for the fitting module

This file contains tests for fitting a function to a data set.

"""

import os
import pytest
import qexpy as q


class TestFitting:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        q.get_settings().reset()

    def test_poly_fit(self):
        """unit tests for polynomial fit functions"""

        xydata = q.XYDataSet(xdata=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             ydata=[0.6, 1.6, 3.5, 4.1, 4.6, 5.6, 6.1, 7.9, 8.7, 9.8],
                             yerr=0.5,
                             xname='length', xunit='m',
                             yname='force', yunit='N')
        result = xydata.fit("linear")
        assert str(result[0]) == "slope = 0.98 +/- 0.04"
        assert str(result[1]) == "intercept = -0.1 +/- 0.3"
        assert result.chi_squared == pytest.approx(4.75151515151515, 1e-10)
        assert result.ndof == 7
        print(result)

        result = q.fit(xdata=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], xerr=0.5,
                       ydata=[3.86, 8.80, 16.11, 24.6, 35.71, 48.75, 64,
                              81.15, 99.72, 120.94], yerr=0.5, model=q.FitModel.QUADRATIC)
        assert str(result[0]) == "a = 1.004 +/- 0.009"
        assert str(result[1]) == "b = 2.0 +/- 0.1"
        assert str(result[2]) == "c = 0.9 +/- 0.2"
        assert result.chi_squared == pytest.approx(1.13355393939395, 1e-10)
        assert result.ndof == 6

        xdata = q.MeasurementArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], error=0.5)
        ydata = q.MeasurementArray([3.89, 18.01, 58.02, 135.92, 264.01, 453.99, 718.02,
                                    1067.98, 1516.01, 2074], error=0.5)
        result = q.fit(xdata, ydata, model=q.FitModel.POLYNOMIAL, degree=3)
        assert result[0].value == pytest.approx(2, 1e-1)
        assert result[1].value == pytest.approx(0.9882, 1e-1)
        assert result[2].value == pytest.approx(-2.928, 1e-1)
        assert result[3].value == pytest.approx(3.86, 1e-1)

    def test_gaussian_fit(self):
        """unit test for gaussian fit function"""
        curr_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(curr_path, "./resources/data_for_test_gaussian_fit.csv")
        data = q.load_data_from_file(filename)
        xydata = q.XYDataSet(xdata=data[0], ydata=data[1], yerr=0.5)

        result = xydata.fit("gaussian", parguess=[55, 20, 4])

        assert result[0].value == pytest.approx(50, 1e-1)
        assert result[1].value == pytest.approx(25, 1e-1)
        assert result[2].value == pytest.approx(3, 1e-1)

    def test_custom_fit(self):
        """unit test for custom fit functions"""

        curr_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(curr_path, "./resources/data_for_test_custom_fit.csv")
        data = q.load_data_from_file(filename)
        xydata = q.XYDataSet(xdata=data[0], ydata=data[1], yerr=1)

        def model(x, a, b):
            return a * q.sin(b * x)

        result = xydata.fit(model, parguess=[1, 1], parnames=["a", "b"], parunits=["m", "kg"])
        assert str(result[0]) == "a = 5.1 +/- 0.2 [m]"
        assert str(result[1]) == "b = 0.501 +/- 0.009 [kg]"
        assert result.chi_squared == pytest.approx(5.533370644202178, 1e-10)
        assert result.ndof == 17