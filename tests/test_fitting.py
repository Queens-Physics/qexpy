"""Tests for the fitting sub-package"""

import pytest
import qexpy as q
import numpy as np

from qexpy.data.datasets import XYDataSet, ExperimentalValueArray
from qexpy.utils.exceptions import IllegalArgumentError


class TestFitting:
    """tests for fitting functions to datasets"""

    def test_fit_result(self):
        """tests for the fit result object"""

        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [5.14440433, 7.14299315, 9.19825169, 11.04786137, 12.98168509,
             15.33559568, 16.92760861, 18.80124373, 21.34893411, 23.16547138]

        result = q.fit(a, b, model=q.FitModel.LINEAR)

        slope, intercept = result.params[0], result.params[1]
        assert slope.value == pytest.approx(2, abs=slope.error)
        assert intercept.value == pytest.approx(3, abs=intercept.error)
        assert result[0].value == pytest.approx(2, abs=result[0].error)
        assert result[1].value == pytest.approx(3, abs=result[1].error)

        assert slope.name == "slope"
        assert intercept.name == "intercept"

        assert callable(result.fit_function)
        test = result.fit_function(3)
        assert test.value == pytest.approx(9, abs=0.2)

        residuals = result.residuals
        assert all(residuals < 0.3)

        assert result.ndof == 7
        assert result.chi_squared == pytest.approx(0)

        assert isinstance(result.dataset, XYDataSet)
        assert all(result.dataset.xdata == a)
        assert all(result.dataset.ydata == b)

        assert str(result)

        b[-1] = 50

        result = q.fit(a, b, model="linear", xrange=(0, 10))
        assert result.xrange == (0, 10)
        assert result[0].value == pytest.approx(2, abs=0.15)
        assert result[1].value == pytest.approx(3, abs=0.15)

    def test_polynomial_fit(self):
        """tests for fitting to a polynomial"""

        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [5.82616885, 10.73323542, 18.53401063, 27.16662982, 37.99711327,
             51.41386193, 66.09297228, 83.46407479, 102.23573159, 122.8573845]

        result = q.fit(a, b, model=q.FitModel.QUADRATIC)

        assert len(result.params) == 3
        assert result[0].value == pytest.approx(1, rel=0.15)
        assert result[1].value == pytest.approx(2, rel=0.15)
        assert result[2].value == pytest.approx(3, rel=0.15)

        a = q.MeasurementArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        b = q.MeasurementArray(
            [9.96073312, 31.18583676, 78.11727423, 161.58352404, 298.7038423, 494.25761959,
             766.3146814, 1123.59437138, 1578.30697946, 2142.70591363])

        result = q.fit(a, b, model=q.FitModel.POLYNOMIAL)

        assert len(result.params) == 4
        assert result[0].value == pytest.approx(2, rel=0.3)
        assert result[1].value == pytest.approx(1, rel=0.3)
        assert result[2].value == pytest.approx(4, rel=0.3)
        assert result[3].value == pytest.approx(3, rel=0.4)

        b = [20.32132071, 64.27190108, 189.14762997, 469.97259457, 999.96248493,
             1899.41641639, 3312.43244643, 5411.38221041, 8379.45187783, 12439.47094005]

        dataset = q.XYDataSet(a, b, yerr=0.2)
        result = q.fit(dataset, model=q.FitModel.POLYNOMIAL, degrees=4)

        assert len(result.params) == 5
        assert result[0].value == pytest.approx(1, rel=0.3)
        assert result[1].value == pytest.approx(2, rel=0.3)
        assert result[2].value == pytest.approx(4, rel=0.3)
        assert result[3].value == pytest.approx(3, rel=0.4)
        assert result[4].value == pytest.approx(10, rel=0.4)

        with pytest.raises(IllegalArgumentError):
            q.fit(1, 2)

    def test_gaussian_fit(self):
        """tests for fitting to a gaussian distribution"""

        data = np.random.normal(20, 10, 10000)
        n, bins = np.histogram(data)
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        result = q.fit(centers, n, model="gaussian", parguess=[10000, 18, 9])
        assert result[1].value == pytest.approx(20, rel=0.3)
        assert result[2].value == pytest.approx(10, rel=0.3)

    def test_exponential_fit(self):
        """tests for fitting to an exponential function"""

        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        b = [3.06297029e+00, 1.84077449e+00, 1.12743994e+00, 6.70756404e-01, 4.13320658e-01,
             2.46274429e-01, 1.48775210e-01, 9.22527208e-02, 5.51925037e-02, 3.39932113e-02,
             2.01913759e-02, 1.24795552e-02, 7.67283714e-03, 4.55000537e-03, 2.75573044e-03,
             1.72345608e-03, 1.00990816e-03, 6.21266331e-04, 3.75164648e-04, 2.26534182e-04]

        result = q.fit(a, b, model=q.FitModel.EXPONENTIAL, parguess=[4.5, 0.45])
        assert result[0].value == pytest.approx(5, rel=0.1)
        assert result[1].value == pytest.approx(0.5, rel=0.1)

        assert result[0].name == "amplitude"
        assert result[1].name == "decay constant"

    def test_custom_fit(self):
        """tests for fitting to a custom function"""

        arr1 = [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.,
                5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.]
        arr2 = [0.84323953, 1.62678942, 2.46301834, 2.96315268, 3.64614702,
                4.11468579, 4.61486981, 4.76099487, 5.04725257, 4.9774383,
                4.85234697, 4.55749775, 4.13774772, 3.64002414, 3.01167174,
                2.26087356, 1.48618986, 0.71204259, -0.0831691, -0.9100453]

        def model(x, a, b):
            return a * q.sin(b * x)

        result = q.fit(arr1, arr2, model=model, parguess=[4, 0.5], parunits=["m", "kg"])
        assert result[0].name == "a"
        assert result[1].name == "b"
        assert result[0].unit == "m"
        assert result[1].unit == "kg"
        assert result[0].value == pytest.approx(5, rel=0.5)
        assert result[1].value == pytest.approx(0.5, rel=0.5)

        with pytest.raises(ValueError):
            q.fit(arr1, arr2, model="model", parguess=[4, 0.5])

        with pytest.warns(UserWarning):
            q.fit(arr1, arr2, model=model)

        with pytest.raises(TypeError):
            q.fit(arr1, arr2, model=model, parguess=['a', 0.5])

        with pytest.raises(TypeError):
            q.fit(arr1, arr2, model=model, parguess=[4, 0.5], parnames=[1, 2])

        with pytest.raises(TypeError):
            q.fit(arr1, arr2, model=model, parguess=[4, 0.5], parunits=[1, 2])

        with pytest.raises(IllegalArgumentError):
            q.fit(arr1, arr2, model=model, parguess=4)

        with pytest.raises(ValueError):
            q.fit(arr1, arr2, model=model, parguess=[4, 0.5], parunits=["m"])

        def func(x, **kwargs):
            return kwargs.get("a") * q.sin(kwargs.get("b") * x)  # pragma: no cover

        with pytest.raises(ValueError):
            q.fit(arr1, arr2, model=func, parguess=[4, 0.5])

        def func2(x):
            return x  # pragma: no cover

        with pytest.raises(ValueError):
            q.fit(arr1, arr2, model=func2, parguess=[4, 0.5])

        def func3(x, *args):
            return args[0] * q.sin(args[1] * x)

        dataset = q.XYDataSet(arr1, arr2, xerr=0.05, yerr=0.05)
        result = dataset.fit(model=func3, parguess=[4, 0.5], parnames=["arr1", "arr2"])
        assert result[0].name == "arr1"
        assert result[1].name == "arr2"

        with pytest.raises(ValueError):
            dataset.fit(model=func3, parguess=[4, 0.5], parnames=["arr1"])

        def func4(x, a, *args):
            return a + args[0] * q.sin(args[1] * x)  # pragma: no cover

        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                q.fit(arr1, arr2, model=func4, parnames=["arr1"])
