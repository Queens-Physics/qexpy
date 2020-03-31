"""Unit tests for taking arrays of measurements"""

import pytest
import qexpy as q
import numpy as np

from qexpy.utils.exceptions import IllegalArgumentError

from qexpy.data.datasets import ExperimentalValueArray


class TestExperimentalValueArray:
    """tests for the ExperimentalValueArray class"""

    def test_record_measurement_array(self):
        """tests for recording a measurement array in different ways"""

        a = q.MeasurementArray([1, 2, 3, 4, 5])
        assert isinstance(a, ExperimentalValueArray)
        assert all(a.values == [1, 2, 3, 4, 5])
        assert all(a.errors == [0, 0, 0, 0, 0])
        assert str(a) == "[ 1 +/- 0, 2 +/- 0, 3 +/- 0, 4 +/- 0, 5 +/- 0 ]"

        with pytest.raises(TypeError):
            q.MeasurementArray(1)

        b = q.MeasurementArray([1, 2, 3, 4, 5], 0.5)
        assert all(b.errors == [0.5, 0.5, 0.5, 0.5, 0.5])
        c = q.MeasurementArray([1, 2, 3, 4, 5], relative_error=0.1)
        assert c.errors == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
        d = q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5])
        assert all(d.errors == [0.1, 0.2, 0.3, 0.4, 0.5])
        e = q.MeasurementArray([1, 2, 3, 4, 5], relative_error=[0.1, 0.2, 0.3, 0.4, 0.5])
        assert e.errors == pytest.approx([0.1, 0.4, 0.9, 1.6, 2.5])

        with pytest.raises(ValueError):
            q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError):
            q.MeasurementArray([1, 2, 3, 4, 5], relative_error=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(TypeError):
            q.MeasurementArray([1, 2, 3, 4, 5], '1')
        with pytest.raises(TypeError):
            q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, '0.5'])
        with pytest.raises(TypeError):
            q.MeasurementArray([1, 2, 3, 4, '5'])
        with pytest.raises(ValueError):
            q.MeasurementArray([1, 2, 3, 4, 5], -0.5)
        with pytest.raises(ValueError):
            q.MeasurementArray([1, 2, 3, 4, 5], [0.5, 0.5, 0.5, 0.5, -0.5])

        f = q.MeasurementArray(data=[1, 2, 3, 4], error=0.5, name="test", unit="m")
        assert f.name == "test"
        assert f.unit == "m"
        assert str(f) == "test = [ 1.0 +/- 0.5, 2.0 +/- 0.5, 3.0 +/- 0.5, 4.0 +/- 0.5 ] (m)"
        assert str(f[0]) == "test_0 = 1.0 +/- 0.5 [m]"
        assert str(f[-1]) == "test_3 = 4.0 +/- 0.5 [m]"

        g = q.MeasurementArray(
            [q.Measurement(5, 0.5), q.Measurement(10, 0.5)], name="test", unit="m")
        assert str(g[-1]) == "test_1 = 10.0 +/- 0.5 [m]"

        h = q.MeasurementArray([q.Measurement(5, 0.5), q.Measurement(10, 0.5)], error=0.1)
        assert str(h[-1]) == "10.0 +/- 0.1"

    def test_manipulate_measurement_array(self):
        """tests for manipulating a measurement array"""

        a = q.MeasurementArray([1, 2, 3, 4], 0.5, name="test", unit="m")
        a = a.append(q.Measurement(5, 0.5))
        assert str(a[-1]) == "test_4 = 5.0 +/- 0.5 [m]"
        a = a.insert(1, (1.5, 0.5))
        assert str(a[1]) == "test_1 = 1.5 +/- 0.5 [m]"
        assert str(a[-1]) == "test_5 = 5.0 +/- 0.5 [m]"
        a = a.delete(1)
        assert str(a[1]) == "test_1 = 2.0 +/- 0.5 [m]"

        with pytest.raises(TypeError):
            a.name = 1
        with pytest.raises(TypeError) as e:
            a.unit = 1
        assert str(e.value) == "Cannot set unit to \"int\"!"

        a.name = "speed"
        a.unit = "m/s"
        assert a.name == "speed"
        assert a.unit == "m⋅s^-1"
        assert str(a[4]) == "speed_4 = 5.0 +/- 0.5 [m⋅s^-1]"

        a = a.append(6)
        assert str(a[5]) == "speed_5 = 6 +/- 0 [m⋅s^-1]"

        a[3] = 10
        assert str(a[3]) == "speed_3 = 10.0 +/- 0.5 [m⋅s^-1]"
        a[4] = (10, 0.6)
        assert str(a[4]) == "speed_4 = 10.0 +/- 0.6 [m⋅s^-1]"

        with pytest.raises(TypeError):
            a[2] = 'a'

        b = q.MeasurementArray([5, 6, 7], 0.5)
        b[-1] = (8, 0.5)

        a = a.append(b)
        assert str(a[-1]) == "speed_8 = 8.0 +/- 0.5 [m⋅s^-1]"

        a = a.append([8, 9, 10])
        assert str(a[-1]) == "speed_11 = 10 +/- 0 [m⋅s^-1]"

    def test_calculations_with_measurement_array(self):
        """tests for calculating properties of a measurement array"""

        a = q.MeasurementArray([1, 2, 3, 4, 5])
        assert a.mean() == 3
        assert a.std() == pytest.approx(1.58113883008419)
        assert a.sum() == 15
        assert a.error_on_mean() == pytest.approx(0.707106781186548)

        with pytest.warns(UserWarning):
            assert np.isnan(a.error_weighted_mean())
        with pytest.warns(UserWarning):
            assert np.isnan(a.propagated_error())

        b = q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5])
        assert b.error_weighted_mean() == pytest.approx(1.5600683241601823)
        assert b.propagated_error() == pytest.approx(0.08265842980736918)


class TestXYDataSet:
    """tests for the XYDataSet class"""

    def test_construct_data_set(self):
        """test for various ways to construct a data set"""

        with pytest.raises(ValueError):
            q.XYDataSet([0, 1, 2, 3, 4], [0, 1, 2, 3])

        with pytest.raises(IllegalArgumentError):
            q.XYDataSet(0, 0)

        dataset = q.XYDataSet([0, 1, 2, 3, 4], [0, 0.2, 0.5, 0.8, 1.3],
                              xerr=0.1, yerr=[0.1, 0.1, 0.1, 0.1, 0.5], name="test",
                              xname="time", xunit="s", yname="distance", yunit="m")
        assert dataset.xname == "time"
        assert dataset.xunit == "s"
        assert dataset.yname == "distance"
        assert dataset.yunit == "m"
        assert dataset.name == "test"

        assert all(dataset.xvalues == [0, 1, 2, 3, 4])
        assert all(dataset.xerr == [0.1, 0.1, 0.1, 0.1, 0.1])
        assert all(dataset.yvalues == [0, 0.2, 0.5, 0.8, 1.3])
        assert all(dataset.yerr == [0.1, 0.1, 0.1, 0.1, 0.5])

        a = q.MeasurementArray([1, 2, 3, 4, 5])
        b = q.MeasurementArray([10, 20, 30, 40, 50])
        dataset = q.XYDataSet(a, b, xerr=0.5, yerr=0.5, name="test",
                              xname="x", yname="y", xunit="m", yunit="s")
        assert dataset.name == "test"
        assert all(dataset.xerr == [0.5, 0.5, 0.5, 0.5, 0.5])
        assert str(dataset.xdata[0]) == "x_0 = 1.0 +/- 0.5 [m]"

        c = q.MeasurementArray([1, 2, 3, 4, 5])
        d = q.MeasurementArray([10, 20, 30, 40, 50])
        dataset = q.XYDataSet(c, d)
        assert all(dataset.xerr == [0, 0, 0, 0, 0])
        assert str(dataset.xdata[0]) == "1 +/- 0"

    def test_manipulate_data_set(self):
        """tests for changing values in a data set"""

        dataset = q.XYDataSet([0, 1, 2, 3, 4], [0, 0.2, 0.5, 0.8, 1.3])
        dataset.name = "test"
        assert dataset.name == "test"
        dataset.xname = "x"
        assert dataset.xname == "x"
        dataset.xunit = "m"
        assert dataset.xunit == "m"
        assert str(dataset.xdata[0]) == "x_0 = 0 +/- 0 [m]"
        dataset.yname = "y"
        assert dataset.yname == "y"
        dataset.yunit = "s"
        assert dataset.yunit == "s"
        assert str(dataset.ydata[0]) == "y_0 = 0 +/- 0 [s]"

        with pytest.raises(TypeError):
            dataset.name = 1
        with pytest.raises(TypeError):
            dataset.xname = 1
        with pytest.raises(TypeError):
            dataset.xunit = 1
        with pytest.raises(TypeError):
            dataset.yname = 1
        with pytest.raises(TypeError):
            dataset.yunit = 1
