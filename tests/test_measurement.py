"""Unit tests for the MeasuredValue class and the MeasurementArray class"""

import pytest
import numpy as np
import qexpy as q

from qexpy.data.data import RepeatedlyMeasuredValue, MeasuredValue
from qexpy.data.datasets import ExperimentalValueArray


class TestMeasuredValue:
    """Tests for single measurements"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):  # pylint: disable=no-self-use
        """Before method that restores all default configurations"""
        q.get_settings().reset()

    def test_measurement_without_uncertainties(self):  # pylint: disable=no-self-use
        """Tests recording measurements without specifying uncertainties"""

        measurement = q.Measurement(12.34)
        assert measurement.value == 12.34
        assert measurement.error == 0
        assert measurement.relative_error == 0

        q.set_sig_figs_for_value(4)
        assert str(measurement) == "12.34 +/- 0"

        measurement = q.Measurement(12.34, unit="m", name="length")
        assert str(measurement) == "length = 12.34 +/- 0 [m]"

    def test_measurement_with_uncertainties(self):  # pylint: disable=no-self-use
        """Tests recording measurements normally"""

        measurement = q.Measurement(12.34, 0.23, unit="m", name="length")
        assert measurement.value == 12.34
        assert measurement.error == 0.23
        assert measurement.relative_error == 0.23 / 12.34
        assert str(measurement) == "length = 12.3 +/- 0.2 [m]"

    def test_modifying_values(self):  # pylint: disable=no-self-use
        """Tests changing the values, uncertainties, or other properties"""

        measurement = q.Measurement(12.34, 0.23)
        assert measurement.value == 12.34
        assert measurement.error == 0.23

        measurement.value = 15.234
        assert measurement.value == 15.234

        measurement.error = 0.05
        assert measurement.error == 0.05
        assert measurement.relative_error == 0.05 / 15.234

        measurement.relative_error = 0.05
        assert measurement.relative_error == 0.05
        assert measurement.error == 0.05 * 15.234

        measurement.name = "test"
        measurement.unit = "m"
        assert str(measurement) == "test = 15.2 +/- 0.8 [m]"

    def test_covariance(self):  # pylint: disable=no-self-use
        """Test setting and getting covariance for values"""

        # test correlation set by user
        var1 = q.Measurement(2, 0.5)
        var2 = q.Measurement(5, 1)

        # set the covariance
        q.set_covariance(var1, var2, 0.2)
        assert q.get_covariance(var1, var2) == 0.2

        # try unphysical covariance
        with pytest.raises(ValueError):
            q.set_covariance(var1, var2, 1)


class TestRepeatedlyMeasuredValue:
    """Tests the recording an array of repeated measurements"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):  # pylint: disable=no-self-use
        """Before method that restores all default configurations"""
        q.get_settings().reset()

    def test_measurements_without_uncertainties(self):  # pylint: disable=no-self-use
        """Tests recording a measurement with repeated values"""

        # create value from repeated measurements
        measurement = q.Measurement([9, 10, 11], unit="m", name="length")
        assert isinstance(measurement, RepeatedlyMeasuredValue)
        assert measurement.value == 10

        # check internally the values are correct
        assert all(measurement.raw_data == np.array([9, 10, 11]))

        # check the uncertainty
        assert measurement.std == 1
        assert measurement.error_on_mean == pytest.approx(0.5773502691896258)
        assert measurement.error == pytest.approx(0.5773502691896258)

        q.set_sig_figs_for_error(2)
        assert str(measurement) == "length = 10.00 +/- 0.58 [m]"

    def test_measurements_with_uncertainties(self):  # pylint: disable=no-self-use
        """Tests recording a measurement with repeated values and uncertainties"""

        # create value from repeated measurements
        measurement = q.Measurement([9, 10, 11], 0.5)
        assert isinstance(measurement, RepeatedlyMeasuredValue)
        assert measurement.value == 10

        # check internally the values are correct
        raw_data = measurement.raw_data
        assert isinstance(raw_data, ExperimentalValueArray)
        assert all(raw_data.errors == 0.5)

        # check relative error works
        measurement = q.Measurement([9, 10, 11], relative_error=0.05)
        assert isinstance(measurement, RepeatedlyMeasuredValue)
        raw_data = measurement.raw_data
        assert isinstance(raw_data, ExperimentalValueArray)
        assert all(raw_data.errors == np.array([9, 10, 11]) * 0.05)

        # test error array
        measurement = q.Measurement([9, 10, 11], [0.5, 0.6, 0.7])
        assert isinstance(measurement, RepeatedlyMeasuredValue)
        raw_data = measurement.raw_data
        assert isinstance(raw_data, ExperimentalValueArray)
        assert all(raw_data.errors == np.array([0.5, 0.6, 0.7]))

        # check the uncertainty
        weights = np.asarray(list(1 / (err ** 2) for err in raw_data.errors))
        error_weighted_mean = float(np.sum(weights * raw_data.values) / np.sum(weights))
        assert measurement.error_weighted_mean == error_weighted_mean
        assert measurement.error_on_mean == raw_data.std() / np.sqrt(raw_data.size)

        # check displaying configurations
        measurement.use_error_weighted_mean_as_value()
        measurement.use_propagated_error_for_uncertainty()
        q.set_sig_figs_for_error(4)
        assert str(measurement) == "9.7778 +/- 0.3367"

    def test_modify_repeated_measurements(self):  # pylint: disable=no-self-use
        """Repeatedly measured values should throw a warning when the value is changed"""

        # create value from repeated measurements
        measurement = q.Measurement([9, 10, 11], unit="m", name="length")

        # check modification of center value
        with pytest.warns(UserWarning):
            # modification of value is allowed but warned
            measurement.value = 20

        assert measurement.value == 20
        # check that the measurement was casted to parent because of change in value

        assert not isinstance(measurement, RepeatedlyMeasuredValue)
        assert isinstance(measurement, MeasuredValue)

    def test_covariance(self):  # pylint: disable=no-self-use
        """Test setting, retrieving, and calculating covariance between values"""

        # create values to test
        var1 = q.Measurement([1, 1.2, 1.3, 1.4])
        var2 = q.Measurement([2, 2.1, 3, 2.3])

        # declare covariance relationship
        q.set_covariance(var1, var2)

        # verify covariance calculations
        assert q.get_covariance(var2, var1) == pytest.approx(0.0416667)


class TestMeasuredValueArray:
    """Tests for the MeasuredValueArray class"""

    def test_record_measurement_array(self):  # pylint: disable=no-self-use
        """Test taking measurements"""

        # test measurement array with 0 error
        array = q.MeasurementArray([1, 2, 3, 4, 5], unit="m", name="length")
        assert all(array.errors == 0)

        # test measurement array with single error
        array = q.MeasurementArray([1, 2, 3, 4, 5], 0.5, unit="m", name="length")
        assert np.sum(array) == 15
        assert array.mean() == 3
        assert array.std() == 1.5811388300841898
        assert array.unit == "m"
        assert array.name == "length"

        first = array[0]
        assert first.value == 1
        assert first.error == 0.5
        assert first.unit == "m"
        assert first.name == "length_0"

        # test measurement array with array of errors
        array = q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5])
        assert np.sum(array) == 15
        assert array.mean() == 3
        assert array.std() == 1.5811388300841898

        second = array[1]
        assert second.value == 2
        assert second.error == 0.2

    def test_manipulate_measurement_array(self):  # pylint: disable=no-self-use
        """Tests changing measurement arrays"""

        arr = q.MeasurementArray([1, 2, 3, 4, 5], 0.5, unit="m", name="length")
        arr = arr.append(6).append((7, 0.2)).append([8, 9]).append([(10, 0.1), (11, 0.3)])
        assert len(arr) == 11
        assert arr[5].value == 6
        assert arr[6].value == 7
        assert arr[6].error == 0.2
        assert arr[7].value == 8
        assert arr[9].value == 10
        assert arr[10].error == 0.3

        arr = arr.delete(0)
        assert len(arr) == 10
        assert arr[0].value == 2
        assert arr[0].error == 0.5

        arr = arr.insert(0, (2, 0.9))
        assert len(arr) == 11
        assert arr[0].value == 2
        assert arr[0].error == 0.9
        assert arr.name == "length"
        assert arr.unit == "m"

        # check that re-indexing happened
        assert arr[1].name == "length_1"
