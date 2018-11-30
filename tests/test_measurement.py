"""Unit tests for the MeasuredValue class

This file contains tests for the basic functionality of the package to record
measurements in MeasuredValue objects. It tests that the values stored are
correct, and that they can be printed in the correct format

"""

# noinspection PyPackageRequirements
import pytest
import qexpy as q


class TestMeasuredValue:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        q.reset_default_configuration()

    def test_record_measurement_with_no_uncertainty(self):
        measurement = q.Measurement(12.34, unit="m", name="test")
        assert measurement.value == 12.34
        assert measurement.error == 0
        assert measurement.relative_error == 0
        assert str(measurement) == "test = 12.34 +/- 0 [m]"

    def test_record_measurement_with_uncertainty(self):
        measurement = q.Measurement(12.34, 0.23, unit="m", name="test")
        assert measurement.value == 12.34
        assert measurement.error == 0.23
        assert measurement.relative_error == pytest.approx(0.0186386, 1e-5)
        assert str(measurement) == "test = 12.3 +/- 0.2 [m]"

    def test_modify_values_of_measurement(self):
        measurement = q.Measurement(12.34, 0.23)
        measurement.value = 15.234
        assert measurement.value == 15.234
        measurement.error = 0.05
        assert measurement.error == 0.05
        measurement.relative_error = 0.05
        assert measurement.relative_error == 0.05
        assert measurement.error == 0.7617
        measurement.name = "test"
        measurement.unit = "m"
        assert str(measurement) == "test = 15.2 +/- 0.8 [m]"


class TestRepeatedlyMeasuredValue:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        q.reset_default_configuration()

    def test_record_repeated_measurements(self):
        # create value from repeated measurements
        measurement = q.Measurement([9, 10, 11], unit="m", name="test")
        assert measurement.value == 10
        for entry, expected in zip(measurement.raw_data, [9, 10, 11]):
            assert entry == expected
        # check the uncertainty
        assert measurement.std == pytest.approx(0.8164966, 1e-6)
        assert measurement.error_on_mean == pytest.approx(0.4714045, 1e-6)
        assert measurement.error == pytest.approx(0.4714045, 1e-6)
        measurement.use_std_for_uncertainty()
        assert measurement.error == pytest.approx(0.8164966, 1e-6)
        assert str(measurement) == "test = 10.0 +/- 0.8 [m]"

    def test_modify_repeated_measurements(self):
        # create value from repeated measurements
        measurement = q.Measurement([9, 10, 11], unit="m", name="test")
        # check modification of uncertainty and values
        with pytest.warns(UserWarning):
            # modification of error is allowed but warned
            measurement.error = 0.5
        assert measurement.error == 0.5
        with pytest.warns(UserWarning):
            # modification of value is allowed but warned
            measurement.value = 20
        assert measurement.value == 20
        # check that the measurement was casted to parent because of change in value
        with pytest.raises(AttributeError):
            # the object is no longer a RepeatedlyMeasuredValue
            print(measurement.raw_data)

    def test_covariance(self):
        # create values to test
        a = q.Measurement([1, 1.2, 1.3, 1.4])
        b = q.Measurement([2, 2.1, 3, 2.3])
        # declare covariance relationship
        q.set_covariance(a, b)
        # verify covariance calculations
        assert q.get_covariance(b, a) == pytest.approx(0.0416667)
