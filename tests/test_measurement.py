"""Unit tests for the MeasuredValue class

This file contains tests for the basic functionality of the package to record
measurements in MeasuredValue objects. It tests that the values stored are
correct, and that they can be printed in the correct format

"""

import pytest
import numpy as np
import qexpy as q

settings = q.get_settings()


class TestMeasuredValue:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        settings.reset()

    def test_record_measurement_with_no_uncertainty(self):
        measurement = q.Measurement(12.34, unit="m", name="test")
        settings.set_sig_figs_for_value(4)
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
        settings.reset()

    def test_record_repeated_measurements(self):
        # create value from repeated measurements
        measurement = q.Measurement([9, 10, 11], unit="m", name="test")
        assert measurement.value == 10
        for entry, expected in zip(measurement.raw_data, [9, 10, 11]):
            assert entry == expected
        # check the uncertainty
        assert measurement.std == 1
        assert measurement.error_on_mean == pytest.approx(0.5773502691896258)
        assert measurement.error == pytest.approx(0.5773502691896258)
        measurement.use_std_for_uncertainty()
        settings.set_sig_figs_for_error(2)
        assert measurement.error == 1
        assert str(measurement) == "test = 10.0 +/- 1.0 [m]"

    def test_modify_repeated_measurements(self):
        # create value from repeated measurements
        measurement = q.Measurement([9, 10, 11], unit="m", name="test")
        # check modification of center value
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

        # test correlation set by user
        c = q.Measurement(2, 0.5)
        d = q.Measurement(5, 1)
        # set the covariance
        q.set_covariance(c, d, 0.2)
        assert q.get_covariance(c, d) == 0.2
        # try unphysical covariance
        with pytest.raises(ValueError):
            q.set_covariance(c, d, 1)

    def test_error_weighted_mean_and_uncertainties(self):
        a = q.Measurement([10, 11], [0.1, 1])
        assert str(a) == "10.5 +/- 0.5"
        assert a.error_weighted_mean == 10.00990099009901
        assert a.propagated_error == 0.09950371902099892
        a.use_error_weighted_mean_as_value()
        a.use_propagated_error_for_uncertainty()
        settings.set_sig_figs_for_error(4)
        assert str(a) == "10.00990 +/- 0.09950"


class TestMeasurementArray:

    @pytest.fixture(autouse=True)
    def reset_environment(self):
        settings.reset()

    def test_record_measurement_array(self):
        a = q.MeasurementArray([1, 2, 3, 4, 5], 0.5, unit="m", name="length")
        assert np.sum(a).value == 15
        assert np.sum(a).error == 1.118033988749895
        assert a.mean().value == 3
        assert a.mean().error == 0.7071067811865476
        assert a.std() == 1.5811388300841898
        assert a.unit == "m"
        assert a.name == "length"

        first = a[0]
        assert first.value == 1
        assert first.error == 0.5
        assert first.unit == "m"
        assert first.name == "length_0"

        b = q.MeasurementArray([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5])
        assert np.sum(b).value == 15
        assert np.sum(b).error == 0.7416198487095663
        assert b.mean().value == 3
        assert b.mean().error == 0.7071067811865476
        assert b.std() == 1.5811388300841898

        second = b[1]
        assert second.value == 2
        assert second.error == 0.2

    def test_manipulate_measurement_array(self):
        a = q.MeasurementArray([1, 2, 3, 4, 5], 0.5, unit="m", name="length")
        a = a.append(6).append((7, 0.2)).append([8, 9]).append([(10, 0.1), (11, 0.3)])
        assert len(a) == 11
        assert a[5].value == 6
        assert a[6].value == 7
        assert a[6].error == 0.2
        assert a[7].value == 8
        assert a[9].value == 10
        assert a[10].error == 0.3

        a = a.delete(0)
        assert len(a) == 10
        assert a[0].value == 2
        assert a[0].error == 0.5

        a = a.insert(0, (2, 0.9))
        assert len(a) == 11
        assert a[0].value == 2
        assert a[0].error == 0.9
        assert a.name == "length"
        assert a.unit == "m"
