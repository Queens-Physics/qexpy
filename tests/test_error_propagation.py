"""Unit tests for error propagation"""

import pytest
import qexpy as q
from qexpy.data.data import ExperimentalValue


@pytest.fixture()
def data():
    test_data = {
        "var1": q.Measurement(5, 0.2),
        "var2": q.Measurement(4, 0.1),
        "var3": q.Measurement(6.3, 0.5),
        "var4": q.Measurement(7.2, 0.5),
        "var5": q.Measurement([399.3, 404.6, 394.6, 396.3, 399.6,
                               404.9, 387.4, 404.9, 398.2, 407.2]),
        "var6": q.Measurement([193.2, 205.1, 192.6, 194.2, 196.6,
                               201, 184.7, 215.2, 203.6, 207.8])
    }
    yield test_data


class TestErrorPropagation:
    """Tests methods of error propagation"""

    @pytest.fixture(autouse=True)
    def reset_environment(self):  # pylint: disable=no-data-use
        """Resets all default configurations"""
        q.get_settings().reset()
        ExperimentalValue._correlations = {}  # pylint: disable=protected-access

    def test_error_propagation(self, data):
        """Tests error propagation for regular values"""

        result = data["var3"] * data["var4"] - data["var2"] / data["var1"]
        q.set_error_method(q.ErrorMethod.DERIVATIVE)
        assert result.value == pytest.approx(44.56, 1e-1)
        assert result.error == pytest.approx(4.783714456361291, 1e-1)
        q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        assert result.value == pytest.approx(44.56, 1e-1)
        assert result.error == pytest.approx(4.783714456361291, 1e-1)

        result = q.sqrt(data["var3"]) * data["var4"] - data["var2"] / q.exp(data["var1"])
        q.set_error_method(q.ErrorMethod.DERIVATIVE)
        assert result.value == pytest.approx(18.04490478513969, 1e-1)
        assert result.error == pytest.approx(1.4454463754287326, 1e-1)
        q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        assert result.value == pytest.approx(18.04490478513969, 1e-1)
        assert result.error == pytest.approx(1.4454463754287326, 1e-1)

        # test proper error propagation for data involved operations
        q.set_error_method(q.ErrorMethod.DERIVATIVE)
        result = data["var1"] * data["var2"] - data["var2"] * data["var1"]
        assert result.value == 0
        assert result.error == 0
        result = 2 * (
            data["var1"] + data["var2"]) - 2 * data["var1"] - data["var2"] - data["var2"]
        assert result.value == 0
        assert result.error == 0
        result = 2 * data["var1"] / data["var2"] * data["var2"] - data["var1"] - data["var1"]
        assert result.value == 0
        assert result.error == 0

    def test_error_propagation_for_correlated_values(self, data):
        """Test the error propagation on correlated values"""

        q.set_covariance(data["var1"], data["var3"], 0.09)
        q.set_covariance(data["var2"], data["var4"], -0.04)

        result = data["var3"] * data["var4"] - data["var2"] / data["var1"]
        q.set_error_method(q.ErrorMethod.DERIVATIVE)
        assert result.value == pytest.approx(44.56, 5e-2)
        assert result.error == pytest.approx(4.81581602638639, 5e-2)
        q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        assert result.value == pytest.approx(44.56, 5e-2)
        assert result.error == pytest.approx(4.81581602638639, 5e-2)

        result = q.sqrt(data["var3"]) * data["var4"] - data["var2"] / q.exp(data["var1"])
        q.set_error_method(q.ErrorMethod.DERIVATIVE)
        assert result.value == pytest.approx(18.04490478513969, 5e-2)
        assert result.error == pytest.approx(1.4483184455244065, 5e-2)
        q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        assert result.value == pytest.approx(18.04490478513969, 5e-2)
        assert result.error == pytest.approx(1.4483184455244065, 5e-2)

    def test_error_propagation_for_repeatedly_measured_correlated_values(self, data):
        """Tests correlated error propagation for repeatedly measured values"""

        q.set_covariance(data["var5"], data["var6"])

        result = data["var5"] + data["var6"]
        assert result.error == pytest.approx(4.54, 1e-1)

        q.set_error_method(q.ErrorMethod.MONTE_CARLO)
        assert result.error == pytest.approx(4.54, 1e-1)
