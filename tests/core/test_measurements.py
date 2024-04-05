"""Unit tests for measurements"""

# pylint: disable=too-few-public-methods

import pytest

import numpy as np
import qexpy as q


class TestMeasurement:
    """Unit tests for taking measurements"""

    def test_single_measurement(self):
        """Tests taking a single measurement"""

        x = q.Measurement(1.23, 0.15)
        assert x.value == 1.23
        assert x.error == 0.15
        assert x.std == 0.15
        assert x.relative_error == 0.15 / 1.23

        x = q.Measurement(1.23, relative_error=0.15)
        assert x.value == 1.23
        assert x.error == 0.15 * 1.23
        assert x.relative_error == 0.15

        x = q.Measurement(1.23)
        assert x.value == 1.23
        assert x.error == 0.0

    def test_invalid_single_measurement(self):
        """Tests taking a single measurement with invalid arguments"""

        with pytest.raises(TypeError, match="The data must be a real number or an array"):
            q.Measurement("a")

        with pytest.raises(TypeError, match="The error must be a real number"):
            q.Measurement(1.23, "a")

        with pytest.raises(ValueError, match="The error must be non-negative"):
            q.Measurement(1.23, -0.1)

        with pytest.raises(ValueError, match="The error must be non-negative"):
            q.Measurement(1.23, relative_error=-0.1)

    def test_repeated_measurement_no_error(self):
        """Tests a repeated measurement"""

        m = q.Measurement([4.9, 5, 5.1])
        assert m.std == pytest.approx(0.1)
        assert m.error == pytest.approx(0.1 / np.sqrt(3))
        assert m.value == 5
        assert m.data.tolist() == [4.9, 5, 5.1]

    def test_repeated_measurement_error_array(self):
        """Tests a repeated measurement with an array of errors"""

        m = q.Measurement([4.9, 5, 5.1], relative_error=[0.1, 0.2, 0.3])
        assert m._errors.tolist() == [0.1 * 4.9, 0.2 * 5, 0.3 * 5.1]

        m = q.Measurement([4.9, 5, 5.1], [0.1, 0.5, 0.5])
        assert m.value == 5
        assert m.std == pytest.approx(0.1)
        assert m.error == pytest.approx(0.1 / np.sqrt(3))

        m.use_error_weighted_mean()
        assert m.value == pytest.approx(4.911111111111111)
        assert m.error == pytest.approx(0.09622504486493764)

        m.use_mean_and_std()
        assert m.value == pytest.approx(5)
        assert m.error == pytest.approx(0.1)

        m.use_mean_and_standard_error()
        assert m.value == pytest.approx(5)
        assert m.error == pytest.approx(0.1 / np.sqrt(3))

    def test_repeated_measurement_single_error(self):
        """Tests a repeated measurement with a single error"""

        m = q.Measurement([4.9, 5, 5.1], 0.1)
        assert m.value == 5
        assert m.error == pytest.approx(0.1 / np.sqrt(3))
        assert m.data.tolist() == [4.9, 5, 5.1]
        assert m._errors.tolist() == [0.1, 0.1, 0.1]

        m = q.Measurement([4.9, 5, 5.1], relative_error=0.1)
        assert m.value == 5
        assert m.error == pytest.approx(0.1 / np.sqrt(3))
        assert m.data.tolist() == [4.9, 5, 5.1]
        assert m._errors.tolist() == [0.1 * 4.9, 0.1 * 5, 0.1 * 5.1]

    @pytest.mark.parametrize(
        "error, args, kwargs, msg",
        [
            (
                TypeError,
                ([4.9, 5.0, 5.1], "a"),
                {},
                "The error must be a non-negative real number or an array",
            ),
            (
                TypeError,
                ([4.9, 5.0, 5.1],),
                {"relative_error": "a"},
                "The relative error must be a non-negative real number or an array",
            ),
            (ValueError, ([4.9, 5.0, 5.1], -0.1), {}, "The error must be non-negative!"),
            (
                ValueError,
                ([4.9, 5.0, 5.1],),
                {"relative_error": -0.1},
                "The error must be non-negative!",
            ),
            (
                ValueError,
                ([4.9, 5.0, 5.1], [0.1, 0.5]),
                {},
                "The data and error arrays must have the same length!",
            ),
            (
                ValueError,
                ([4.9, 5.0, 5.1],),
                {"relative_error": [0.1, 0.5]},
                "The data and error arrays must have the same length!",
            ),
            (
                ValueError,
                ([4.9, 5.0, 5.1], [0.1, -0.5, 0.5]),
                {},
                "The error must be non-negative!",
            ),
            (
                ValueError,
                ([4.9, 5.0, 5.1],),
                {"relative_error": [0.1, -0.5, 0.5]},
                "The error must be non-negative!",
            ),
        ],
    )
    def test_invalid_repeated_measurement(self, error, args, kwargs, msg):
        """Tests taking a repeated measurement with invalid arguments"""

        with pytest.raises(error, match=msg):
            q.Measurement(*args, **kwargs)


class TestCorrelations:
    """Tests for setting correlations between measurements"""

    def test_zero_correlation(self):
        """Tests that measurements are not originally correlated"""

        m1 = q.Measurement([4.9, 5, 5.1])
        m2 = q.Measurement([3.1, 3.3, 3.2])
        assert m1.get_correlation(m2) == 0.0
        assert m1.get_covariance(m2) == 0.0

        m1 = q.Measurement(5, 0.1)
        m2 = q.Measurement(3, 0.5)
        assert m2.get_correlation(m1) == 0.0
        assert m2.get_covariance(m1) == 0.0

    def test_self_correlation(self):
        """Tests that measurements are correlated to themselves"""

        m1 = q.Measurement(5, 0.2)
        assert m1.get_correlation(m1) == 1.0
        assert m1.get_covariance(m1) == pytest.approx(0.04)

        m2 = q.Measurement([4.9, 5, 5.1])
        assert m2.get_correlation(m2) == 1.0
        assert m2.get_covariance(m2) == pytest.approx(0.01)

    def test_set_correlation(self):
        """Tests that correlation can be set"""

        a = q.Measurement([4.9, 5, 5.1])
        b = q.Measurement([3.1, 3.3, 3.2])
        a.set_covariance(b)
        assert a.get_covariance(b) == 0.005
        assert b.get_correlation(a) == 0.5

        a = q.Measurement([4.9, 5, 5.1])
        b = q.Measurement([3.1, 3.3, 3.2])
        b.set_correlation(a)
        assert a.get_covariance(b) == 0.005
        assert b.get_correlation(a) == 0.5

        a = q.Measurement([4.9, 5, 5.1])
        b = q.Measurement([3.1, 3.3, 3.2])
        b.set_correlation(a, 0.25)
        assert a.get_covariance(b) == 0.0025
        assert b.get_correlation(a) == 0.25

        a = q.Measurement([4.9, 5, 5.1])
        b = q.Measurement([3.1, 3.3, 3.2])
        b.set_covariance(a, 0.0025)
        assert a.get_covariance(b) == 0.0025
        assert b.get_correlation(a) == 0.25

    def test_invalid_argument_type(self):
        """Tests covariance is only defined between measurements"""

        m = q.Measurement([4.9, 5, 5.1])
        with pytest.raises(TypeError, match="only defined between measurements"):
            m.set_correlation(2)
        with pytest.raises(TypeError, match="only defined between measurements"):
            m.set_covariance(2)
        with pytest.raises(TypeError, match="only defined between measurements"):
            m.get_correlation(2)
        with pytest.raises(TypeError, match="only defined between measurements"):
            m.get_covariance(2)

    def test_invalid_covariance_and_correlation(self):
        """Tests that the correlation must be between -1 and 1"""

        a = q.Measurement([4.9, 5, 5.1])
        b = q.Measurement([3.1, 3.3, 3.2])

        with pytest.raises(ValueError, match="must be between -1 and 1"):
            a.set_correlation(b, 2)

        with pytest.raises(ValueError, match="non-physical"):
            a.set_covariance(b, 100)

        a = q.Measurement(5)
        b = q.Measurement(3)

        with pytest.raises(ArithmeticError, match="values with a 0 uncertainty"):
            a.set_correlation(b, 0.5)

        with pytest.raises(ArithmeticError, match="values with a 0 uncertainty"):
            a.set_covariance(b, 0.005)
