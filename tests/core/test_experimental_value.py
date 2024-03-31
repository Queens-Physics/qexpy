"""Unit tests for the ExperimentalValue class"""

import pytest

import qexpy as q
from qexpy.utils import Unit


class TestAttributes:
    """Tests accessing the attributes of the ExperimentalValue class"""

    def test_value_and_error(self):
        """Tests the value and the error attributes"""

        x = q.Measurement(1.23, 0.15)
        assert x.value == 1.23
        assert x.error == 0.15
        assert x.relative_error == 0.15 / 1.23

        x = q.Measurement(1.23, relative_error=0.15)
        assert x.value == 1.23
        assert x.error == 0.15 * 1.23
        assert x.relative_error == 0.15

    def test_name(self):
        """Tests the name attribute"""

        x = q.Measurement(1.23, 0.15, name="x")
        assert x.name == "x"

        x.name = "y"
        assert x.name == "y"

        with pytest.raises(TypeError):
            x.name = 1

    def test_unit(self):
        """Tests the unit attribute"""

        x = q.Measurement(1.23, 0.15, unit="kg*m/s^2")
        assert x.unit == Unit({"kg": 1, "m": 1, "s": -2})
        assert str(x.unit) == "kg⋅m/s^2"

        x.unit = "kg^2m^2s^-4"
        assert x.unit == Unit({"kg": 2, "m": 2, "s": -4})
        assert str(x.unit) == "kg^2⋅m^2/s^4"

        with pytest.raises(TypeError):
            x.unit = 1

    @pytest.mark.parametrize(
        "args, kwargs, expected",
        [
            ((1.23, 0.15), {}, "1.23 +/- 0.15"),
            ((1.23,), {"relative_error": 0.15}, "1.23 +/- 0.18"),
            ((1.23, 0.15), {"name": "x", "unit": "kg*m/s^2"}, "x = 1.23 +/- 0.15 [kg⋅m/s^2]"),
            ((1.23,), {"relative_error": 0.13, "name": "x"}, "x = 1.23 +/- 0.16"),
            ((1.23, 0.15), {"unit": "kg^2m^2s^-4"}, "1.23 +/- 0.15 [kg^2⋅m^2/s^4]"),
        ],
    )
    def test_str(self, args, kwargs, expected):
        """Tests the __str__ method of ExperimentalValue"""

        with q.option_context(
            "format.style.unit",
            "fraction",
            "format.precision.sig_fig",
            2,
            "format.precision.mode",
            "error",
        ):
            x = q.Measurement(*args, **kwargs)
            assert str(x) == expected
            assert repr(x) == expected


class TestCompare:
    """Tests for comparing experimental values"""

    def test_equal(self):
        """Tests the equal comparison methods of ExperimentalValue"""

        x = q.Measurement(1.23, 0.15)
        y = q.Measurement(1.23, 0.13)
        assert x == y
        assert x == 1.23
        assert 1.23 == y

        z = q.Measurement(1.22, 0.15)
        assert x != z
        assert x != 1.22
        assert 1.22 != x

        assert not x == "a"

    def test_not_equal(self):
        """Tests the not equal methods of ExperimentalValue"""

        x = q.Measurement(2.23, 0.13)
        y = q.Measurement(1.23, 0.15)

        assert x > y
        assert x >= y
        assert y < x
        assert y <= x

        assert x > 1
        assert x < 3
        assert x <= 2.23
        assert x >= 2.23

    def test_not_defined(self):
        """Test comparisons that are not defined"""

        x = q.Measurement(2.23, 0.13)
        with pytest.raises(TypeError):
            print(x < "a")
        with pytest.raises(TypeError):
            print(x > "a")
        with pytest.raises(TypeError):
            print(x <= "a")
        with pytest.raises(TypeError):
            print(x >= "a")
