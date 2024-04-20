"""Unit tests for the ExperimentalValue class"""

import pytest

import qexpy as q
from qexpy.utils import Unit


class TestAttributes:
    """Tests accessing the attributes of the ExperimentalValue class"""

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

    @pytest.mark.parametrize(
        "constant, value, error",
        [
            (q.e, 1.602176634e-19, 0.0),
            (q.G, 6.6743e-11, 0.00015e-11),
            (q.me, 9.1093837015e-31, 0.0000000028e-31),
            (q.c, 299792458, 0.0),
            (q.eps0, 8.8541878128e-12, 0.0000000013e-12),
            (q.mu0, 1.25663706212e-6, 0.00000000019e-6),
            (q.h, 6.62607015e-34, 0.0),
            (q.hbar, 1.05457181e-34, 0.0),
            (q.kb, 1.380649e-23, 0.0),
        ],
    )
    def test_constants(self, constant, value, error):
        """Tests physical constants"""

        assert isinstance(constant, q.core.Constant)
        assert constant.value == value
        assert constant.error == error
