"""Unit tests for the internal options API."""

import pytest

import qexpy._config.config as cf


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    """Patch the global options with an empty dictionary."""

    with monkeypatch.context() as m:
        m.setattr(cf, "_global_options", {})
        m.setattr(cf, "options", cf.DictWrapper(cf._global_options))
        m.setattr(cf, "_registered_options", {})
        yield


def test_register_option():
    """Test registering a new configurable option."""

    cf.register_option("foo", 1)
    cf.register_option("abc.efg", 2)
    cf.register_option("bar.foo.abc", 3)
    cf.register_option("bar.test", 4)

    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 2},
        "bar": {"foo": {"abc": 3}, "test": 4},
    }


def test_option_accessors():
    """Test retrieving and modifying options."""

    cf.register_option("foo", 1)
    cf.register_option("abc.efg", 2)
    cf.register_option("bar.foo.abc", 3)
    cf.register_option("bar.test", 4)

    assert cf.get_option("bar.foo.abc") == 3
    assert cf.options.bar.foo.abc == 3

    cf.set_option("bar.test", 8, "foo", 9)
    assert cf.options.bar.test == 8
    assert cf.options.foo == 9

    cf.set_option({"foo": 10, "abc.efg": 20})
    assert cf.options.foo == 10
    assert cf.options.abc.efg == 20

    cf.options.abc.efg = 10
    assert cf.get_option("abc.efg") == 10


def test_set_option_context():
    """Test using `set_option_context` as a context manager."""

    cf.register_option("foo", 1)
    cf.register_option("abc.efg", 2)
    cf.register_option("bar.foo.abc", 3)
    cf.register_option("bar.test", 4)

    with cf.set_option_context("bar.test", 8, "foo", 9):
        assert cf.options.bar.test == 8
        assert cf.options.foo == 9

    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 2},
        "bar": {"foo": {"abc": 3}, "test": 4},
    }


def test_reset_options():
    """Test resetting options to default values."""

    cf.register_option("foo", 1)
    cf.register_option("abc.efg", 2)
    cf.register_option("bar.foo.abc", 3)
    cf.register_option("bar.test", 4)

    cf.options.foo = 10
    cf.options.abc.efg = 20
    cf.options.bar.foo.abc = 30
    cf.options.bar.test = 40

    cf.reset_options("foo")
    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 20},
        "bar": {"foo": {"abc": 30}, "test": 40},
    }

    cf.reset_options("bar")
    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 20},
        "bar": {"foo": {"abc": 3}, "test": 4},
    }

    cf.reset_options()
    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 2},
        "bar": {"foo": {"abc": 3}, "test": 4},
    }


def test_validation():
    """Test that the validator works."""

    def validator(x):
        if not isinstance(x, int):
            raise TypeError("x is not an integer.")
        if x < 0:
            raise ValueError("x must be positive.")

    cf.register_option("foo", 1, validator=validator)
    with pytest.raises(TypeError, match="not an integer"):
        cf.options.foo = "x"
    with pytest.raises(ValueError, match="must be positive"):
        cf.options.foo = -1


@pytest.mark.parametrize(
    "default, validator, value, error, message",
    [
        (
            "foo",
            cf.is_one_of_factory(["foo", "bar"]),
            "a",
            ValueError,
            "Value must be one of",
        ),
        (
            1,
            cf.is_positive_integer,
            -1,
            ValueError,
            "Value must be a positive integer",
        ),
        (True, cf.is_boolean, "hello", TypeError, "Value must be a boolean"),
        (
            (1.0, 2.0),
            cf.is_tuple_of_numbers,
            1,
            TypeError,
            "Value must be a tuple of length 2",
        ),
        (
            (1.0, 2.0),
            cf.is_tuple_of_numbers,
            (1, "a"),
            TypeError,
            "Value must be a tuple of numbers",
        ),
        (
            (1.0, 2.0),
            cf.is_tuple_of_numbers,
            (1, 2, 3),
            TypeError,
            "Value must be a tuple of length 2",
        ),
        (
            0.68,
            cf.is_number_in_range(0.0, 1.0),
            2.0,
            ValueError,
            "Value must be in range",
        ),
        (
            0.68,
            cf.is_number_in_range(0.0, 1.0),
            "1.5",
            TypeError,
            "The value must be a number",
        ),
    ],
)
def test_validators(default, validator, value, error, message):
    """Test the predefined validators."""

    cf.register_option("a", default, "", validator)

    with pytest.raises(error, match=message):
        cf.set_option("a", value)
