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
    """Test using `set_option` as a context manager."""

    cf.register_option("foo", 1)
    cf.register_option("abc.efg", 2)
    cf.register_option("bar.foo.abc", 3)
    cf.register_option("bar.test", 4)

    with cf.set_option("bar.test", 8, "foo", 9):
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

    cf.reset_option("foo")
    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 20},
        "bar": {"foo": {"abc": 30}, "test": 40},
    }

    cf.reset_option("bar")
    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 20},
        "bar": {"foo": {"abc": 3}, "test": 4},
    }

    cf.reset_option()
    assert cf._global_options == {
        "foo": 1,
        "abc": {"efg": 2},
        "bar": {"foo": {"abc": 3}, "test": 4},
    }
