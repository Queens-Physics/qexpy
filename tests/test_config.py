"""Unit tests for accessing and changing configurations"""

import pytest

import qexpy._config.config as cf


class TestConfig:
    """Tests for accessing and changing configurations"""

    @pytest.fixture(autouse=True)
    def clean_config(self, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(cf, "_global_config", {})
            m.setattr(cf, "options", cf.DictWrapper(cf._global_config))
            m.setattr(cf, "_registered_options", {})
            yield

    def test_register_option(self):
        """Tests registering a new option"""

        cf.register_option("foo", 1)
        cf.register_option("abc.cba", 2)
        cf.register_option("bar.foo.test", 3)
        cf.register_option("bar.foo.abc", 4)

        with pytest.raises(ValueError, match="def is a python keyword"):
            cf.register_option("abc.def", 2)

        assert cf._global_config == {
            'foo': 1, 'abc': {'cba': 2}, 'bar': {'foo': {'test': 3, 'abc': 4}}}

        with pytest.raises(ValueError, match="Path prefix is already an option"):
            cf.register_option("foo.bar", 1)

        with pytest.raises(ValueError, match="Path prefix is already an option"):
            cf.register_option("foo.bar.abc", 1)

        with pytest.raises(ValueError, match="Path is already a prefix to other options"):
            cf.register_option("bar.foo", 1)

    def test_get_option(self):
        """Tests getting an option"""

        cf.register_option("a", 1)
        cf.register_option("b.c", "hello")

        assert cf.get_option("a") == 1
        assert cf.get_option("b.c") == "hello"

        with pytest.raises(KeyError, match="No such option"):
            cf.get_option("no_such_option")

    def test_set_option(self):
        """Tests setting an option"""

        cf.register_option("a", 1)
        cf.register_option("b.c", "hello")

        cf.set_option("a", 2)
        cf.set_option("b.c", "world")

        assert cf.get_option("a") == 2
        assert cf.get_option("b.c") == "world"

        with pytest.raises(KeyError, match="No such option"):
            cf.set_option("no_such_option", 3)

    def test_validation(self):
        """Tests validation of option values"""

        cf.register_option("a", "foo", cf.is_one_of_factory(["foo", "bar"]))
        cf.register_option("b.c", 1, cf.is_positive_integer)
        cf.register_option("d", True, cf.is_boolean)
        cf.register_option("e", (1.0, 2.0), cf.is_tuple_of_floats)

        with pytest.raises(ValueError, match="Value must be one of"):
            cf.set_option("a", "hello")

        with pytest.raises(ValueError, match="Value must be a positive integer"):
            cf.set_option("b.c", -1)

        with pytest.raises(TypeError, match="Value must be a boolean"):
            cf.set_option("d", "hello")

        with pytest.raises(TypeError, match="Value must be a tuple of length 2"):
            cf.set_option("e", (1, 2, 3))

        with pytest.raises(TypeError, match="Value must be a tuple of length 2"):
            cf.set_option("e", 2)

        with pytest.raises(ValueError, match="Value must be a tuple of floats"):
            cf.set_option("e", (1.0, 'a'))

    def test_reset_options(self):
        """Tests resetting options"""

        cf.register_option("a", 1)
        cf.register_option("b.c", "hello")

        cf.set_option("a", 2)
        cf.set_option("b.c", "world")

        assert cf.get_option("a") == 2
        assert cf.get_option("b.c") == "world"

        cf.reset_option("a")
        assert cf.get_option("a") == 1
        assert cf.get_option("b.c") == "world"

        cf.reset_option()
        assert cf.get_option("a") == 1
        assert cf.get_option("b.c") == "hello"

        with pytest.raises(KeyError, match="No such option"):
            cf.reset_option("no_such_option")

    def test_attribute_access(self):
        """Tests accessing options using the dot notation"""

        cf.register_option("foo", 1)
        cf.register_option("abc.cba", 2)
        cf.register_option("bar.foo.test", 3)
        cf.register_option("bar.foo.abc", 4)

        assert cf.options.foo == 1
        assert cf.options.abc.cba == 2
        assert cf.options.bar.foo.test == 3
        assert cf.options.bar.foo.abc == 4

        cf.options.abc.cba = 5
        cf.options.bar.foo.abc = 6

        assert cf.get_option("abc.cba") == 5
        assert cf.get_option("bar.foo.abc") == 6

        with pytest.raises(KeyError, match="You can only set the value of existing options"):
            cf.options.no_such_option = 1

        with pytest.raises(KeyError, match="No such option"):
            print(cf.options.no_such_option)
