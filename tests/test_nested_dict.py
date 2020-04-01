from ngram.nested_dict import NestedDict
import pytest


class TestNestedDict:
    def test_getitem(self):
        testdict = NestedDict({"b": {"c": 50}})

        assert testdict["b"]["c"] == 50
        assert testdict["b"] == NestedDict({"c": 50})
        assert testdict["d"] == NestedDict()

    def test_get_by_path(self):
        testdict = NestedDict({"a": {"b": {"c": 100}}, "d": 1000})

        assert testdict.get_by_path(["a", "b", "c"]) == 100

    def test_set_by_path(self):
        testdict = NestedDict({"d": 1000})
        testdict.set_by_path(["a", "b", "c"], 100)

        assert testdict.get_by_path(["a", "b", "c"]) == 100

    def test_add_by_path(self):
        testdict = NestedDict({"d": 1000})
        testdict.add_by_path(["a", "b", "c"], 100)

        assert testdict.get_by_path(["a", "b", "c"]) == 100

        testdict.add_by_path(["a", "b", "c"], 100)

        assert testdict.get_by_path(["a", "b", "c"]) == 200
