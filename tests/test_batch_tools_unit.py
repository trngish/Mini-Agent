"""Comprehensive unit tests for batch_tools module.

Tests cover the _ensure_list helper function with all input variants:
None, list, JSON-encoded string, plain string, and edge cases.
"""

import json

from mini_agent.tools.batch_tools import _ensure_list


class TestEnsureListNone:
    """Tests for _ensure_list with None input."""

    def test_none_returns_empty_list(self):
        result = _ensure_list(None)
        assert result == []

    def test_none_returns_list_type(self):
        result = _ensure_list(None)
        assert isinstance(result, list)


class TestEnsureListListInput:
    """Tests for _ensure_list with list input."""

    def test_empty_list(self):
        result = _ensure_list([])
        assert result == []

    def test_list_of_strings(self):
        data = ["a", "b", "c"]
        result = _ensure_list(data)
        assert result == ["a", "b", "c"]

    def test_list_of_integers(self):
        data = [1, 2, 3]
        result = _ensure_list(data)
        assert result == [1, 2, 3]

    def test_list_of_dicts(self):
        data = [{"key": "val1"}, {"key": "val2"}]
        result = _ensure_list(data)
        assert result == [{"key": "val1"}, {"key": "val2"}]

    def test_list_with_mixed_types(self):
        data = ["str", 42, None, True]
        result = _ensure_list(data)
        assert result == ["str", 42, None, True]

    def test_list_with_single_element(self):
        data = ["only"]
        result = _ensure_list(data)
        assert result == ["only"]

    def test_list_identity(self):
        data = ["a", "b"]
        result = _ensure_list(data)
        assert result is data

    def test_nested_list(self):
        data = [[1, 2], [3, 4]]
        result = _ensure_list(data)
        assert result == [[1, 2], [3, 4]]


class TestEnsureListJsonString:
    """Tests for _ensure_list with JSON-encoded string input."""

    def test_json_list_of_strings(self):
        result = _ensure_list('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_list_of_integers(self):
        result = _ensure_list("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_json_empty_list(self):
        result = _ensure_list("[]")
        assert result == []

    def test_json_single_element_list(self):
        result = _ensure_list('["only"]')
        assert result == ["only"]

    def test_json_list_of_dicts(self):
        data = json.dumps([{"key": "val1"}, {"key": "val2"}])
        result = _ensure_list(data)
        assert result == [{"key": "val1"}, {"key": "val2"}]

    def test_json_nested_list(self):
        result = _ensure_list("[[1, 2], [3, 4]]")
        assert result == [[1, 2], [3, 4]]

    def test_json_list_with_null(self):
        result = _ensure_list('["a", null, "b"]')
        assert result == ["a", None, "b"]

    def test_json_list_with_booleans(self):
        result = _ensure_list("[true, false]")
        assert result == [True, False]


class TestEnsureListJsonNonList:
    """Tests for _ensure_list with JSON string that is NOT a list."""

    def test_json_object_returns_wrapped(self):
        result = _ensure_list('{"key": "value"}')
        assert result == ['{"key": "value"}']

    def test_json_number_returns_wrapped(self):
        result = _ensure_list("42")
        assert result == ["42"]

    def test_json_string_returns_wrapped(self):
        result = _ensure_list('"hello"')
        assert result == ['"hello"']

    def test_json_boolean_true_returns_wrapped(self):
        result = _ensure_list("true")
        assert result == ["true"]

    def test_json_boolean_false_returns_wrapped(self):
        result = _ensure_list("false")
        assert result == ["false"]

    def test_json_null_returns_wrapped(self):
        result = _ensure_list("null")
        assert result == ["null"]


class TestEnsureListPlainString:
    """Tests for _ensure_list with non-JSON plain string input."""

    def test_plain_string_returns_wrapped(self):
        result = _ensure_list("hello world")
        assert result == ["hello world"]

    def test_string_with_special_chars(self):
        result = _ensure_list("file*.py")
        assert result == ["file*.py"]

    def test_string_with_newlines(self):
        result = _ensure_list("line1\nline2")
        assert result == ["line1\nline2"]

    def test_invalid_json_string(self):
        result = _ensure_list("[invalid json")
        assert result == ["[invalid json"]

    def test_partial_json_string(self):
        result = _ensure_list('{"key":')
        assert result == ['{"key":']

    def test_single_character_string(self):
        result = _ensure_list("x")
        assert result == ["x"]

    def test_whitespace_string(self):
        result = _ensure_list("   ")
        assert result == ["   "]


class TestEnsureListEdgeCases:
    """Tests for _ensure_list with edge case inputs."""

    def test_integer_input(self):
        result = _ensure_list(42)
        assert result == [42]

    def test_float_input(self):
        result = _ensure_list(3.14)
        assert result == [3.14]

    def test_boolean_true_input(self):
        result = _ensure_list(True)
        assert result == [True]

    def test_boolean_false_input(self):
        result = _ensure_list(False)
        assert result == [False]

    def test_dict_input(self):
        data = {"key": "value"}
        result = _ensure_list(data)
        assert result == [{"key": "value"}]

    def test_tuple_input(self):
        data = ("a", "b")
        result = _ensure_list(data)
        assert result == [("a", "b")]

    def test_set_input(self):
        data = {"a", "b"}
        result = _ensure_list(data)
        assert len(result) == 1
        assert isinstance(result[0], set)

    def test_json_list_with_whitespace(self):
        result = _ensure_list('  ["a", "b"]  ')
        assert result == ["a", "b"]

    def test_json_list_compact(self):
        result = _ensure_list('["a","b"]')
        assert result == ["a", "b"]

    def test_json_list_with_spaces(self):
        result = _ensure_list('[ "a" , "b" ]')
        assert result == ["a", "b"]

    def test_json_list_unicode(self):
        result = _ensure_list('["你好", "世界"]')
        assert result == ["你好", "世界"]

    def test_json_list_escaped_chars(self):
        result = _ensure_list('["path/to/file", "line\\nbreak"]')
        assert result == ["path/to/file", "line\nbreak"]

    def test_json_list_empty_strings(self):
        result = _ensure_list('["", ""]')
        assert result == ["", ""]

    def test_json_list_with_nested_objects(self):
        data = json.dumps([{"oldText": "a", "newText": "b"}, {"oldText": "c", "newText": "d"}])
        result = _ensure_list(data)
        assert len(result) == 2
        assert result[0]["oldText"] == "a"
        assert result[1]["newText"] == "d"
