from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mini_agent.tools.multi_edit import MultiEditTool
from mini_agent.tools.multi_edit import _ensure_list as edit_ensure_list
from mini_agent.tools.multi_grep import MultiGrepTool
from mini_agent.tools.multi_grep import _ensure_list as grep_ensure_list
from mini_agent.tools.multi_read import MultiReadTool
from mini_agent.tools.multi_read import _ensure_list as read_ensure_list


class TestMultiEditToolCreateNewFile:
    async def test_create_new_file(self, tmp_path):
        tool = MultiEditTool(str(tmp_path))
        new_file = str(tmp_path / "new_file.txt")
        result = await tool.execute(edits=[{"path": new_file, "old_str": "", "new_str": "hello world"}])
        assert result.success is True
        assert Path(new_file).read_text(encoding="utf-8") == "hello world"
        assert "Created new file" in result.content

    async def test_create_new_file_in_subdirectory(self, tmp_path):
        tool = MultiEditTool(str(tmp_path))
        new_file = str(tmp_path / "sub" / "dir" / "deep.txt")
        result = await tool.execute(edits=[{"path": new_file, "old_str": "", "new_str": "deep content"}])
        assert result.success is True
        assert Path(new_file).read_text(encoding="utf-8") == "deep content"


class TestMultiEditToolEditExisting:
    async def test_edit_existing_file(self, tmp_path):
        f = tmp_path / "edit_me.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(edits=[{"path": str(f), "old_str": "hello", "new_str": "goodbye"}])
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "goodbye world"
        assert "Applied successfully" in result.content

    async def test_edit_replaces_only_first_occurrence(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("aaa bbb aaa", encoding="utf-8")
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(edits=[{"path": str(f), "old_str": "aaa", "new_str": "zzz"}])
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "zzz bbb aaa"


class TestMultiEditToolTextNotFound:
    async def test_text_not_found(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(edits=[{"path": str(f), "old_str": "not_here", "new_str": "x"}])
        assert result.success is False
        assert "Text not found" in result.content


class TestMultiEditToolMissingPath:
    async def test_missing_path(self, tmp_path):
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(edits=[{"path": "", "old_str": "x", "new_str": "y"}])
        assert result.success is False
        assert "Missing path" in result.content

    async def test_edit_nonexistent_file(self, tmp_path):
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(edits=[{"path": str(tmp_path / "nope.txt"), "old_str": "x", "new_str": "y"}])
        assert result.success is False
        assert "File not found" in result.content


class TestMultiEditToolUnicodeError:
    async def test_unicode_decode_error(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x80\x81\x82\xff\xfe")
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(edits=[{"path": str(f), "old_str": "x", "new_str": "y"}])
        assert result.success is False
        assert "Failed to decode" in result.content


class TestMultiEditToolMultipleEdits:
    async def test_multiple_edits(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha", encoding="utf-8")
        f2.write_text("beta", encoding="utf-8")
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(
            edits=[
                {"path": str(f1), "old_str": "alpha", "new_str": "ALPHA"},
                {"path": str(f2), "old_str": "beta", "new_str": "BETA"},
            ]
        )
        assert result.success is True
        assert f1.read_text(encoding="utf-8") == "ALPHA"
        assert f2.read_text(encoding="utf-8") == "BETA"
        assert "2 succeeded" in result.content

    async def test_mixed_success_and_failure(self, tmp_path):
        f1 = tmp_path / "good.txt"
        f1.write_text("hello", encoding="utf-8")
        tool = MultiEditTool(str(tmp_path))
        result = await tool.execute(
            edits=[
                {"path": str(f1), "old_str": "hello", "new_str": "world"},
                {"path": str(tmp_path / "missing.txt"), "old_str": "x", "new_str": "y"},
            ]
        )
        assert result.success is False
        assert "1 succeeded" in result.content
        assert "1 failed" in result.content


class TestEditEnsureListHelper:
    def test_none_returns_empty_list(self):
        assert edit_ensure_list(None) == []

    def test_list_returns_list(self):
        data = [{"path": "a"}]
        assert edit_ensure_list(data) is data

    def test_json_string_parses(self):
        data = '[{"path": "a"}]'
        result = edit_ensure_list(data)
        assert isinstance(result, list)
        assert result[0]["path"] == "a"

    def test_invalid_json_returns_wrapped(self):
        result = edit_ensure_list("not json")
        assert result == ["not json"]

    def test_non_list_json_returns_wrapped(self):
        result = edit_ensure_list('{"key": "val"}')
        assert result == ['{"key": "val"}']


class TestMultiGrepToolTextSearch:
    async def test_text_search(self, tmp_path):
        (tmp_path / "search.txt").write_text("findme here\nother line", encoding="utf-8")
        tool = MultiGrepTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_grep.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_grep.should_compress_result", return_value=False),
        ):
            mock_cache.return_value = MagicMock(
                get_grep_result=MagicMock(return_value=None),
                set_grep_result=MagicMock(),
            )
            result = await tool.execute(searches=[{"pattern": "findme", "path": str(tmp_path)}])
        assert result.success is True
        assert "findme" in result.content


class TestMultiGrepToolRegex:
    async def test_regex_search(self, tmp_path):
        (tmp_path / "code.py").write_text("def hello():\nclass World:", encoding="utf-8")
        tool = MultiGrepTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_grep.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_grep.should_compress_result", return_value=False),
        ):
            mock_cache.return_value = MagicMock(
                get_grep_result=MagicMock(return_value=None),
                set_grep_result=MagicMock(),
            )
            result = await tool.execute(searches=[{"pattern": r"^class\s+\w+", "path": str(tmp_path), "regex": True}])
        assert result.success is True
        assert "class World" in result.content


class TestMultiGrepToolCaseSensitive:
    async def test_case_sensitive_search(self, tmp_path):
        (tmp_path / "mixed.txt").write_text("Hello World\nhello world", encoding="utf-8")
        tool = MultiGrepTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_grep.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_grep.should_compress_result", return_value=False),
        ):
            mock_cache.return_value = MagicMock(
                get_grep_result=MagicMock(return_value=None),
                set_grep_result=MagicMock(),
            )
            result = await tool.execute(searches=[{"pattern": "Hello", "path": str(tmp_path), "case_sensitive": True}])
        assert result.success is True
        assert "Hello World" in result.content


class TestMultiGrepToolCachedResults:
    async def test_cached_results_returned(self, tmp_path):
        tool = MultiGrepTool(str(tmp_path))
        cached_matches = ["  file.txt:1| cached line"]
        mock_cache = MagicMock()
        mock_cache.get_grep_result.return_value = cached_matches
        with (
            patch("mini_agent.tools.multi_grep.get_context_cache", return_value=mock_cache),
            patch("mini_agent.tools.multi_grep.should_compress_result", return_value=False),
        ):
            result = await tool.execute(searches=[{"pattern": "cached_term", "path": str(tmp_path)}])
        assert result.success is True
        assert "cached" in result.content
        mock_cache.get_grep_result.assert_called_once()


class TestMultiGrepToolNonexistentPath:
    async def test_nonexistent_path(self, tmp_path):
        tool = MultiGrepTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_grep.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_grep.should_compress_result", return_value=False),
        ):
            mock_cache.return_value = MagicMock(
                get_grep_result=MagicMock(return_value=None),
                set_grep_result=MagicMock(),
            )
            result = await tool.execute(searches=[{"pattern": "anything", "path": str(tmp_path / "no_such_dir")}])
        assert result.success is True
        assert "Path not found" in result.content


class TestMultiGrepToolInvalidRegex:
    async def test_invalid_regex(self, tmp_path):
        (tmp_path / "dummy.txt").write_text("text", encoding="utf-8")
        tool = MultiGrepTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_grep.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_grep.should_compress_result", return_value=False),
        ):
            mock_cache.return_value = MagicMock(
                get_grep_result=MagicMock(return_value=None),
                set_grep_result=MagicMock(),
            )
            result = await tool.execute(searches=[{"pattern": "[invalid(", "path": str(tmp_path), "regex": True}])
        assert result.success is True
        assert "Invalid regex" in result.content


class TestGrepEnsureListHelper:
    def test_none_returns_empty_list(self):
        assert grep_ensure_list(None) == []

    def test_list_returns_list(self):
        data = [{"pattern": "a"}]
        assert grep_ensure_list(data) is data

    def test_json_string_parses(self):
        data = '[{"pattern": "a"}]'
        result = grep_ensure_list(data)
        assert isinstance(result, list)
        assert result[0]["pattern"] == "a"

    def test_invalid_json_returns_wrapped(self):
        result = grep_ensure_list("not json")
        assert result == ["not json"]


class TestMultiReadToolExistingFiles:
    async def test_read_existing_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content a", encoding="utf-8")
        f2.write_text("content b", encoding="utf-8")
        tool = MultiReadTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_read.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_read.get_file_token_limit", return_value=64000),
            patch("mini_agent.tools.multi_read.truncate_text_by_tokens", side_effect=lambda t, m: t),
        ):
            mock_cache.return_value = MagicMock(get_file_content=MagicMock(return_value=None))
            result = await tool.execute(paths=[str(f1), str(f2)])
        assert result.success is True
        assert "content a" in result.content
        assert "content b" in result.content


class TestMultiReadToolNonexistentFile:
    async def test_nonexistent_file(self, tmp_path):
        tool = MultiReadTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_read.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_read.get_file_token_limit", return_value=64000),
            patch("mini_agent.tools.multi_read.truncate_text_by_tokens", side_effect=lambda t, m: t),
        ):
            mock_cache.return_value = MagicMock(get_file_content=MagicMock(return_value=None))
            result = await tool.execute(paths=[str(tmp_path / "nope.txt")])
        assert result.success is False
        assert "File not found" in result.content


class TestMultiReadToolOffsetLimit:
    async def test_offset_and_limit(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 11)), encoding="utf-8")
        tool = MultiReadTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_read.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_read.get_file_token_limit", return_value=64000),
            patch("mini_agent.tools.multi_read.truncate_text_by_tokens", side_effect=lambda t, m: t),
        ):
            mock_cache.return_value = MagicMock(get_file_content=MagicMock(return_value=None))
            result = await tool.execute(paths=[str(f)], offset=3, limit=2)
        assert result.success is True
        assert "line 3" in result.content
        assert "line 4" in result.content
        assert "line 1" not in result.content
        assert "line 5" not in result.content

    async def test_limit_with_truncation_notice(self, tmp_path):
        f = tmp_path / "long.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8")
        tool = MultiReadTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_read.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_read.get_file_token_limit", return_value=64000),
            patch("mini_agent.tools.multi_read.truncate_text_by_tokens", side_effect=lambda t, m: t),
        ):
            mock_cache.return_value = MagicMock(get_file_content=MagicMock(return_value=None))
            result = await tool.execute(paths=[str(f)], limit=5)
        assert result.success is True
        assert "total lines" in result.content


class TestMultiReadToolUnicodeError:
    async def test_unicode_decode_error(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x80\x81\x82\xff\xfe")
        tool = MultiReadTool(str(tmp_path))
        with (
            patch("mini_agent.tools.multi_read.get_context_cache") as mock_cache,
            patch("mini_agent.tools.multi_read.get_file_token_limit", return_value=64000),
            patch("mini_agent.tools.multi_read.truncate_text_by_tokens", side_effect=lambda t, m: t),
        ):
            mock_cache.return_value = MagicMock(get_file_content=MagicMock(return_value=None))
            result = await tool.execute(paths=[str(f)])
        assert result.success is False
        assert "Decode error" in result.content


class TestReadEnsureListHelper:
    def test_none_returns_empty_list(self):
        assert read_ensure_list(None) == []

    def test_list_returns_list(self):
        data = ["a.txt", "b.txt"]
        assert read_ensure_list(data) is data

    def test_json_string_parses(self):
        data = '["a.txt", "b.txt"]'
        result = read_ensure_list(data)
        assert isinstance(result, list)
        assert result == ["a.txt", "b.txt"]

    def test_invalid_json_returns_wrapped(self):
        result = read_ensure_list("not json")
        assert result == ["not json"]

    def test_non_list_json_returns_wrapped(self):
        result = read_ensure_list('{"key": "val"}')
        assert result == ['{"key": "val"}']
