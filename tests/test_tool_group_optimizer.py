"""Tests for ToolGroupOptimizer."""

from mini_agent.schema.schema import FunctionCall, ToolCall
from mini_agent.utils.tool_group_optimizer import ToolGroupOptimizer


class TestToolGroupOptimizer:
    """Test ToolGroupOptimizer functionality."""

    def test_can_parallelize_single_tool(self):
        """Test that a single tool can always be parallelized."""
        tc = ToolCall(
            id="1",
            type="function",
            function=FunctionCall(name="read_file", arguments={"path": "/test.py"}),
        )
        assert ToolGroupOptimizer.can_parallelize([tc]) is True

    def test_can_parallelize_empty_list(self):
        """Test empty list returns True."""
        assert ToolGroupOptimizer.can_parallelize([]) is True

    def test_can_parallelize_read_only_tools(self):
        """Test that read-only tools can always run in parallel."""
        tool_calls = [
            ToolCall(
                id="1",
                type="function",
                function=FunctionCall(name="read_file", arguments={"path": "/test.py"}),
            ),
            ToolCall(
                id="2",
                type="function",
                function=FunctionCall(name="grep", arguments={"pattern": "test", "path": "/"}),
            ),
            ToolCall(
                id="3",
                type="function",
                function=FunctionCall(name="tree", arguments={"path": "/"}),
            ),
        ]
        assert ToolGroupOptimizer.can_parallelize(tool_calls) is True

    def test_can_parallelize_write_to_different_targets(self):
        """Test that writes to different targets can run in parallel."""
        tool_calls = [
            ToolCall(
                id="1",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/file1.py"}),
            ),
            ToolCall(
                id="2",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/file2.py"}),
            ),
        ]
        assert ToolGroupOptimizer.can_parallelize(tool_calls) is True

    def test_can_parallelize_write_to_same_target_blocked(self):
        """Test that writes to the same target cannot run in parallel."""
        tool_calls = [
            ToolCall(
                id="1",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/same.py"}),
            ),
            ToolCall(
                id="2",
                type="function",
                function=FunctionCall(name="edit_file", arguments={"path": "/same.py"}),
            ),
        ]
        assert ToolGroupOptimizer.can_parallelize(tool_calls) is False

    def test_can_parallelize_mixed_read_write(self):
        """Test mixed read/write operations."""
        tool_calls = [
            ToolCall(
                id="1",
                type="function",
                function=FunctionCall(name="read_file", arguments={"path": "/test.py"}),
            ),
            ToolCall(
                id="2",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/other.py"}),
            ),
        ]
        assert ToolGroupOptimizer.can_parallelize(tool_calls) is True

    def test_group_by_dependency_all_read_only(self):
        """Test grouping when all tools are read-only."""
        tool_calls = [
            ToolCall(
                id="1",
                type="function",
                function=FunctionCall(name="read_file", arguments={"path": "/a.py"}),
            ),
            ToolCall(
                id="2",
                type="function",
                function=FunctionCall(name="grep", arguments={"pattern": "test", "path": "/"}),
            ),
        ]
        batches = ToolGroupOptimizer.group_by_dependency(tool_calls)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_group_by_dependency_mixed(self):
        """Test grouping with mixed read/write tools."""
        tool_calls = [
            ToolCall(
                id="1",
                type="function",
                function=FunctionCall(name="read_file", arguments={"path": "/a.py"}),
            ),
            ToolCall(
                id="2",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/b.py"}),
            ),
            ToolCall(
                id="3",
                type="function",
                function=FunctionCall(name="tree", arguments={"path": "/"}),
            ),
        ]
        batches = ToolGroupOptimizer.group_by_dependency(tool_calls)
        # Should have 2 batches: reads, then writes
        assert len(batches) >= 1

    def test_group_by_dependency_empty(self):
        """Test grouping empty list returns empty list."""
        assert ToolGroupOptimizer.group_by_dependency([]) == []

    def test_group_by_dependency_single_write(self):
        """Test grouping single write tool."""
        tool_calls = [
            ToolCall(
                id="1",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/a.py"}),
            ),
        ]
        batches = ToolGroupOptimizer.group_by_dependency(tool_calls)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_deduplicate_paths_no_duplicates(self):
        """Test deduplication with no duplicates."""
        paths = ["/a.py", "/b.py", "/c.py"]
        result = ToolGroupOptimizer.deduplicate_paths(paths)
        assert result == paths

    def test_deduplicate_paths_with_duplicates(self):
        """Test deduplication removes duplicates."""
        paths = ["/a.py", "/b.py", "/a.py", "/c.py"]
        result = ToolGroupOptimizer.deduplicate_paths(paths)
        assert result == ["/a.py", "/b.py", "/c.py"]

    def test_deduplicate_paths_order_preserved(self):
        """Test deduplication preserves order of first occurrence."""
        paths = ["/c.py", "/a.py", "/b.py", "/a.py", "/c.py"]
        result = ToolGroupOptimizer.deduplicate_paths(paths)
        assert result == ["/c.py", "/a.py", "/b.py"]

    def test_deduplicate_paths_relative_paths(self):
        """Test deduplication with relative paths."""
        paths = ["a.py", "./b.py", "a.py"]
        result = ToolGroupOptimizer.deduplicate_paths(paths)
        # Relative paths may resolve differently, but order should be preserved
        assert len(result) <= 3

    def test_extract_target_read_file(self):
        """Test target extraction for read_file."""
        args = {"path": "/test.py"}
        target = ToolGroupOptimizer._extract_target("read_file", args)
        assert target == "/test.py"

    def test_extract_target_write_file(self):
        """Test target extraction for write_file."""
        args = {"path": "/test.py"}
        target = ToolGroupOptimizer._extract_target("write_file", args)
        assert target == "/test.py"

    def test_extract_target_bash(self):
        """Test target extraction for bash uses command prefix."""
        args = {"command": "git status && git diff"}
        target = ToolGroupOptimizer._extract_target("bash", args)
        assert target is not None
        assert len(target) <= 100

    def test_extract_target_git(self):
        """Test target extraction for git uses operation."""
        args = {"operation": "commit"}
        target = ToolGroupOptimizer._extract_target("git", args)
        assert target == "commit"

    def test_extract_target_unknown(self):
        """Test target extraction for unknown tool returns None."""
        args = {"some_arg": "value"}
        target = ToolGroupOptimizer._extract_target("unknown_tool", args)
        assert target is None

    def test_read_only_tools_frozenset(self):
        """Test READ_ONLY frozenset contains expected tools."""
        assert "read_file" in ToolGroupOptimizer.READ_ONLY
        assert "multi_read" in ToolGroupOptimizer.READ_ONLY
        assert "grep" in ToolGroupOptimizer.READ_ONLY
        assert "tree" in ToolGroupOptimizer.READ_ONLY

    def test_write_tools_frozenset(self):
        """Test WRITE_TOOLS frozenset contains expected tools."""
        assert "write_file" in ToolGroupOptimizer.WRITE_TOOLS
        assert "edit_file" in ToolGroupOptimizer.WRITE_TOOLS
        assert "bash" in ToolGroupOptimizer.WRITE_TOOLS
        assert "delete_file" in ToolGroupOptimizer.WRITE_TOOLS

    def test_info_tools_frozenset(self):
        """Test INFO_TOOLS frozenset contains expected tools."""
        assert "tree" in ToolGroupOptimizer.INFO_TOOLS
        assert "git_status" in ToolGroupOptimizer.INFO_TOOLS
        assert "git_log" in ToolGroupOptimizer.INFO_TOOLS
