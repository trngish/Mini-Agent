"""Comprehensive unit tests for git_tool module.

Tests cover GitTool, GitStatusTool, and the _shell_quote helper.
All subprocess/shell calls are mocked via BashTool.execute.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.tools.git_tool import GitStatusTool, GitTool, _shell_quote


class TestShellQuote:
    """Tests for the _shell_quote helper function."""

    def test_unix_simple_string(self):
        result = _shell_quote("hello", is_windows=False)
        assert result == "hello"

    def test_unix_string_with_spaces(self):
        result = _shell_quote("hello world", is_windows=False)
        assert result == "'hello world'"

    def test_unix_string_with_special_chars(self):
        result = _shell_quote("test; rm -rf /", is_windows=False)
        assert ";" not in result or result.startswith("'")

    def test_unix_empty_string(self):
        result = _shell_quote("", is_windows=False)
        assert result == "''"

    def test_windows_simple_string(self):
        result = _shell_quote("hello", is_windows=True)
        assert result == "'hello'"

    def test_windows_string_with_spaces(self):
        result = _shell_quote("hello world", is_windows=True)
        assert result == "'hello world'"

    def test_windows_string_with_single_quote(self):
        result = _shell_quote("it's", is_windows=True)
        assert result == "'it''s'"

    def test_windows_multiple_single_quotes(self):
        result = _shell_quote("it's a test's", is_windows=True)
        assert "''" in result
        assert result == "'it''s a test''s'"

    def test_windows_empty_string(self):
        result = _shell_quote("", is_windows=True)
        assert result == "''"

    def test_unix_string_with_single_quote(self):
        result = _shell_quote("it's", is_windows=False)
        assert result != "it's"


class TestGitToolProperties:
    """Tests for GitTool properties and initialization."""

    def test_name(self):
        tool = GitTool(workspace_dir="/tmp/test")
        assert tool.name == "git"

    def test_description_contains_operations(self):
        tool = GitTool(workspace_dir="/tmp/test")
        for op in ["status", "add", "commit", "log", "diff", "branch", "checkout", "pull", "push"]:
            assert op in tool.description

    def test_parameters_schema(self):
        tool = GitTool(workspace_dir="/tmp/test")
        params = tool.parameters
        assert params["type"] == "object"
        assert "operation" in params["properties"]
        assert "path" in params["properties"]
        assert "message" in params["properties"]
        assert "branch" in params["properties"]
        assert "all" in params["properties"]
        assert params["required"] == ["operation"]

    def test_parameters_operation_enum(self):
        tool = GitTool(workspace_dir="/tmp/test")
        enum = tool.parameters["properties"]["operation"]["enum"]
        expected = ["status", "add", "commit", "log", "diff", "branch", "checkout", "pull", "push"]
        assert enum == expected

    def test_parameters_defaults(self):
        tool = GitTool(workspace_dir="/tmp/test")
        props = tool.parameters["properties"]
        assert props["path"]["default"] == "."
        assert props["all"]["default"] is False

    def test_workspace_dir_resolved_to_absolute(self):
        tool = GitTool(workspace_dir=".")
        assert tool.workspace_dir.is_absolute()

    def test_workspace_dir_custom(self):
        tool = GitTool(workspace_dir="/custom/path")
        assert tool.workspace_dir == Path("/custom/path").absolute()

    def test_bash_tool_initialized(self):
        tool = GitTool(workspace_dir="/tmp/test")
        assert tool._bash is not None


def _make_bash_result(success: bool, content: str = "", error: str | None = None):
    """Helper to create a mock BashTool execute result."""
    result = MagicMock()
    result.success = success
    result.content = content
    result.error = error
    return result


class TestGitToolStatus:
    """Tests for GitTool execute with operation='status'."""

    @pytest.mark.asyncio
    async def test_status_clean_working_tree(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="status")

        assert result.success
        assert "Working tree clean" in result.content

    @pytest.mark.asyncio
    async def test_status_with_staged_changes(self):
        tool = GitTool(workspace_dir="/tmp/test")
        output = "M  file1.py\nA  file2.py\n"
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, output))

        result = await tool.execute(operation="status")

        assert result.success
        assert "Staged" in result.content
        assert "file1.py" in result.content
        assert "file2.py" in result.content

    @pytest.mark.asyncio
    async def test_status_with_unstaged_changes(self):
        tool = GitTool(workspace_dir="/tmp/test")
        output = " M file3.py\n?? new_file.py\n"
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, output))

        result = await tool.execute(operation="status")

        assert result.success
        assert "Unstaged" in result.content
        assert "file3.py" in result.content
        assert "new_file.py" in result.content

    @pytest.mark.asyncio
    async def test_status_with_staged_and_unstaged(self):
        tool = GitTool(workspace_dir="/tmp/test")
        output = "M  staged.py\n M unstaged.py\n?? untracked.py\n"
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, output))

        result = await tool.execute(operation="status")

        assert result.success
        assert "Staged" in result.content
        assert "Unstaged" in result.content
        assert "staged.py" in result.content
        assert "unstaged.py" in result.content
        assert "untracked.py" in result.content

    @pytest.mark.asyncio
    async def test_status_bash_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "not a git repo"))

        result = await tool.execute(operation="status")

        assert not result.success

    @pytest.mark.asyncio
    async def test_status_with_custom_path(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="status", path="subdir")

        assert result.success
        call_args = tool._bash.execute.call_args[0][0]
        assert "subdir" in call_args


class TestGitToolAdd:
    """Tests for GitTool execute with operation='add'."""

    @pytest.mark.asyncio
    async def test_add_specific_file(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="add", path="src/main.py")

        assert result.success
        assert "Staged" in result.content
        assert "src/main.py" in result.content
        call_args = tool._bash.execute.call_args[0][0]
        assert "git add" in call_args

    @pytest.mark.asyncio
    async def test_add_all(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="add", all=True)

        assert result.success
        call_args = tool._bash.execute.call_args[0][0]
        assert "git add -A" in call_args

    @pytest.mark.asyncio
    async def test_add_not_all_uses_path(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="add", path="file.py", all=False)

        assert result.success
        call_args = tool._bash.execute.call_args[0][0]
        assert "git add" in call_args
        assert "-A" not in call_args

    @pytest.mark.asyncio
    async def test_add_bash_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "error"))

        result = await tool.execute(operation="add", path="file.py")

        assert not result.success


class TestGitToolCommit:
    """Tests for GitTool execute with operation='commit'."""

    @pytest.mark.asyncio
    async def test_commit_with_message(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="commit", message="Fix bug")

        assert result.success
        assert "Committed" in result.content
        assert "Fix bug" in result.content
        call_args = tool._bash.execute.call_args[0][0]
        assert "git commit" in call_args
        assert "Fix bug" in call_args

    @pytest.mark.asyncio
    async def test_commit_without_message(self):
        tool = GitTool(workspace_dir="/tmp/test")

        result = await tool.execute(operation="commit")

        assert not result.success
        assert "Commit message required" in result.error

    @pytest.mark.asyncio
    async def test_commit_empty_message(self):
        tool = GitTool(workspace_dir="/tmp/test")

        result = await tool.execute(operation="commit", message="")

        assert not result.success
        assert "Commit message required" in result.error

    @pytest.mark.asyncio
    async def test_commit_bash_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "nothing to commit"))

        result = await tool.execute(operation="commit", message="test")

        assert not result.success


class TestGitToolLog:
    """Tests for GitTool execute with operation='log'."""

    @pytest.mark.asyncio
    async def test_log_with_commits(self):
        tool = GitTool(workspace_dir="/tmp/test")
        output = "abc1234 Fix bug\ndef5678 Add feature"
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, output))

        result = await tool.execute(operation="log")

        assert result.success
        assert "Recent Commits" in result.content
        assert "abc1234" in result.content

    @pytest.mark.asyncio
    async def test_log_no_commits(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="log")

        assert result.success
        assert "No commits yet" in result.content

    @pytest.mark.asyncio
    async def test_log_bash_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "error"))

        result = await tool.execute(operation="log")

        assert not result.success


class TestGitToolDiff:
    """Tests for GitTool execute with operation='diff'."""

    @pytest.mark.asyncio
    async def test_diff_with_changes(self):
        tool = GitTool(workspace_dir="/tmp/test")
        stat_result = _make_bash_result(True, "file1.py | 5 ++---\n2 files changed")
        diff_result = _make_bash_result(True, "diff --git a/file1.py b/file1.py\n+new line")

        call_count = 0

        async def mock_execute(cmd, timeout=30):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stat_result
            return diff_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute(operation="diff")

        assert result.success
        assert "Uncommitted Changes" in result.content
        assert "file1.py" in result.content

    @pytest.mark.asyncio
    async def test_diff_no_changes(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="diff")

        assert result.success
        assert "No uncommitted changes" in result.content

    @pytest.mark.asyncio
    async def test_diff_stat_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "error"))

        result = await tool.execute(operation="diff")

        assert not result.success

    @pytest.mark.asyncio
    async def test_diff_stat_success_but_diff_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        stat_result = _make_bash_result(True, "file1.py | 5 ++---")
        diff_result = _make_bash_result(False, "", "diff error")

        call_count = 0

        async def mock_execute(cmd, timeout=30):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stat_result
            return diff_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute(operation="diff")

        assert result.success
        assert "Uncommitted Changes" in result.content


class TestGitToolBranch:
    """Tests for GitTool execute with operation='branch'."""

    @pytest.mark.asyncio
    async def test_branch_list(self):
        tool = GitTool(workspace_dir="/tmp/test")
        output = "* main\n  develop\n  feature/x"
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, output))

        result = await tool.execute(operation="branch")

        assert result.success
        assert "Branches" in result.content
        assert "main" in result.content
        assert "develop" in result.content

    @pytest.mark.asyncio
    async def test_branch_bash_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "error"))

        result = await tool.execute(operation="branch")

        assert not result.success


class TestGitToolCheckout:
    """Tests for GitTool execute with operation='checkout'."""

    @pytest.mark.asyncio
    async def test_checkout_with_branch(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        result = await tool.execute(operation="checkout", branch="develop")

        assert result.success
        assert "Switched to branch" in result.content
        assert "develop" in result.content
        call_args = tool._bash.execute.call_args[0][0]
        assert "git checkout" in call_args
        assert "develop" in call_args

    @pytest.mark.asyncio
    async def test_checkout_without_branch(self):
        tool = GitTool(workspace_dir="/tmp/test")

        result = await tool.execute(operation="checkout")

        assert not result.success
        assert "Branch name required" in result.error

    @pytest.mark.asyncio
    async def test_checkout_empty_branch(self):
        tool = GitTool(workspace_dir="/tmp/test")

        result = await tool.execute(operation="checkout", branch="")

        assert not result.success
        assert "Branch name required" in result.error

    @pytest.mark.asyncio
    async def test_checkout_bash_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "branch not found"))

        result = await tool.execute(operation="checkout", branch="nonexistent")

        assert not result.success


class TestGitToolPull:
    """Tests for GitTool execute with operation='pull'."""

    @pytest.mark.asyncio
    async def test_pull_success(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, "Already up to date."))

        result = await tool.execute(operation="pull")

        assert result.success
        call_args = tool._bash.execute.call_args[0][0]
        assert "git pull" in call_args

    @pytest.mark.asyncio
    async def test_pull_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "network error"))

        result = await tool.execute(operation="pull")

        assert not result.success


class TestGitToolPush:
    """Tests for GitTool execute with operation='push'."""

    @pytest.mark.asyncio
    async def test_push_success(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, "Everything up-to-date"))

        result = await tool.execute(operation="push")

        assert result.success
        call_args = tool._bash.execute.call_args[0][0]
        assert "git push" in call_args

    @pytest.mark.asyncio
    async def test_push_failure(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(False, "", "rejected"))

        result = await tool.execute(operation="push")

        assert not result.success


class TestGitToolUnknownOperation:
    """Tests for GitTool with unknown operations."""

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        tool = GitTool(workspace_dir="/tmp/test")

        result = await tool.execute(operation="rebase")

        assert not result.success
        assert "Unknown operation" in result.error
        assert "rebase" in result.error

    @pytest.mark.asyncio
    async def test_unknown_operation_mentions_supported(self):
        tool = GitTool(workspace_dir="/tmp/test")

        result = await tool.execute(operation="stash")

        assert not result.success
        for op in ["status", "add", "commit", "log", "diff", "branch", "checkout", "pull", "push"]:
            assert op in result.error


class TestGitToolPathResolution:
    """Tests for path resolution in GitTool."""

    @pytest.mark.asyncio
    async def test_relative_path_resolved(self):
        tool = GitTool(workspace_dir="/workspace")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        with patch.object(Path, "is_absolute", return_value=False):
            await tool.execute(operation="status", path="src/app")

        call_args = tool._bash.execute.call_args[0][0]
        assert "src" in call_args or "app" in call_args

    @pytest.mark.asyncio
    async def test_absolute_path_used_directly(self):
        tool = GitTool(workspace_dir="/workspace")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        with patch.object(Path, "is_absolute", return_value=True):
            await tool.execute(operation="status", path="/absolute/path")

        call_args = tool._bash.execute.call_args[0][0]
        assert "absolute" in call_args or "path" in call_args


class TestGitToolPlatformHandling:
    """Tests for platform-specific command construction."""

    @pytest.mark.asyncio
    async def test_windows_separator_used(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        with patch("mini_agent.tools.git_tool._platform.system", return_value="Windows"):
            await tool.execute(operation="status")

        call_args = tool._bash.execute.call_args[0][0]
        assert "; " in call_args

    @pytest.mark.asyncio
    async def test_unix_separator_used(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        with patch("mini_agent.tools.git_tool._platform.system", return_value="Linux"):
            await tool.execute(operation="status")

        call_args = tool._bash.execute.call_args[0][0]
        assert " && " in call_args


class TestGitToolTimeouts:
    """Tests that correct timeouts are passed to BashTool."""

    @pytest.mark.asyncio
    async def test_status_timeout(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        await tool.execute(operation="status")

        assert tool._bash.execute.call_args[1].get("timeout") == 30 or tool._bash.execute.call_args[0][1] == 30

    @pytest.mark.asyncio
    async def test_commit_timeout(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        await tool.execute(operation="commit", message="test")

        timeout = tool._bash.execute.call_args[1].get("timeout")
        if timeout is None:
            timeout = tool._bash.execute.call_args[0][1]
        assert timeout == 60

    @pytest.mark.asyncio
    async def test_pull_timeout(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        await tool.execute(operation="pull")

        timeout = tool._bash.execute.call_args[1].get("timeout")
        if timeout is None:
            timeout = tool._bash.execute.call_args[0][1]
        assert timeout == 120

    @pytest.mark.asyncio
    async def test_push_timeout(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        await tool.execute(operation="push")

        timeout = tool._bash.execute.call_args[1].get("timeout")
        if timeout is None:
            timeout = tool._bash.execute.call_args[0][1]
        assert timeout == 120


class TestGitStatusToolProperties:
    """Tests for GitStatusTool properties and initialization."""

    def test_name(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        assert tool.name == "git_status"

    def test_description_content(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        assert "Git" in tool.description
        assert "status" in tool.description.lower()

    def test_parameters_schema(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert params["properties"]["path"]["default"] == "."

    def test_workspace_dir_resolved(self):
        tool = GitStatusTool(workspace_dir=".")
        assert tool.workspace_dir.is_absolute()


class TestGitStatusToolExecute:
    """Tests for GitStatusTool execute method."""

    @pytest.mark.asyncio
    async def test_status_clean(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        status_result = _make_bash_result(True, "")

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert result.success
        assert "Branch: main" in result.content
        assert "Working tree clean" in result.content

    @pytest.mark.asyncio
    async def test_status_with_staged_and_unstaged(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "develop")
        status_output = "M  staged.py\nA  new.py\n M modified.py\n?? untracked.py\n D deleted.py"
        status_result = _make_bash_result(True, status_output)

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert result.success
        assert "Branch: develop" in result.content
        assert "Staged" in result.content
        assert "Unstaged" in result.content
        assert "staged.py" in result.content
        assert "untracked.py" in result.content

    @pytest.mark.asyncio
    async def test_status_with_deleted_staged(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        status_output = "D  removed.py"
        status_result = _make_bash_result(True, status_output)

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert result.success
        assert "Staged" in result.content
        assert "removed.py" in result.content

    @pytest.mark.asyncio
    async def test_status_branch_unknown_on_failure(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(False, "", "error")
        status_result = _make_bash_result(True, "")

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert result.success
        assert "Branch: unknown" in result.content

    @pytest.mark.asyncio
    async def test_status_bash_failure(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        status_result = _make_bash_result(False, "", "not a git repo")

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert not result.success

    @pytest.mark.asyncio
    async def test_status_limits_output_to_20(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        staged_lines = [f"M  file_{i:02d}.py" for i in range(25)]
        status_output = "\n".join(staged_lines)
        status_result = _make_bash_result(True, status_output)

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert result.success
        assert "Staged (25)" in result.content
        assert "file_19" in result.content
        assert "file_24" not in result.content

    @pytest.mark.asyncio
    async def test_status_with_custom_path(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        status_result = _make_bash_result(True, "")

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute(path="subdir")

        assert result.success
        first_call = tool._bash.execute.call_args_list[0][0][0]
        assert "subdir" in first_call

    @pytest.mark.asyncio
    async def test_status_windows_separator(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        status_result = _make_bash_result(True, "")

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        with patch("mini_agent.tools.git_tool._platform.system", return_value="Windows"):
            await tool.execute()

        first_call = tool._bash.execute.call_args_list[0][0][0]
        assert "; " in first_call

    @pytest.mark.asyncio
    async def test_status_unix_separator(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        status_result = _make_bash_result(True, "")

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        with patch("mini_agent.tools.git_tool._platform.system", return_value="Linux"):
            await tool.execute()

        first_call = tool._bash.execute.call_args_list[0][0][0]
        assert " && " in first_call

    @pytest.mark.asyncio
    async def test_status_only_unstaged(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "feature")
        status_output = "?? new_file.py\n?? another.py"
        status_result = _make_bash_result(True, status_output)

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert result.success
        assert "Unstaged" in result.content
        assert "Staged" not in result.content

    @pytest.mark.asyncio
    async def test_status_only_staged(self):
        tool = GitStatusTool(workspace_dir="/tmp/test")
        branch_result = _make_bash_result(True, "main")
        status_output = "M  staged.py\nA  added.py"
        status_result = _make_bash_result(True, status_output)

        call_count = 0

        async def mock_execute(cmd, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return branch_result
            return status_result

        tool._bash.execute = AsyncMock(side_effect=mock_execute)

        result = await tool.execute()

        assert result.success
        assert "Staged" in result.content
        assert "Unstaged" not in result.content


class TestGitToolShellInjectionPrevention:
    """Tests that shell injection is prevented via quoting."""

    @pytest.mark.asyncio
    async def test_commit_message_injection_prevented(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        malicious_msg = "fix; rm -rf /"
        await tool.execute(operation="commit", message=malicious_msg)

        call_args = tool._bash.execute.call_args[0][0]
        assert "rm -rf" not in call_args or "'" in call_args

    @pytest.mark.asyncio
    async def test_checkout_branch_injection_prevented(self):
        tool = GitTool(workspace_dir="/tmp/test")
        tool._bash.execute = AsyncMock(return_value=_make_bash_result(True, ""))

        malicious_branch = "main; rm -rf /"
        await tool.execute(operation="checkout", branch=malicious_branch)

        call_args = tool._bash.execute.call_args[0][0]
        assert "rm -rf" not in call_args or "'" in call_args or '"' in call_args
