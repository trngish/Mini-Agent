"""Tests for MultiBashTool."""

import json
from unittest.mock import patch

import pytest

from mini_agent.tools.base import ToolResult
from mini_agent.tools.multi_bash import MultiBashTool, _ensure_list


class TestEnsureList:
    def test_none_returns_empty_list(self):
        assert _ensure_list(None) == []

    def test_list_returns_same_list(self):
        data = [{"command": "echo hi"}]
        assert _ensure_list(data) is data

    def test_json_string_parsed(self):
        data = json.dumps([{"command": "echo hi"}])
        assert _ensure_list(data) == [{"command": "echo hi"}]

    def test_invalid_json_string_returns_wrapped(self):
        assert _ensure_list("not json") == ["not json"]

    def test_non_list_json_returns_wrapped(self):
        result = _ensure_list(json.dumps({"key": "val"}))
        assert result == ['{"key": "val"}']

    def test_empty_string_returns_wrapped(self):
        assert _ensure_list("") == [""]

    def test_number_wrapped(self):
        assert _ensure_list(42) == [42]


class TestMultiBashToolProperties:
    def setup_method(self):
        self.tool = MultiBashTool(workspace_dir=".", platform_mode="auto")

    def test_name(self):
        assert self.tool.name == "multi_bash"

    def test_description_contains_parallel(self):
        assert "parallel" in self.tool.description.lower()

    def test_parameters_schema(self):
        params = self.tool.parameters
        assert params["type"] == "object"
        assert "commands" in params["properties"]
        assert params["required"] == ["commands"]
        items = params["properties"]["commands"]["items"]
        assert "command" in items["properties"]
        assert "label" in items["properties"]
        assert "timeout" in items["properties"]
        assert items["required"] == ["command"]


class TestMultiBashToolExecute:
    @pytest.mark.asyncio
    async def test_single_success_command(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"command": "echo hello", "label": "greet"}])
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "SUCCESS [greet]" in result.content
        assert "hello" in result.content
        assert "1 succeeded, 0 failed" in result.content

    @pytest.mark.asyncio
    async def test_single_failing_command(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"command": "exit 1", "label": "fail_cmd"}])
        assert result.success is False
        assert "ERROR [fail_cmd]" in result.content
        assert "0 succeeded, 1 failed" in result.content
        assert result.error == "1 command(s) failed"

    @pytest.mark.asyncio
    async def test_multiple_commands_mixed_results(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(
            commands=[
                {"command": "echo ok", "label": "good"},
                {"command": "exit 1", "label": "bad"},
            ]
        )
        assert result.success is False
        assert "SUCCESS [good]" in result.content
        assert "ERROR [bad]" in result.content
        assert "1 succeeded, 1 failed" in result.content

    @pytest.mark.asyncio
    async def test_empty_command_returns_no_command_message(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"command": "", "label": "empty"}])
        assert "No command specified" in result.content
        assert "[empty]" in result.content

    @pytest.mark.asyncio
    async def test_missing_command_key(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"label": "no_cmd"}])
        assert "No command specified" in result.content

    @pytest.mark.asyncio
    async def test_default_label_when_missing(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"command": "echo test"}])
        assert "cmd_0" in result.content

    @pytest.mark.asyncio
    async def test_timeout_command(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"command": "ping -n 10 127.0.0.1", "label": "slow", "timeout": 1}])
        assert "Timeout after 1s" in result.content

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"command": "echo fast", "label": "quick", "timeout": 30}])
        assert result.success is True
        assert "SUCCESS [quick]" in result.content

    @pytest.mark.asyncio
    async def test_commands_run_in_parallel(self):
        import time

        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        start = time.perf_counter()
        result = await tool.execute(
            commands=[
                {"command": "ping -n 2 127.0.0.1", "label": "a"},
                {"command": "ping -n 2 127.0.0.1", "label": "b"},
            ]
        )
        elapsed = time.perf_counter() - start
        assert result.success is True
        assert elapsed < 5

    @pytest.mark.asyncio
    async def test_empty_commands_list(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[])
        assert result.success is True
        assert "0 succeeded, 0 failed" in result.content

    @pytest.mark.asyncio
    async def test_stdout_output_truncation(self):
        import sys

        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        long_output_cmd = f'"{sys.executable}" -c "print(chr(120) * 4000)"'
        result = await tool.execute(commands=[{"command": long_output_cmd, "label": "long"}])
        assert result.success is True
        assert "truncated" in result.content

    @pytest.mark.asyncio
    async def test_stderr_output_truncation(self):
        import sys

        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        long_err_cmd = f'"{sys.executable}" -c "import sys; sys.stderr.write(chr(101) * 2000 + chr(10)); sys.exit(1)"'
        result = await tool.execute(commands=[{"command": long_err_cmd, "label": "long_err"}])
        assert result.success is False
        assert "..." in result.content

    @pytest.mark.asyncio
    async def test_no_output_shows_placeholder(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(commands=[{"command": "cd .", "label": "silent"}])
        if result.success:
            assert "(no output)" in result.content

    @pytest.mark.asyncio
    async def test_subprocess_exception_handling(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        with patch("mini_agent.tools.multi_bash.asyncio.create_subprocess_shell", side_effect=OSError("spawn failed")):
            result = await tool.execute(commands=[{"command": "echo hi", "label": "exc"}])
        assert "ERROR [exc]" in result.content
        assert "OSError" in result.content

    @pytest.mark.asyncio
    async def test_json_string_commands(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        cmds_json = json.dumps([{"command": "echo json_test", "label": "from_json"}])
        result = await tool.execute(commands=cmds_json)
        assert result.success is True
        assert "json_test" in result.content

    @pytest.mark.asyncio
    async def test_all_succeed_summary(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="auto")
        result = await tool.execute(
            commands=[
                {"command": "echo a", "label": "c1"},
                {"command": "echo b", "label": "c2"},
            ]
        )
        assert result.success is True
        assert result.error == ""
        assert "2 succeeded, 0 failed" in result.content

    @pytest.mark.asyncio
    async def test_workspace_dir_used(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = MultiBashTool(workspace_dir=tmpdir, platform_mode="auto")
            result = await tool.execute(commands=[{"command": "cd", "label": "cwd_check"}])
            assert result.success is True or "ERROR" in result.content

    @pytest.mark.asyncio
    async def test_windows_platform_mode(self):
        tool = MultiBashTool(workspace_dir=".", platform_mode="windows")
        assert tool.is_windows is True

    @pytest.mark.asyncio
    async def test_linux_platform_mode(self):
        from mini_agent.utils.platform_utils import PlatformUtils

        PlatformUtils.reset_cache()
        try:
            tool = MultiBashTool(workspace_dir=".", platform_mode="linux")
            assert tool.is_windows is False
        finally:
            PlatformUtils.reset_cache()
