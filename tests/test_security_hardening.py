import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.schema import AgentMode, FunctionCall, ToolCall
from mini_agent.tools.file_tools import _is_path_blacklisted, _resolve_and_validate_path
from mini_agent.utils.command_validator import (
    DangerLevel,
    assess_command_danger,
    sanitize_file_path,
    validate_command_safety,
)


class TestPathBlacklist:
    def test_is_path_blacklisted_windows_system_dir(self):
        if os.name == "nt":
            result, reason = _is_path_blacklisted(Path("C:\\Windows\\System32\\config"))
            assert result is True
            assert reason

    def test_is_path_blacklisted_unix_etc(self):
        if os.name != "nt":
            result, reason = _is_path_blacklisted(Path("/etc/passwd"))
            assert result is True
            assert reason

    def test_is_path_blacklisted_home_ssh(self):
        home_ssh = Path.home() / ".ssh" / "id_rsa"
        result, reason = _is_path_blacklisted(home_ssh)
        assert result is True

    def test_is_path_blacklisted_home_gnupg(self):
        home_gnupg = Path.home() / ".gnupg" / "private-keys"
        result, reason = _is_path_blacklisted(home_gnupg)
        assert result is True

    def test_is_path_blacklisted_allowed_path(self):
        result, reason = _is_path_blacklisted(Path("/home/user/project/file.py"))
        assert result is False
        assert reason == ""

    def test_resolve_and_validate_path_blocks_blacklisted_absolute(self):
        if os.name == "nt":
            with pytest.raises(ValueError, match="blacklisted"):
                _resolve_and_validate_path("C:\\Windows\\System32\\config", Path("C:\\workspace"))
        else:
            with pytest.raises(ValueError, match="blacklisted"):
                _resolve_and_validate_path("/etc/passwd", Path("/workspace"))

    def test_resolve_and_validate_path_allows_normal_absolute(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        result = _resolve_and_validate_path(str(test_file), Path("/workspace"))
        assert result == test_file.resolve()

    def test_resolve_and_validate_path_blocks_workspace_escape(self):
        with pytest.raises(ValueError, match="escapes workspace"):
            _resolve_and_validate_path("../../etc/passwd", Path("/workspace"))


class TestSanitizeFilePath:
    def test_removes_null_bytes(self):
        result = sanitize_file_path("file\x00name.txt")
        assert "\x00" not in result

    def test_removes_shell_metacharacters(self):
        result = sanitize_file_path("file;rm.txt")
        assert ";" not in result

    def test_resolves_directory_traversal(self):
        result = sanitize_file_path("../../../etc/passwd")
        assert ".." not in result or "etc" not in result

    def test_handles_empty_path(self):
        result = sanitize_file_path("")
        assert isinstance(result, str)


class TestCommandSafety:
    def test_blocked_rm_rf_root(self):
        level, reason = assess_command_danger("rm -rf /")
        assert level == DangerLevel.BLOCKED

    def test_blocked_fork_bomb(self):
        level, reason = assess_command_danger(":(){ :|:& };:")
        assert level == DangerLevel.BLOCKED

    def test_caution_sudo(self):
        level, reason = assess_command_danger("sudo apt install something")
        assert level == DangerLevel.CAUTION

    def test_safe_git(self):
        level, reason = assess_command_danger("git status")
        assert level == DangerLevel.SAFE

    def test_validate_command_safety_empty(self):
        is_safe, msg, level = validate_command_safety("")
        assert is_safe is False
        assert level == DangerLevel.BLOCKED

    def test_validate_command_safety_too_long(self):
        is_safe, msg, level = validate_command_safety("x" * 10001)
        assert is_safe is False
        assert level == DangerLevel.BLOCKED


class TestAgentSecurityFixes:
    def test_check_approved_defaults_to_reject_on_exception(self):
        from mini_agent.agent import Agent
        from mini_agent.llm import LLMClient

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model = "test-model"

        agent = Agent(
            llm_client=mock_llm,
            system_prompt="test",
            tools=[],
            mode=AgentMode.AGENT,
        )

        with patch("builtins.input", side_effect=RuntimeError("no input")):
            result = agent._check_approved("test_tool")
            assert result is False

    def test_record_context_uses_running_loop(self):
        from mini_agent.agent import Agent
        from mini_agent.llm import LLMClient

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model = "test-model"

        agent = Agent(
            llm_client=mock_llm,
            system_prompt="test",
            tools=[],
        )

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no running loop")):
            agent.record_context("test context")
            assert True


class TestSubAgentSecurity:
    def test_subagent_blocks_dangerous_tools(self):
        from mini_agent.subagent import BLOCKED_TOOLS_FOR_SUBAGENT

        assert "bash_kill" in BLOCKED_TOOLS_FOR_SUBAGENT
        assert "team_dispatch" in BLOCKED_TOOLS_FOR_SUBAGENT

    @pytest.mark.asyncio
    async def test_subagent_rejects_blocked_tool(self):
        from mini_agent.llm import LLMClient
        from mini_agent.subagent import SubAgent

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model = "test-model"
        mock_llm.generate = AsyncMock()

        agent = SubAgent(llm_client=mock_llm, tools=[])

        tool_call = ToolCall(
            id="test-1",
            type="function",
            function=FunctionCall(name="bash_kill", arguments={"bash_id": "abc123"}),
        )

        tool_call_obj, tool_msg = await agent._execute_single_tool(tool_call)
        assert "blocked" in tool_msg.content.lower() or "Error" in tool_msg.content


class TestConfigValidation:
    def test_config_missing_api_key_raises_valueerror(self, tmp_path):
        from mini_agent.config import Config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test\napi_base: https://api.test.com\n")

        env_vars = {"MINIMAX_API_KEY", "MINI_AGENT_API_KEY"}
        saved = {k: os.environ.pop(k, None) for k in env_vars}
        try:
            with pytest.raises(ValueError, match="api_key"):
                Config.from_yaml(config_file)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_merge_cli_overrides_revalidates(self):
        from mini_agent.config import AgentConfig, CLIOverrideConfig, Config, LLMConfig, ToolsConfig

        config = Config(
            llm=LLMConfig(api_key="sk-test-key-12345678", api_base="https://api.test.com", model="test"),
            agent=AgentConfig(),
            tools=ToolsConfig(),
        )

        cli_overrides = CLIOverrideConfig(max_steps=200)
        config.merge_cli_overrides(cli_overrides)
        assert config.agent.max_steps == 200
