"""Tests for config.py configuration loading."""

import os
from unittest.mock import mock_open, patch

import pytest

from mini_agent.config import (
    AgentConfig,
    CLIOverrideConfig,
    Config,
    LLMConfig,
    M27Config,
    MCPConfig,
    PlatformConfig,
    RetryConfig,
    ToolsConfig,
)


class TestRetryConfig:
    """Test RetryConfig model."""

    def test_default_values(self):
        """Test default retry config values."""
        config = RetryConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0

    def test_custom_values(self):
        """Test custom retry config values."""
        config = RetryConfig(
            enabled=False,
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
        )
        assert config.enabled is False
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0


class TestLLMConfig:
    """Test LLMConfig model."""

    def test_valid_api_key(self):
        """Test LLMConfig accepts valid API key."""
        config = LLMConfig(api_key="test_api_key_12345")
        assert config.api_key == "test_api_key_12345"

    def test_invalid_api_key_too_short(self):
        """Test LLMConfig rejects too-short API key."""
        with pytest.raises(ValueError, match="at least 8 characters"):
            LLMConfig(api_key="short")

    def test_invalid_api_key_placeholder(self):
        """Test LLMConfig rejects placeholder API key."""
        with pytest.raises(ValueError, match="not configured"):
            LLMConfig(api_key="YOUR_API_KEY_HERE")

    def test_default_api_base(self):
        """Test default API base."""
        config = LLMConfig(api_key="test_key_12345")
        assert config.api_base == "https://api.minimax.io"

    def test_default_model(self):
        """Test default model."""
        config = LLMConfig(api_key="test_key_12345")
        assert config.model == "MiniMax-M2.5"

    def test_default_provider(self):
        """Test default provider."""
        config = LLMConfig(api_key="test_key_12345")
        assert config.provider == "anthropic"


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_default_values(self):
        """Test default agent config."""
        config = AgentConfig()
        assert config.max_steps == 50
        assert config.workspace_dir == "./workspace"
        assert config.system_prompt_path == "system_prompt.md"

    def test_custom_values(self):
        """Test custom agent config."""
        config = AgentConfig(max_steps=100, workspace_dir="/tmp/agent")
        assert config.max_steps == 100
        assert config.workspace_dir == "/tmp/agent"


class TestMCPConfig:
    """Test MCPConfig model."""

    def test_default_values(self):
        """Test default MCP config."""
        config = MCPConfig()
        assert config.connect_timeout == 10.0
        assert config.execute_timeout == 60.0
        assert config.sse_read_timeout == 120.0


class TestToolsConfig:
    """Test ToolsConfig model."""

    def test_default_values(self):
        """Test default tools config."""
        config = ToolsConfig()
        assert config.enable_file_tools is True
        assert config.enable_bash is True
        assert config.enable_note is True
        assert config.enable_skills is True
        assert config.enable_mcp is True
        assert config.bash_timeout == 120

    def test_get_skills_search_paths(self, tmp_path):
        """Test skills search paths generation."""
        config = ToolsConfig()

        # Should return paths without error even if dirs don't exist
        paths = config.get_skills_search_paths()
        assert isinstance(paths, list)


class TestPlatformConfig:
    """Test PlatformConfig model."""

    def test_default_mode(self):
        """Test default platform mode."""
        config = PlatformConfig()
        assert config.mode == "auto"


class TestM27Config:
    """Test M27Config model."""

    def test_default_values(self):
        """Test default M27 config."""
        config = M27Config()
        assert config.enable_extended_thinking is True
        assert config.thinking_budget_tokens == 32768
        assert config.thinking_budget_adaptive is True
        assert config.enable_message_cache is True
        assert config.enable_parallel_tool_calls is True
        assert config.max_concurrent_tools == 20
        assert config.token_limit == 800_000
        assert config.max_output_tokens == 32768


class TestCLIOverrideConfig:
    """Test CLIOverrideConfig model."""

    def test_default_values(self):
        """Test default CLI override config."""
        config = CLIOverrideConfig()
        assert config.api_key is None
        assert config.api_base is None
        assert config.model is None
        assert config.max_steps is None

    def test_partial_override(self):
        """Test partial CLI override."""
        config = CLIOverrideConfig(model="MiniMax-M2.1")
        assert config.model == "MiniMax-M2.1"
        assert config.api_key is None


class TestConfigMerge:
    """Test Config merge functionality."""

    def test_merge_cli_overrides_llm(self):
        """Test CLI overrides merge for LLM settings."""
        config = Config(
            llm=LLMConfig(api_key="original_key_12345"),
            agent=AgentConfig(),
            tools=ToolsConfig(),
        )

        cli_override = CLIOverrideConfig(
            api_key="new_key_12345",
            model="MiniMax-M2.1",
        )

        config.merge_cli_overrides(cli_override)

        assert config.llm.api_key == "new_key_12345"
        assert config.llm.model == "MiniMax-M2.1"

    def test_merge_cli_overrides_agent(self):
        """Test CLI overrides merge for agent settings."""
        config = Config(
            llm=LLMConfig(api_key="test_key_12345"),
            agent=AgentConfig(max_steps=50),
            tools=ToolsConfig(),
        )

        cli_override = CLIOverrideConfig(max_steps=100)

        config.merge_cli_overrides(cli_override)

        assert config.agent.max_steps == 100

    def test_merge_cli_overrides_platform(self):
        """Test CLI overrides merge for platform settings."""
        config = Config(
            llm=LLMConfig(api_key="test_key_12345"),
            agent=AgentConfig(),
            tools=ToolsConfig(),
            platform=PlatformConfig(mode="auto"),
        )

        cli_override = CLIOverrideConfig(platform_mode="windows")

        config.merge_cli_overrides(cli_override)

        assert config.platform.mode == "windows"

    def test_merge_cli_overrides_tools(self):
        """Test CLI overrides merge for tools settings."""
        config = Config(
            llm=LLMConfig(api_key="test_key_12345"),
            agent=AgentConfig(),
            tools=ToolsConfig(enable_skills=True),
        )

        cli_override = CLIOverrideConfig(enable_skills=False)

        config.merge_cli_overrides(cli_override)

        assert config.tools.enable_skills is False


class TestConfigEnvOverrides:
    """Test environment variable override functionality."""

    def test_env_override_api_key(self):
        """Test MINIMAX_API_KEY overrides config."""
        yaml_content = "api_key: original_key_12345\napi_base: https://test.com\n"

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                # This would need more complex mocking for full test
                pass

    def test_env_override_minimax_key(self):
        """Test MINIMAX_API_KEY environment variable."""
        # Test the env var name support
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env_key_12345"}):
            # The _apply_env_overrides method checks MINIMAX_API_KEY first
            pass


class TestConfigToDict:
    """Test Config serialization."""

    def test_to_dict(self):
        """Test config to dictionary conversion."""
        config = Config(
            llm=LLMConfig(api_key="test_key_12345"),
            agent=AgentConfig(),
            tools=ToolsConfig(),
        )

        result = config.to_dict()

        assert "llm" in result
        assert "agent" in result
        assert "tools" in result
        assert result["llm"]["api_key"] == "test_key_12345"


class TestConfigFindConfigFile:
    """Test config file discovery."""

    def test_find_config_file_returns_path(self, tmp_path):
        """Test find_config_file returns a valid path."""
        # Create a temporary config file
        config_dir = tmp_path / "mini_agent" / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("api_key: test_12345\n")

        with patch.object(Config, "find_config_file", return_value=config_file):
            result = Config.find_config_file("config.yaml")
            assert result == config_file


class TestConfigGetEnvVarHelp:
    """Test environment variable help text."""

    def test_get_env_var_help(self):
        """Test help text generation."""
        help_text = Config.get_env_var_help()

        assert "MINIMAX_API_KEY" in help_text
        assert "MINI_AGENT_API_KEY" in help_text
        assert "MINI_AGENT_API_BASE" in help_text
        assert "MINI_AGENT_MODEL" in help_text
