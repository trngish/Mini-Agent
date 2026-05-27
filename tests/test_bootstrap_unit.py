from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.bootstrap import (
    add_workspace_tools,
    build_m27_config,
    cleanup_mcp,
    create_llm_client,
    initialize_base_tools,
)
from mini_agent.config import AgentConfig, Config, LLMConfig, M27Config, PlatformConfig, RetryConfig, ToolsConfig


def _make_config(**overrides):
    llm = overrides.pop(
        "llm", LLMConfig(api_key="sk-test-key-12345678", api_base="https://api.test.com", model="test-model")
    )
    agent = overrides.pop("agent", AgentConfig())
    tools = overrides.pop("tools", ToolsConfig())
    platform = overrides.pop("platform", PlatformConfig())
    m27 = overrides.pop("m27", M27Config())
    return Config(llm=llm, agent=agent, tools=tools, platform=platform, m27=m27)


class TestInitializeBaseTools:
    @pytest.mark.asyncio
    async def test_returns_tools_list(self):
        config = _make_config()
        tools, skill_loader = await initialize_base_tools(config)
        assert isinstance(tools, list)
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "bash" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    @pytest.mark.asyncio
    async def test_file_tools_disabled(self):
        tools_cfg = ToolsConfig(enable_file_tools=False)
        config = _make_config(tools=tools_cfg)
        tools, _ = await initialize_base_tools(config)
        tool_names = [t.name for t in tools]
        assert "read_file" not in tool_names
        assert "write_file" not in tool_names

    @pytest.mark.asyncio
    async def test_bash_tools_disabled(self):
        tools_cfg = ToolsConfig(enable_bash=False)
        config = _make_config(tools=tools_cfg)
        tools, _ = await initialize_base_tools(config)
        tool_names = [t.name for t in tools]
        assert "bash" not in tool_names

    @pytest.mark.asyncio
    async def test_note_tools_disabled(self):
        tools_cfg = ToolsConfig(enable_note=False)
        config = _make_config(tools=tools_cfg)
        tools, _ = await initialize_base_tools(config)
        tool_names = [t.name for t in tools]
        assert "session_note" not in tool_names

    @pytest.mark.asyncio
    async def test_search_tools_always_present(self):
        tools_cfg = ToolsConfig(enable_file_tools=False, enable_bash=False, enable_note=False)
        config = _make_config(tools=tools_cfg)
        tools, _ = await initialize_base_tools(config)
        tool_names = [t.name for t in tools]
        assert "grep" in tool_names
        assert "find" in tool_names
        assert "tree" in tool_names

    @pytest.mark.asyncio
    async def test_git_tools_always_present(self):
        config = _make_config()
        tools, _ = await initialize_base_tools(config)
        tool_names = [t.name for t in tools]
        assert "git" in tool_names
        assert "git_status" in tool_names

    @pytest.mark.asyncio
    async def test_with_skill_loader(self):
        mock_loader = MagicMock()
        config = _make_config()
        tools, loader = await initialize_base_tools(config, skill_loader_arg=mock_loader)
        assert isinstance(tools, list)


class TestAddWorkspaceTools:
    @pytest.mark.asyncio
    async def test_adds_nothing_when_mcp_disabled(self, tmp_path):
        tools_cfg = ToolsConfig(enable_mcp=False)
        config = _make_config(tools=tools_cfg)
        tools = []
        await add_workspace_tools(tools, config, tmp_path)
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_adds_nothing_when_no_mcp_config(self, tmp_path):
        tools_cfg = ToolsConfig(enable_mcp=True)
        config = _make_config(tools=tools_cfg)
        tools = []
        with patch("mini_agent.bootstrap.load_mcp_tools_async", new_callable=AsyncMock, return_value=[]):
            await add_workspace_tools(tools, config, tmp_path)
        assert len(tools) == 0


class TestCreateLLMClient:
    def test_creates_anthropic_client(self):
        config = _make_config()
        client = create_llm_client(config)
        assert client is not None
        assert client.api_key == "sk-test-key-12345678"

    def test_creates_openai_client(self):
        llm = LLMConfig(
            api_key="sk-test-key-12345678", api_base="https://api.test.com", model="test-model", provider="openai"
        )
        config = _make_config(llm=llm)
        client = create_llm_client(config)
        assert client is not None

    def test_creates_with_retry_callback(self):
        config = _make_config()
        callback = MagicMock()
        client = create_llm_client(config, on_retry_callback=callback)
        assert client is not None
        assert client.retry_callback is callback

    def test_no_retry_when_disabled(self):
        llm = LLMConfig(
            api_key="sk-test-key-12345678",
            api_base="https://api.test.com",
            model="test-model",
            retry=RetryConfig(enabled=False),
        )
        config = _make_config(llm=llm)
        client = create_llm_client(config)
        assert client is not None


class TestBuildM27Config:
    def test_returns_dict(self):
        config = _make_config()
        result = build_m27_config(config)
        assert isinstance(result, dict)
        assert "thinking_budget_tokens" in result
        assert "enable_extended_thinking" in result
        assert "enable_parallel_tool_calls" in result
        assert "max_concurrent_tools" in result
        assert "enable_message_cache" in result
        assert "token_limit" in result
        assert "max_output_tokens" in result

    def test_values_match_config(self):
        config = _make_config()
        result = build_m27_config(config)
        assert result["thinking_budget_tokens"] == config.m27.thinking_budget_tokens
        assert result["enable_extended_thinking"] == config.m27.enable_extended_thinking


class TestCleanupMcp:
    @pytest.mark.asyncio
    async def test_cleanup_no_error(self):
        with patch("mini_agent.bootstrap.cleanup_mcp_connections", new_callable=AsyncMock):
            await cleanup_mcp()

    @pytest.mark.asyncio
    async def test_cleanup_handles_exception(self):
        with patch(
            "mini_agent.bootstrap.cleanup_mcp_connections", new_callable=AsyncMock, side_effect=Exception("fail")
        ):
            await cleanup_mcp()
