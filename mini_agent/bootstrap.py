"""Agent and tool bootstrap logic."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from .agent import Agent
from .config import Config
from .llm import LLMClient
from .retry import RetryConfig
from .schema import LLMProvider
from .tools.base import Tool
from .tools.bash_tool import BashKillTool, BashOutputTool, BashTool
from .tools.file_tools import EditTool, ReadTool, WriteTool
from .tools.search_tools import GrepTool, FindTool, TreeTool
from .tools.git_tool import GitTool, GitStatusTool
from .tools.mcp_loader import cleanup_mcp_connections, load_mcp_tools_async, set_mcp_timeout_config
from .tools.note_tool import SessionNoteTool
from .tools.skill_tool import create_skill_tools
from .tools.skill_tool import create_skill_tools
from .utils import Colors

logger = logging.getLogger(__name__)


async def initialize_base_tools(config: Config, skill_loader_arg=None):
    """Initialize basic tools (no workspace dependency)."""
    tools: list[Tool] = []
    skill_loader = skill_loader_arg

    if config.tools.enable_file_tools:
        tools.append(ReadTool())
        tools.append(WriteTool())
        tools.append(EditTool())
        print(f"{Colors.GREEN}✅ File tools enabled{Colors.RESET}")

    if config.tools.enable_bash:
        tools.append(BashTool(platform_mode=config.platform.mode))
        tools.append(BashOutputTool())
        tools.append(BashKillTool())
        print(f"{Colors.GREEN}✅ Bash tools enabled{Colors.RESET}")

    if config.tools.enable_note:
        tools.append(SessionNoteTool())
        print(f"{Colors.GREEN}✅ Note tool enabled{Colors.RESET}")

    if config.tools.enable_skills:
        skill_tools, skill_loader = create_skill_tools(config.tools.skills_dir)
        if skill_tools:
            tools.extend(skill_tools)
            print(f"{Colors.GREEN}✅ Skills enabled ({len(skill_tools)} skill tools){Colors.RESET}")
        else:
            print(f"{Colors.DIM}⏭️  Skills enabled but no skill configs found{Colors.RESET}")

    tools.append(GrepTool())
    tools.append(FindTool())
    tools.append(TreeTool())
    print(f"{Colors.GREEN}✅ Search tools enabled{Colors.RESET}")

    tools.append(GitTool())
    tools.append(GitStatusTool())
    print(f"{Colors.GREEN}✅ Git tools enabled{Colors.RESET}")

    return tools, skill_loader


def add_workspace_tools(tools: list, config: Config, workspace_dir: Path):
    """Add workspace-dependent tools."""
    if config.tools.enable_mcp:
        print(f"{Colors.DIM}⏳ Loading MCP tools...{Colors.RESET}")
        mcp_config_path = workspace_dir / config.tools.mcp_config_path
        if not mcp_config_path.exists():
            alt_path = Path(config.tools.mcp_config_path)
            if alt_path.exists():
                mcp_config_path = alt_path

        if mcp_config_path.exists():
            set_mcp_timeout_config(
                connect_timeout=config.tools.mcp.connect_timeout,
                execute_timeout=config.tools.mcp.execute_timeout,
                sse_read_timeout=config.tools.mcp.sse_read_timeout,
            )
            mcp_tools = asyncio.run(load_mcp_tools_async(str(mcp_config_path)))
            if mcp_tools:
                tools.extend(mcp_tools)
                print(f"{Colors.GREEN}✅ MCP tools enabled ({len(mcp_tools)} tools){Colors.RESET}")
            else:
                print(f"{Colors.BRIGHT_YELLOW}⚠️  MCP enabled but no tools loaded{Colors.RESET}")
        else:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  MCP config not found: {mcp_config_path}{Colors.RESET}")


async def cleanup_mcp():
    """Clean up MCP connections quietly."""
    try:
        await cleanup_mcp_connections()
    except Exception:
        logger.debug("MCP cleanup completed (with warnings)")


def create_llm_client(config: Config, on_retry_callback=None) -> LLMClient:
    """Create and configure an LLM client."""
    provider = LLMProvider.ANTHROPIC if config.llm.provider.lower() == "anthropic" else LLMProvider.OPENAI

    retry_config = None
    if config.llm.retry.enabled:
        retry_config = RetryConfig(
            enabled=True,
            max_retries=config.llm.retry.max_retries,
            initial_delay=config.llm.retry.initial_delay,
            max_delay=config.llm.retry.max_delay,
            exponential_base=config.llm.retry.exponential_base,
        )

    client = LLMClient(
        api_key=config.llm.api_key,
        provider=provider,
        api_base=config.llm.api_base,
        model=config.llm.model,
        retry_config=retry_config,
    )

    if retry_config and on_retry_callback:
        client.retry_callback = on_retry_callback
        print(f"{Colors.GREEN}✅ LLM retry mechanism enabled (max {config.llm.retry.max_retries} retries){Colors.RESET}")

    return client


def build_m27_config(config: Config) -> dict:
    """Build M2.7 configuration dict."""
    if not hasattr(config, 'm27'):
        return {}
    return {
        "enable_extended_thinking": config.m27.enable_extended_thinking,
        "thinking_budget_tokens": config.m27.thinking_budget_tokens,
        "enable_parallel_tool_calls": config.m27.enable_parallel_tool_calls,
        "max_concurrent_tools": config.m27.max_concurrent_tools,
        "enable_message_cache": config.m27.enable_message_cache,
        "token_limit": config.m27.token_limit,
        "max_output_tokens": config.m27.max_output_tokens,
    }
