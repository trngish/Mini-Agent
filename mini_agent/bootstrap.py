"""Agent and tool bootstrap logic."""

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .llm import LLMClient
from .retry import RetryConfig
from .schema import LLMProvider
from .tools.base import Tool
from .tools.bash_tool import BashKillTool, BashOutputTool, BashTool
from .tools.deep_context import DeepContextTool
from .tools.file_tools import EditTool, ReadTool, WriteTool
from .tools.git_tool import GitStatusTool, GitTool
from .tools.mcp_loader import cleanup_mcp_connections, load_mcp_tools_async, set_mcp_timeout_config
from .tools.multi_bash import MultiBashTool
from .tools.multi_edit import MultiEditTool
from .tools.multi_grep import MultiGrepTool
from .tools.multi_read import MultiReadTool
from .tools.note_tool import RecallNoteTool, SessionNoteTool
from .tools.search_tools import FindTool, GrepTool, TreeTool
from .tools.skill_tool import create_skill_tools
from .tools.workspace_context import WorkspaceContextTool
from .utils import Colors

logger = logging.getLogger(__name__)


async def initialize_base_tools(config: Config, skill_loader_arg: Any = None) -> tuple[list[Tool], Any]:
    """Initialize basic tools (no workspace dependency)."""
    tools: list[Tool] = []
    skill_loader = skill_loader_arg

    if config.tools.enable_file_tools:
        tools.append(ReadTool())
        tools.append(WriteTool())
        tools.append(EditTool())
        # Batch operation tools - per-call billing optimization: merge operations to reduce API calls
        tools.append(MultiReadTool())
        tools.append(MultiEditTool())
        tools.append(WorkspaceContextTool())
        tools.append(DeepContextTool())
        print(f"{Colors.GREEN}✅ File tools enabled (with batch operations + deep context){Colors.RESET}")

    if config.tools.enable_bash:
        tools.append(BashTool(platform_mode=config.platform.mode, default_timeout=config.tools.bash_timeout))
        tools.append(BashOutputTool())
        tools.append(BashKillTool())
        # Batch bash tool - per-call billing optimization: merge multiple independent commands into one call
        tools.append(MultiBashTool(platform_mode=config.platform.mode))
        print(f"{Colors.GREEN}✅ Bash tools enabled (with batch execution){Colors.RESET}")

    if config.tools.enable_note:
        tools.append(SessionNoteTool())
        tools.append(RecallNoteTool())
        print(f"{Colors.GREEN}✅ Note tools enabled (record + recall){Colors.RESET}")

    if config.tools.enable_skills:
        # Get skills search paths (user config dir + project dir)
        skills_search_paths = config.tools.get_skills_search_paths()
        skill_tools, skill_loader = create_skill_tools(
            config.tools.skills_dir,
            additional_search_paths=skills_search_paths,
        )
        if skill_tools:
            tools.extend(skill_tools)
            print(f"{Colors.GREEN}✅ Skills enabled ({len(skill_tools)} skill tools){Colors.RESET}")
        else:
            print(f"{Colors.DIM}⏭️  Skills enabled but no skill configs found{Colors.RESET}")

    tools.append(GrepTool())
    tools.append(FindTool())
    tools.append(TreeTool())
    # Batch search tool - per-call billing optimization: merge multiple searches into one call
    tools.append(MultiGrepTool())
    print(f"{Colors.GREEN}✅ Search tools enabled (with batch search){Colors.RESET}")

    tools.append(GitTool())
    tools.append(GitStatusTool())
    print(f"{Colors.GREEN}✅ Git tools enabled{Colors.RESET}")

    return tools, skill_loader


async def add_workspace_tools(tools: list[Tool], config: Config, workspace_dir: Path) -> None:
    """Add workspace-dependent tools."""
    if config.tools.enable_mcp:
        print(f"{Colors.DIM}⏳ Loading MCP tools...{Colors.RESET}")

        # Get MCP config paths from user dir and project dir
        mcp_config_paths = config.tools.get_mcp_config_paths()

        # Fallback to legacy behavior if no paths found
        if not mcp_config_paths:
            mcp_config_path = workspace_dir / config.tools.mcp_config_path
            if not mcp_config_path.exists():
                alt_path = Path(config.tools.mcp_config_path)
                if alt_path.exists():
                    mcp_config_path = alt_path
            if mcp_config_path.exists():
                mcp_config_paths = [mcp_config_path]

        if mcp_config_paths:
            set_mcp_timeout_config(
                connect_timeout=config.tools.mcp.connect_timeout,
                execute_timeout=config.tools.mcp.execute_timeout,
                sse_read_timeout=config.tools.mcp.sse_read_timeout,
            )

            all_mcp_tools = []
            for mcp_config_path in mcp_config_paths:
                print(f"{Colors.DIM}  Loading MCP config: {mcp_config_path}{Colors.RESET}")
                mcp_tools = await load_mcp_tools_async(str(mcp_config_path))
                if mcp_tools:
                    all_mcp_tools.extend(mcp_tools)

            if all_mcp_tools:
                tools.extend(all_mcp_tools)
                print(
                    f"{Colors.GREEN}✅ MCP tools enabled "
                    f"({len(all_mcp_tools)} tools from "
                    f"{len(mcp_config_paths)} config(s)){Colors.RESET}"
                )
            else:
                print(f"{Colors.BRIGHT_YELLOW}⚠️  MCP enabled but no tools loaded{Colors.RESET}")
        else:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  MCP config not found in any location:{Colors.RESET}")
            print(f"{Colors.DIM}  - ~/.mini-agent/config/mcp.json (user dir){Colors.RESET}")
            print(f"{Colors.DIM}  - ./mcp.json (project dir){Colors.RESET}")


async def cleanup_mcp() -> None:
    """Clean up MCP connections quietly."""
    try:
        await cleanup_mcp_connections()
    except Exception:
        logger.debug("MCP cleanup completed (with warnings)")


def create_llm_client(config: Config, on_retry_callback: Any = None) -> LLMClient:
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
        print(
            f"{Colors.GREEN}✅ LLM retry mechanism enabled (max {config.llm.retry.max_retries} retries){Colors.RESET}"
        )

    return client


def build_m27_config(config: Config) -> dict[str, Any]:
    """Build M2.7 configuration dict."""
    if not hasattr(config, "m27"):
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
