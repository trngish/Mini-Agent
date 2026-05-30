"""代理和工具的引导逻辑。"""

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
    """初始化基础工具（无工作区依赖）。"""
    tools: list[Tool] = []
    skill_loader = skill_loader_arg

    if config.tools.enable_file_tools:
        tools.append(ReadTool())
        tools.append(WriteTool())
        tools.append(EditTool())
        # 批量操作工具 - 按调用计费优化：合并操作以减少API调用
        tools.append(MultiReadTool())
        tools.append(MultiEditTool())
        tools.append(WorkspaceContextTool())
        tools.append(DeepContextTool())
        print(f"{Colors.GREEN}✅ 文件工具已启用（批量操作 + 深度上下文）{Colors.RESET}")

    if config.tools.enable_bash:
        tools.append(BashTool(platform_mode=config.platform.mode, default_timeout=config.tools.bash_timeout))
        tools.append(BashOutputTool())
        tools.append(BashKillTool())
        # 批量bash工具 - 按调用计费优化：将多个独立命令合并为一次调用
        tools.append(MultiBashTool(platform_mode=config.platform.mode))
        print(f"{Colors.GREEN}✅ Bash 工具已启用（批量执行）{Colors.RESET}")

    if config.tools.enable_note:
        tools.append(SessionNoteTool())
        tools.append(RecallNoteTool())
        print(f"{Colors.GREEN}✅ 笔记工具已启用（记录 + 回忆）{Colors.RESET}")

    if config.tools.enable_skills:
        # 获取技能搜索路径（用户配置目录 + 项目目录）
        skills_search_paths = config.tools.get_skills_search_paths()
        skill_tools, skill_loader = create_skill_tools(
            config.tools.skills_dir,
            additional_search_paths=skills_search_paths,
        )
        if skill_tools:
            tools.extend(skill_tools)
            print(f"{Colors.GREEN}✅ 技能已启用（{len(skill_tools)} 个技能工具）{Colors.RESET}")
        else:
            print(f"{Colors.DIM}⏭️  技能已启用但未找到技能配置文件{Colors.RESET}")

    tools.append(GrepTool())
    tools.append(FindTool())
    tools.append(TreeTool())
    # 批量搜索工具 - 按调用计费优化：将多个搜索合并为一次调用
    tools.append(MultiGrepTool())
    print(f"{Colors.GREEN}✅ 搜索工具已启用（批量搜索）{Colors.RESET}")

    tools.append(GitTool())
    tools.append(GitStatusTool())
    print(f"{Colors.GREEN}✅ Git 工具已启用{Colors.RESET}")

    return tools, skill_loader


async def add_workspace_tools(tools: list[Tool], config: Config, workspace_dir: Path) -> None:
    """添加工具区依赖工具。"""
    if config.tools.enable_mcp:
        print(f"{Colors.DIM}⏳ 正在加载 MCP 工具...{Colors.RESET}")

        # 从用户目录和项目目录获取MCP配置路径
        mcp_config_paths = config.tools.get_mcp_config_paths()

        # 如果没有找到路径，回退到旧版行为
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
                print(f"{Colors.DIM}  正在加载 MCP 配置: {mcp_config_path}{Colors.RESET}")
                mcp_tools = await load_mcp_tools_async(str(mcp_config_path))
                if mcp_tools:
                    all_mcp_tools.extend(mcp_tools)

            if all_mcp_tools:
                tools.extend(all_mcp_tools)
                print(
                    f"{Colors.GREEN}✅ MCP 工具已启用 "
                    f"（{len(all_mcp_tools)} 个工具，"
                    f"来自 {len(mcp_config_paths)} 个配置）{Colors.RESET}"
                )
            else:
                print(f"{Colors.BRIGHT_YELLOW}⚠️  MCP 已启用但未加载任何工具{Colors.RESET}")
        else:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  未在任何位置找到 MCP 配置：{Colors.RESET}")
            print(f"{Colors.DIM}  - ~/.mini-agent/config/mcp.json（用户目录）{Colors.RESET}")
            print(f"{Colors.DIM}  - ./mcp.json（项目目录）{Colors.RESET}")


async def cleanup_mcp() -> None:
    """静默清理MCP连接。"""
    try:
        await cleanup_mcp_connections()
    except Exception:
        logger.debug("MCP cleanup completed (with warnings)")


def create_llm_client(config: Config, on_retry_callback: Any = None) -> LLMClient:
    """创建并配置LLM客户端。"""
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
            f"{Colors.GREEN}✅ LLM 重试机制已启用（最多 {config.llm.retry.max_retries} 次重试）{Colors.RESET}"
        )

    return client


def build_m27_config(config: Config) -> dict[str, Any]:
    """构建M2.7配置字典。"""
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
