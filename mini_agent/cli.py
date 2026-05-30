"""
Mini Agent - 交互式运行时

用法:
    mini-agent [--workspace DIR] [--task TASK]

示例:
    mini-agent                              # 交互模式（当前目录）
    mini-agent --workspace /path/to/dir     # 交互模式（指定目录）
    mini-agent --task "create a file"       # 非交互式执行任务
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime
from pathlib import Path

from .config import CLIOverrideConfig
from .utils import Colors


def on_retry(exception: Exception, attempt: int) -> None:
    """LLM重试事件的回调函数。"""
    from .retry import RetryConfig

    delay = RetryConfig().calculate_delay(attempt - 1)
    print(f"\n{Colors.YELLOW}⚠️  LLM 调用失败（尝试 {attempt}）：{exception}{Colors.RESET}")
    print(f"{Colors.DIM}   将在 {delay:.1f} 秒后重试...{Colors.RESET}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Mini Agent - AI 助手")
    parser.add_argument("--workspace", "-w", type=str, help="工作区目录路径")
    parser.add_argument("--task", "-t", type=str, help="非交互式执行指定任务")
    parser.add_argument("--api-key", type=str, help="覆盖 API 密钥")
    parser.add_argument("--api-base", type=str, help="覆盖 API 基础 URL")
    parser.add_argument("--model", type=str, help="覆盖模型名称")
    parser.add_argument("--provider", type=str, choices=["anthropic", "openai"], help="覆盖 LLM 提供商")
    parser.add_argument("--max-steps", type=int, help="覆盖最大执行步数")
    parser.add_argument("--platform", type=str, choices=["windows", "linux", "auto"], help="覆盖平台模式")
    parser.add_argument("--no-skills", action="store_true", help="禁用技能")
    parser.add_argument("--no-mcp", action="store_true", help="禁用 MCP")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="从上一个会话继续")

    subparsers = parser.add_subparsers(dest="command")
    log_parser = subparsers.add_parser("log", help="查看日志文件")
    log_parser.add_argument("filename", nargs="?", type=str, help="要读取的日志文件名")

    return parser.parse_args()


async def run_agent(
    workspace_dir: Path,
    task: str | None = None,
    cli_overrides: CLIOverrideConfig | None = None,
    continue_session: bool = False,
) -> None:
    """主代理初始化和执行。"""
    from .agent import Agent
    from .bootstrap import (
        add_workspace_tools,
        build_m27_config,
        cleanup_mcp,
        create_llm_client,
        initialize_base_tools,
    )
    from .config import Config
    from .repl import InteractiveLoop
    from .schema import AgentMode
    from .ui import print_stats

    session_start = datetime.now()

    if cli_overrides is None:
        cli_overrides = CLIOverrideConfig()

    config = Config.load()
    if cli_overrides:
        config.merge_cli_overrides(cli_overrides)

    llm_client = create_llm_client(config, on_retry_callback=on_retry)
    tools, skill_loader = await initialize_base_tools(config)
    await add_workspace_tools(tools, config, workspace_dir)

    system_prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if system_prompt_path and system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
        print(f"{Colors.GREEN}✅ 已加载系统提示（来源：{system_prompt_path}）{Colors.RESET}")
    else:
        system_prompt = "You are Mini-Agent, a versatile AI assistant."
        print(f"{Colors.DIM}⏭️  未找到系统提示文件，使用默认提示{Colors.RESET}")

    m27_config = build_m27_config(config)
    if m27_config:
        llm_client.configure_m27(m27_config)

    agent = Agent(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tools=tools,
        max_steps=config.agent.max_steps,
        workspace_dir=str(workspace_dir),
        m27_config=m27_config,
        mode=AgentMode.YOLO,
    )

    # 如果请求了继续，则从上一个会话继续
    if continue_session:
        latest_id = agent._session_manager.get_latest_session_id()
        if latest_id:
            if agent.load_session(latest_id):
                print(f"{Colors.GREEN}✅ 正在恢复会话：{latest_id}{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠️  无法加载会话：{latest_id}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠️  未找到上一个会话{Colors.RESET}")

    from .tools.team_dispatch_tool import TeamDispatchTool

    team_tool = TeamDispatchTool(
        llm_client=llm_client,
        tools=tools,
        system_prompt=system_prompt,
        m27_config=m27_config,
    )
    agent.tools["team_dispatch"] = team_tool
    agent.tool_list.append(team_tool)

    if task:
        print(
            f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET}"
            f" {Colors.DIM}正在执行任务...{Colors.RESET}\n"
        )
        agent.add_user_message(task)
        try:
            await agent.run()
        except Exception as e:
            print(f"\n{Colors.RED}❌ 错误：{e}{Colors.RESET}")
        finally:
            print_stats(agent, session_start)
            await cleanup_mcp()
        return

    repl = InteractiveLoop(agent, workspace_dir, config, skill_loader, m27_config)
    await repl.run()


def main() -> None:
    """CLI主入口点。"""
    from .config import CLIOverrideConfig
    from .ui import read_log_file, show_log_directory

    args = parse_args()

    if args.command == "log":
        if args.filename:
            read_log_file(args.filename)
        else:
            show_log_directory(open_file_manager=True)
        return

    workspace_dir = Path(args.workspace).resolve() if args.workspace else Path.cwd().resolve()

    cli_overrides = CLIOverrideConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        provider=args.provider,
        max_steps=args.max_steps,
        workspace_dir=str(workspace_dir) if args.workspace else None,
        platform_mode=args.platform,
        enable_skills=False if args.no_skills else None,
        enable_mcp=False if args.no_mcp else None,
    )

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            run_agent(
                workspace_dir, task=args.task, cli_overrides=cli_overrides, continue_session=args.continue_session
            )
        )
