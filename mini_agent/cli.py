"""
Mini Agent - Interactive Runtime

Usage:
    mini-agent [--workspace DIR] [--task TASK]

Examples:
    mini-agent                              # Interactive mode (current directory)
    mini-agent --workspace /path/to/dir     # Interactive mode (specific directory)
    mini-agent --task "create a file"       # Execute a task non-interactively
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
    """Callback for LLM retry events."""
    from .retry import RetryConfig

    delay = RetryConfig().calculate_delay(attempt - 1)
    print(f"\n{Colors.YELLOW}⚠️  LLM call failed (attempt {attempt}): {exception}{Colors.RESET}")
    print(f"{Colors.DIM}   Retrying in {delay:.1f}s...{Colors.RESET}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mini Agent - AI-powered assistant")
    parser.add_argument("--workspace", "-w", type=str, help="Workspace directory path")
    parser.add_argument("--task", "-t", type=str, help="Execute a specific task non-interactively")
    parser.add_argument("--api-key", type=str, help="Override API key")
    parser.add_argument("--api-base", type=str, help="Override API base URL")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--provider", type=str, choices=["anthropic", "openai"], help="Override LLM provider")
    parser.add_argument("--max-steps", type=int, help="Override max execution steps")
    parser.add_argument("--platform", type=str, choices=["windows", "linux", "auto"], help="Override platform mode")
    parser.add_argument("--no-skills", action="store_true", help="Disable skills")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue from last session")

    subparsers = parser.add_subparsers(dest="command")
    log_parser = subparsers.add_parser("log", help="View log files")
    log_parser.add_argument("filename", nargs="?", type=str, help="Log filename to read")

    return parser.parse_args()


async def run_agent(
    workspace_dir: Path,
    task: str | None = None,
    cli_overrides: CLIOverrideConfig | None = None,
    continue_session: bool = False,
) -> None:
    """Main agent initialization and execution."""
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
        print(f"{Colors.GREEN}✅ Loaded system prompt (from: {system_prompt_path}){Colors.RESET}")
    else:
        system_prompt = "You are Mini-Agent, a versatile AI assistant."
        print(f"{Colors.DIM}⏭️  No system prompt file found, using default{Colors.RESET}")

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

    # Continue from last session if requested
    if continue_session:
        latest_id = agent._session_manager.get_latest_session_id()
        if latest_id:
            if agent.load_session(latest_id):
                print(f"{Colors.GREEN}✅ Resuming session: {latest_id}{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠️  Failed to load session: {latest_id}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠️  No previous session found{Colors.RESET}")

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
            f" {Colors.DIM}Executing task...{Colors.RESET}\n"
        )
        agent.add_user_message(task)
        try:
            await agent.run()
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
        finally:
            print_stats(agent, session_start)
            await cleanup_mcp()
        return

    repl = InteractiveLoop(agent, workspace_dir, config, skill_loader, m27_config)
    await repl.run()


def main() -> None:
    """Main entry point for CLI."""
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
        asyncio.run(run_agent(workspace_dir, task=args.task, cli_overrides=cli_overrides, continue_session=args.continue_session))
