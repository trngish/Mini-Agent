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
import platform
import sys
import threading
from datetime import datetime
from pathlib import Path

from .utils import Colors


def on_retry(exception: Exception, attempt: int) -> None:
    """Callback for LLM retry events."""
    from .retry import RetryConfig
    retry_config = RetryConfig()
    delay = retry_config.calculate_delay(attempt - 1)
    print(f"\n{Colors.YELLOW}⚠️  LLM call failed (attempt {attempt}): {exception}{Colors.RESET}")
    print(f"{Colors.DIM}   Retrying in {delay:.1f}s...{Colors.RESET}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    from .schema import AgentMode

    parser = argparse.ArgumentParser(description="Mini Agent - AI-powered assistant")
    parser.add_argument("--workspace", "-w", type=str, help="Workspace directory path")
    parser.add_argument("--task", "-t", type=str, help="Execute a specific task non-interactively")

    # Configuration overrides
    parser.add_argument("--api-key", type=str, help="Override API key")
    parser.add_argument("--api-base", type=str, help="Override API base URL")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--provider", type=str, choices=["anthropic", "openai"], help="Override LLM provider")

    # Agent overrides
    parser.add_argument("--max-steps", type=int, help="Override max execution steps")
    parser.add_argument("--platform", type=str, choices=["windows", "linux", "auto"], help="Override platform mode")
    parser.add_argument("--no-skills", action="store_true", help="Disable skills")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    log_parser = subparsers.add_parser("log", help="View log files")
    log_parser.add_argument("filename", nargs="?", type=str, help="Log filename to read")

    return parser.parse_args()


async def run_agent(workspace_dir: Path, task: str = None, cli_overrides: None = None):
    """Main agent initialization and execution."""
    from .agent import Agent
    from .bootstrap import (
        build_m27_config,
        cleanup_mcp,
        create_llm_client,
        initialize_base_tools,
        add_workspace_tools,
    )
    from .config import Config, CLIOverrideConfig
    from .schema import AgentMode
    from .subagent import SubAgent
    from .ui import (
        print_banner,
        print_help,
        print_session_info,
        print_stats,
        read_log_file,
        show_log_directory,
    )
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    session_start = datetime.now()

    if cli_overrides is None:
        cli_overrides = CLIOverrideConfig()

    config = Config.load()

    # Apply CLI overrides if provided
    if cli_overrides:
        config.merge_cli_overrides(cli_overrides)

    # 2. Create LLM client
    llm_client = create_llm_client(config, on_retry_callback=on_retry)

    # 3. Initialize tools
    tools, skill_loader = await initialize_base_tools(config)
    await add_workspace_tools(tools, config, workspace_dir)

    # 4. Load system prompt
    system_prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if system_prompt_path and system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
        print(f"{Colors.GREEN}✅ Loaded system prompt (from: {system_prompt_path}){Colors.RESET}")
    else:
        system_prompt = "You are Mini-Agent, a versatile AI assistant."
        print(f"{Colors.DIM}⏭️  No system prompt file found, using default{Colors.RESET}")

    # 5. Prepare M2.7 config
    m27_config = build_m27_config(config)

    if m27_config:
        llm_client.configure_m27(m27_config)

    # 6. Create Agent
    agent = Agent(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tools=tools,
        max_steps=config.agent.max_steps,
        workspace_dir=str(workspace_dir),
        m27_config=m27_config,
        mode=AgentMode.YOLO,
    )

    # 6.5 Add team dispatch tool for multi-agent collaboration
    from .tools.team_dispatch_tool import TeamDispatchTool
    team_tool = TeamDispatchTool(
        llm_client=llm_client,
        tools=tools,
        system_prompt=system_prompt,
        m27_config=m27_config,
    )
    agent.tools["team_dispatch"] = team_tool
    agent.tool_list.append(team_tool)

    # 7. Interactive mode (original prompt_toolkit loop)
    if task:
        task_text = task
    else:
        task_text = None

    if task_text:
        print(f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Executing task...{Colors.RESET}\n")
        agent.add_user_message(task_text)
        try:
            await agent.run()
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
        finally:
            print_stats(agent, session_start)
            await cleanup_mcp()
        return

    # Interactive loop
    print_banner()
    print_session_info(agent, workspace_dir, config.llm.model)

    kb = KeyBindings()

    @kb.add("c-u")
    def _(event):
        event.current_buffer.reset()

    @kb.add("c-l")
    def _(event):
        event.app.renderer.clear()

    @kb.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    @kb.add("tab")
    def _(event):
        modes = [AgentMode.PLAN, AgentMode.AGENT, AgentMode.YOLO]
        agent.mode = modes[(modes.index(agent.mode) + 1) % len(modes)]
        print(f"\n  {Colors.BOLD}{Colors.GREEN}Mode: {agent.mode.value.upper()}{Colors.RESET}")

    history_file = Path.home() / ".mini-agent" / ".history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(
            ["/help", "/clear", "/history", "/stats", "/log", "/mode", "/save", "/load", "/list", "/subagent", "/skills", "/brainstorm", "/plan", "/debug", "/exit"],
            ignore_case=True,
        ),
        style=Style.from_dict({"prompt": "#00ff00 bold", "separator": "#666666"}),
        key_bindings=kb,
    )

    while True:
        try:
            user_input = await session.prompt_async(
                [("class:prompt", "You"), ("", " › ")],
                enable_history_search=True,
            )
            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                if cmd[0] in ("/exit", "/quit", "/q"):
                    print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye!{Colors.RESET}\n")
                    print_stats(agent, session_start)
                    break
                elif cmd[0] == "/help":
                    print_help()
                elif cmd[0] == "/clear":
                    agent.messages = [agent.messages[0]]
                    print(f"{Colors.GREEN}✅ Cleared, starting new session{Colors.RESET}\n")
                elif cmd[0] == "/history":
                    print(f"\n{Colors.BRIGHT_CYAN}Messages: {len(agent.messages)}{Colors.RESET}\n")
                elif cmd[0] == "/stats":
                    print_stats(agent, session_start)
                elif cmd[0] == "/log" or cmd[0].startswith("/log "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        show_log_directory(open_file_manager=True)
                    else:
                        read_log_file(parts[1].strip("\"'"))
                elif cmd[0] == "/mode" and len(cmd) > 1:
                    m = {"plan": AgentMode.PLAN, "agent": AgentMode.AGENT, "yolo": AgentMode.YOLO}
                    if cmd[1] in m:
                        agent.mode = m[cmd[1]]
                        print(f"{Colors.GREEN}✅ Mode: {cmd[1].upper()}{Colors.RESET}\n")
                elif cmd[0] == "/save":
                    sid = agent.save_session(" ".join(cmd[1:]))
                    print(f"{Colors.GREEN}✅ Saved: {sid}{Colors.RESET}\n")
                elif cmd[0] == "/load" and len(cmd) > 1:
                    agent.load_session(cmd[1])
                elif cmd[0] == "/list":
                    for s in agent.list_sessions()[:10]:
                        print(f"  {s['id']}  {s['created'][:19]}  {s.get('label','')}")
                    print()
                elif cmd[0] == "/subagent" and len(cmd) > 1:
                    task = " ".join(cmd[1:])
                    print(f"\n  {Colors.BRIGHT_YELLOW}⚡ Sub-agent: {task[:60]}{Colors.RESET}")
                    try:
                        # Use agent's m27_config (built at line 115)
                        r = await SubAgent(llm_client=agent.llm, tools=list(agent.tools.values()), m27_config=m27_config).run(task)
                        print(f"  {Colors.GREEN}✓ ({r.elapsed:.1f}s): {r.content[:200]}{Colors.RESET}\n" if r.success else f"  {Colors.RED}✗ {r.error[:200]}{Colors.RESET}\n")
                    except Exception as e:
                        print(f"  {Colors.RED}✗ {e}{Colors.RESET}\n")
                elif cmd[0] == "/skills":
                    if skill_loader:
                        skills = skill_loader.list_skills()
                        print(f"\n{Colors.BRIGHT_CYAN}Available Skills ({len(skills)}):{Colors.RESET}")
                        for name in sorted(skills):
                            skill = skill_loader.get_skill(name)
                            desc = skill.description[:60] if skill else ""
                            print(f"  {Colors.GREEN}{name}{Colors.RESET} - {desc}...")
                        print()
                    else:
                        print(f"{Colors.DIM}⏭️  Skills not loaded{Colors.RESET}\n")
                elif cmd[0] == "/brainstorm":
                    if skill_loader:
                        skill = skill_loader.get_skill("brainstorming")
                        if skill:
                            print(f"\n{Colors.BRIGHT_CYAN}🧠 Brainstorming Skill{Colors.RESET}")
                            print(f"  {Colors.DIM}Use this skill before any creative work - creating features,")
                            print(f"  building components, adding functionality, or modifying behavior.{Colors.RESET}")
                            print(f"\n  To start brainstorming, simply describe what you want to build!")
                            print(f"  {Colors.DIM}(The agent will invoke the brainstorming skill automatically){Colors.RESET}\n")
                        else:
                            print(f"{Colors.RED}❌ Brainstorming skill not found{Colors.RESET}\n")
                    else:
                        print(f"{Colors.DIM}⏭️  Skills not loaded{Colors.RESET}\n")
                elif cmd[0] == "/plan":
                    if skill_loader:
                        skill = skill_loader.get_skill("writing-plans")
                        if skill:
                            print(f"\n{Colors.BRIGHT_CYAN}📋 Writing Plans Skill{Colors.RESET}")
                            print(f"  {Colors.DIM}Use when you have a spec or requirements for a multi-step task.{Colors.RESET}")
                            print(f"\n  Workflow: brainstorm → writing-plans → executing-plans → finishing-a-development-branch")
                            print(f"  {Colors.DIM}(Start with /brainstorm to define the design first){Colors.RESET}\n")
                        else:
                            print(f"{Colors.RED}❌ Writing plans skill not found{Colors.RESET}\n")
                    else:
                        print(f"{Colors.DIM}⏭️  Skills not loaded{Colors.RESET}\n")
                elif cmd[0] == "/debug":
                    print(f"\n  Log: {agent.logger.get_log_file_path()}\n")
                else:
                    print(f"{Colors.RED}❌ Unknown: {user_input}{Colors.RESET}\n")
                continue

            # ── Normal message ──
            print(f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Thinking... (Esc to cancel){Colors.RESET}\n")
            agent.add_user_message(user_input)

            cancel_event = asyncio.Event()
            agent.cancel_event = cancel_event
            esc_stop = threading.Event()
            esc_cancelled = [False]

            def esc_listener() -> None:
                if platform.system() == "Windows":
                    try:
                        import ctypes
                        while not esc_stop.is_set():
                            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                                esc_cancelled[0] = True
                                cancel_event.set()
                                break
                            esc_stop.wait(0.05)
                    except Exception:
                        pass
                else:
                    try:
                        import select, termios, tty
                        fd = sys.stdin.fileno()
                        old = termios.tcgetattr(fd)
                        try:
                            tty.setcbreak(fd)
                            while not esc_stop.is_set():
                                if select.select([sys.stdin], [], [], 0.05)[0]:
                                    if sys.stdin.read(1) == "\x1b":
                                        esc_cancelled[0] = True
                                        cancel_event.set()
                                        break
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    except Exception:
                        pass

            esc_thread = threading.Thread(target=esc_listener, daemon=True)
            esc_thread.start()

            try:
                agent_task = asyncio.create_task(agent.run())
                while not agent_task.done():
                    if esc_cancelled[0]:
                        cancel_event.set()
                    await asyncio.sleep(0.1)
                agent_task.result()
            except asyncio.CancelledError:
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  Cancelled{Colors.RESET}")
            finally:
                agent.cancel_event = None
                esc_stop.set()
                esc_thread.join(timeout=0.2)

            print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.BRIGHT_YELLOW}👋 Interrupt, exiting...{Colors.RESET}\n")
            print_stats(agent, session_start)
            break
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
            print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

    await cleanup_mcp()


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

    try:
        asyncio.run(run_agent(workspace_dir, task=args.task, cli_overrides=cli_overrides))
    except KeyboardInterrupt:
        pass
