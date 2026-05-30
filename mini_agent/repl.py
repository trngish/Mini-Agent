"""Interactive REPL loop for Mini-Agent.

Extracted from cli.py to separate concerns:
- cli.py: argument parsing, initialization
- repl.py: interactive input loop, ESC cancellation, command dispatch
"""

from __future__ import annotations

import asyncio
import platform
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from .agent import Agent
from .bootstrap import cleanup_mcp
from .schema import AgentMode
from .subagent import SubAgent
from .ui import print_banner, print_help, print_session_info, print_stats, read_log_file, show_log_directory
from .utils import Colors
from .utils.task_state import get_task_manager


class InteractiveLoop:
    """Interactive REPL loop for agent conversation.

    Manages user input, ESC cancellation, command dispatch,
    and the prompt_toolkit session lifecycle.
    """

    def __init__(
        self,
        agent: Agent,
        workspace_dir: Path,
        config: Any,
        skill_loader: Any = None,
        m27_config: dict[str, Any] | None = None,
    ):
        self.agent = agent
        self.workspace_dir = workspace_dir
        self.config = config
        self.skill_loader = skill_loader
        self.m27_config = m27_config or {}
        self.session_start = datetime.now()

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-u")
        def clear_line(event: Any) -> None:
            event.current_buffer.reset()

        @kb.add("c-l")
        def clear_screen(event: Any) -> None:
            event.app.renderer.clear()

        @kb.add("c-j")
        def insert_newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("tab")
        def cycle_mode(event: Any) -> None:
            modes = [AgentMode.PLAN, AgentMode.AGENT, AgentMode.YOLO]
            self.agent.mode = modes[(modes.index(self.agent.mode) + 1) % len(modes)]
            print(f"\n  {Colors.BOLD}{Colors.GREEN}Mode: {self.agent.mode.value.upper()}{Colors.RESET}")

        return kb

    def _build_prompt_session(self) -> PromptSession[str]:
        kb = self._build_key_bindings()
        history_file = Path.home() / ".mini-agent" / ".history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        return PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(
                [
                    "/help",
                    "/clear",
                    "/history",
                    "/stats",
                    "/log",
                    "/mode",
                    "/save",
                    "/load",
                    "/list",
                    "/subagent",
                    "/skills",
                    "/brainstorm",
                    "/plan",
                    "/task",
                    "/status",
                    "/debug",
                    "/exit",
                ],
                ignore_case=True,
            ),
            style=Style.from_dict({"prompt": "#00ff00 bold", "separator": "#666666"}),
            key_bindings=kb,
        )

    def _start_esc_listener(self, cancel_event: asyncio.Event) -> tuple[threading.Event, list[bool], threading.Thread]:
        esc_stop = threading.Event()
        esc_cancelled = [False]

        def _listen() -> None:
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
                    import select
                    import sys as _sys
                    import termios
                    import tty

                    fd = _sys.stdin.fileno()
                    old = termios.tcgetattr(fd)  # type: ignore[attr-defined]
                    try:
                        tty.setcbreak(fd)  # type: ignore[attr-defined]
                        while not esc_stop.is_set():
                            if select.select([_sys.stdin], [], [], 0.05)[0] and _sys.stdin.read(1) == "\x1b":
                                esc_cancelled[0] = True
                                cancel_event.set()
                                break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)  # type: ignore[attr-defined]
                except Exception:
                    pass

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()
        return esc_stop, esc_cancelled, thread

    async def _dispatch_command(self, cmd: str, user_input: str) -> bool:
        """Dispatch a /command. Returns False if should exit."""
        parts = cmd.split()
        if parts[0] in ("/exit", "/quit", "/q"):
            print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye!{Colors.RESET}\n")
            print_stats(self.agent, self.session_start)
            return False

        if parts[0] == "/help":
            print_help()
        elif parts[0] == "/clear":
            self.agent.replace_messages([self.agent.messages[0]])
            print(f"{Colors.GREEN}✅ Cleared, starting new session{Colors.RESET}\n")
        elif parts[0] == "/history":
            print(f"\n{Colors.BRIGHT_CYAN}Messages: {len(self.agent.messages)}{Colors.RESET}\n")
        elif parts[0] == "/stats":
            print_stats(self.agent, self.session_start)
        elif parts[0] == "/log" or parts[0].startswith("/log"):
            log_parts = user_input.split(maxsplit=1)
            if len(log_parts) == 1:
                show_log_directory(open_file_manager=True)
            else:
                read_log_file(log_parts[1].strip("\"'"))
        elif parts[0] == "/mode" and len(parts) > 1:
            m = {"plan": AgentMode.PLAN, "agent": AgentMode.AGENT, "yolo": AgentMode.YOLO}
            if parts[1] in m:
                self.agent.mode = m[parts[1]]
                print(f"{Colors.GREEN}✅ Mode: {parts[1].upper()}{Colors.RESET}\n")
        elif parts[0] == "/save":
            sid = self.agent.save_session(" ".join(parts[1:]))
            print(f"{Colors.GREEN}✅ Saved: {sid}{Colors.RESET}\n")
        elif parts[0] == "/load" and len(parts) > 1:
            self.agent.load_session(parts[1])
        elif parts[0] == "/list":
            for s in self.agent.list_sessions()[:10]:
                print(f"  {s['id']}  {s['created'][:19]}  {s.get('label', '')}")
            print()
        elif parts[0] == "/subagent" and len(parts) > 1:
            task = " ".join(parts[1:])
            print(f"\n  {Colors.BRIGHT_YELLOW}⚡ Sub-agent: {task[:60]}{Colors.RESET}")
            try:
                r = await SubAgent(
                    llm_client=self.agent.llm,
                    tools=list(self.agent.tools.values()),
                    m27_config=self.m27_config,
                ).run(task)
                print(
                    f"  {Colors.GREEN}✓ ({r.elapsed:.1f}s): {r.content[:200]}{Colors.RESET}\n"
                    if r.success
                    else f"  {Colors.RED}✗ {r.error[:200]}{Colors.RESET}\n"
                )
            except Exception as e:
                print(f"  {Colors.RED}✗ {e}{Colors.RESET}\n")
        elif parts[0] == "/skills":
            if self.skill_loader:
                skills = self.skill_loader.list_skills()
                print(f"\n{Colors.BRIGHT_CYAN}Available Skills ({len(skills)}):{Colors.RESET}")
                for name in sorted(skills):
                    skill = self.skill_loader.get_skill(name)
                    desc = skill.description[:60] if skill else ""
                    print(f"  {Colors.GREEN}{name}{Colors.RESET} - {desc}...")
                print()
            else:
                print(f"{Colors.DIM}⏭️  Skills not loaded{Colors.RESET}\n")
        elif parts[0] == "/brainstorm":
            if self.skill_loader and self.skill_loader.get_skill("brainstorming"):
                print(f"\n{Colors.BRIGHT_CYAN}🧠 Brainstorming Skill{Colors.RESET}")
                print(f"  {Colors.DIM}Use this skill before any creative work - creating features,")
                print(f"  building components, adding functionality, or modifying behavior.{Colors.RESET}")
                print("\n  To start brainstorming, simply describe what you want to build!")
                print(f"  {Colors.DIM}(The agent will invoke the brainstorming skill automatically){Colors.RESET}\n")
            else:
                print(f"{Colors.RED}❌ Brainstorming skill not found{Colors.RESET}\n")
        elif parts[0] == "/plan":
            if self.skill_loader and self.skill_loader.get_skill("writing-plans"):
                print(f"\n{Colors.BRIGHT_CYAN}📋 Writing Plans Skill{Colors.RESET}")
                print(f"  {Colors.DIM}Use when you have a spec or requirements for a multi-step task.{Colors.RESET}")
                print("\n  Workflow: brainstorm → writing-plans → executing-plans → finishing-a-development-branch")
                print(f"  {Colors.DIM}(Start with /brainstorm to define the design first){Colors.RESET}\n")
            else:
                print(f"{Colors.RED}❌ Writing plans skill not found{Colors.RESET}\n")
        elif parts[0] == "/task":
            task_mgr = get_task_manager()
            if len(parts) > 1 and parts[1] == "start":
                description = " ".join(parts[2:]) if len(parts) > 2 else "unnamed"
                task_id = description[:50] or "unnamed"
                task_mgr.start_task(task_id, description, max_steps=self.agent.max_steps)
                print(f"{Colors.GREEN}✅ Task started: {task_id}{Colors.RESET}\n")
            elif len(parts) > 1 and parts[1] == "end":
                task_mgr.end_task()
                print(f"{Colors.GREEN}✅ Task ended{Colors.RESET}\n")
            elif len(parts) > 1 and parts[1] == "cancel":
                task_mgr.cancel_task()
                print(f"{Colors.YELLOW}⚠️ Task cancelled{Colors.RESET}\n")
            else:
                print(f"\n{task_mgr.get_status_report()}\n")
        elif parts[0] == "/status":
            task_mgr = get_task_manager()
            print(f"\n{task_mgr.get_status_report()}\n")
        elif parts[0] == "/debug":
            print(f"\n  Log: {self.agent.logger.get_log_file_path()}\n")
        else:
            print(f"{Colors.RED}❌ Unknown: {user_input}{Colors.RESET}\n")

        return True

    async def _process_user_message(self, user_input: str) -> None:
        """Process a normal (non-command) user message."""
        print(
            f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET}"
            f" {Colors.DIM}Thinking... (Esc to cancel){Colors.RESET}\n"
        )
        self.agent.add_user_message(user_input)

        cancel_event = asyncio.Event()
        self.agent.cancel_event = cancel_event
        esc_stop, esc_cancelled, esc_thread = self._start_esc_listener(cancel_event)

        try:
            agent_task = asyncio.create_task(self.agent.run())
            while not agent_task.done():
                if esc_cancelled[0]:
                    cancel_event.set()
                await asyncio.sleep(0.1)
            agent_task.result()
        except asyncio.CancelledError:
            print(f"\n{Colors.BRIGHT_YELLOW}⚠️  Cancelled{Colors.RESET}")
        finally:
            self.agent.cancel_event = None
            esc_stop.set()
            esc_thread.join(timeout=0.2)

    async def run(self) -> None:
        """Start the interactive REPL loop."""
        print_banner()
        print_session_info(self.agent, self.workspace_dir, self.config.llm.model)

        session = self._build_prompt_session()

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
                    should_continue = await self._dispatch_command(user_input.lower().split()[0], user_input)
                    if not should_continue:
                        break
                else:
                    await self._process_user_message(user_input)

                print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

            except KeyboardInterrupt:
                print(f"\n\n{Colors.BRIGHT_YELLOW}👋 Interrupt, exiting...{Colors.RESET}\n")
                print_stats(self.agent, self.session_start)
                break
            except Exception as e:
                print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
                print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

        await cleanup_mcp()
