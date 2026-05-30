"""Mini-Agent的交互式REPL循环。

从cli.py提取以分离关注点：
- cli.py: 参数解析、初始化
- repl.py: 交互式输入循环、ESC取消、命令分发
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
    """代理对话的交互式REPL循环。

    管理用户输入、ESC取消、命令分发，
    以及prompt_toolkit会话生命周期。
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
            print(f"\n  {Colors.BOLD}{Colors.GREEN}模式：{self.agent.mode.value.upper()}{Colors.RESET}")

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
        """分发/command。如果应该退出则返回False。"""
        parts = cmd.split()
        if parts[0] in ("/exit", "/quit", "/q"):
            print(f"\n{Colors.BRIGHT_YELLOW}👋 再见！{Colors.RESET}\n")
            print_stats(self.agent, self.session_start)
            return False

        if parts[0] == "/help":
            print_help()
        elif parts[0] == "/clear":
            self.agent.replace_messages([self.agent.messages[0]])
            print(f"{Colors.GREEN}✅ 已清除，开始新对话{Colors.RESET}\n")
        elif parts[0] == "/history":
            print(f"\n{Colors.BRIGHT_CYAN}消息数：{len(self.agent.messages)}{Colors.RESET}\n")
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
                print(f"{Colors.GREEN}✅ 模式：{parts[1].upper()}{Colors.RESET}\n")
        elif parts[0] == "/save":
            sid = self.agent.save_session(" ".join(parts[1:]))
            print(f"{Colors.GREEN}✅ 已保存：{sid}{Colors.RESET}\n")
        elif parts[0] == "/load" and len(parts) > 1:
            self.agent.load_session(parts[1])
        elif parts[0] == "/list":
            for s in self.agent.list_sessions()[:10]:
                print(f"  {s['id']}  {s['created'][:19]}  {s.get('label', '')}")
            print()
        elif parts[0] == "/subagent" and len(parts) > 1:
            task = " ".join(parts[1:])
            print(f"\n  {Colors.BRIGHT_YELLOW}⚡ 子代理：{task[:60]}{Colors.RESET}")
            try:
                r = await SubAgent(
                    llm_client=self.agent.llm,
                    tools=list(self.agent.tools.values()),
                    m27_config=self.m27_config,
                ).run(task)
                print(
                    f"  {Colors.GREEN}✓ ({r.elapsed:.1f}秒): {r.content[:200]}{Colors.RESET}\n"
                    if r.success
                    else f"  {Colors.RED}✗ 错误: {r.error[:200]}{Colors.RESET}\n"
                )
            except Exception as e:
                print(f"  {Colors.RED}✗ {e}{Colors.RESET}\n")
        elif parts[0] == "/skills":
            if self.skill_loader:
                skills = self.skill_loader.list_skills()
                print(f"\n{Colors.BRIGHT_CYAN}可用技能 ({len(skills)})：{Colors.RESET}")
                for name in sorted(skills):
                    skill = self.skill_loader.get_skill(name)
                    desc = skill.description[:60] if skill else ""
                    print(f"  {Colors.GREEN}{name}{Colors.RESET} - {desc}...")
                print()
            else:
                print(f"{Colors.DIM}⏭️  技能未加载{Colors.RESET}\n")
        elif parts[0] == "/brainstorm":
            if self.skill_loader and self.skill_loader.get_skill("brainstorming"):
                print(f"\n{Colors.BRIGHT_CYAN}🧠 头脑风暴技能{Colors.RESET}")
                print(f"  {Colors.DIM}在任何创意工作之前使用此技能 - 包括创建功能、")
                print(f"  构建组件、添加功能或修改行为。{Colors.RESET}")
                print("\n  只需描述你想要构建的内容，即可开始头脑风暴！")
                print(f"  {Colors.DIM}（智能体会自动调用头脑风暴技能）{Colors.RESET}\n")
            else:
                print(f"{Colors.RED}❌ 找不到头脑风暴技能{Colors.RESET}\n")
        elif parts[0] == "/plan":
            if self.skill_loader and self.skill_loader.get_skill("writing-plans"):
                print(f"\n{Colors.BRIGHT_CYAN}📋 制定计划技能{Colors.RESET}")
                print(f"  {Colors.DIM}当你有多个步骤任务的需求或规格时使用此技能。{Colors.RESET}")
                print("\n  工作流程：头脑风暴 → 制定计划 → 执行计划 → 完成开发分支")
                print(f"  {Colors.DIM}（首先使用 /brainstorm 来定义设计）{Colors.RESET}\n")
            else:
                print(f"{Colors.RED}❌ 找不到制定计划技能{Colors.RESET}\n")
        elif parts[0] == "/task":
            task_mgr = get_task_manager()
            if len(parts) > 1 and parts[1] == "start":
                description = " ".join(parts[2:]) if len(parts) > 2 else "unnamed"
                task_id = description[:50] or "unnamed"
                task_mgr.start_task(task_id, description, max_steps=self.agent.max_steps)
                print(f"{Colors.GREEN}✅ 任务已开始：{task_id}{Colors.RESET}\n")
            elif len(parts) > 1 and parts[1] == "end":
                task_mgr.end_task()
                print(f"{Colors.GREEN}✅ 任务已结束{Colors.RESET}\n")
            elif len(parts) > 1 and parts[1] == "cancel":
                task_mgr.cancel_task()
                print(f"{Colors.YELLOW}⚠️ 任务已取消{Colors.RESET}\n")
            else:
                print(f"\n{task_mgr.get_status_report()}\n")
        elif parts[0] == "/status":
            task_mgr = get_task_manager()
            print(f"\n{task_mgr.get_status_report()}\n")
        elif parts[0] == "/debug":
            print(f"\n  日志: {self.agent.logger.get_log_file_path()}\n")
        else:
            print(f"{Colors.RED}❌ 未知命令：{user_input}{Colors.RESET}\n")

        return True

    async def _process_user_message(self, user_input: str) -> None:
        """处理普通（非命令）用户消息。"""
        print(
            f"\n{Colors.BRIGHT_BLUE}智能体{Colors.RESET} {Colors.DIM}›{Colors.RESET}"
            f" {Colors.DIM}思考中... (按 Esc 取消){Colors.RESET}\n"
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
            print(f"\n{Colors.BRIGHT_YELLOW}⚠️  已取消{Colors.RESET}")
        finally:
            self.agent.cancel_event = None
            esc_stop.set()
            esc_thread.join(timeout=0.2)

    async def run(self) -> None:
        """启动交互式REPL循环。"""
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
                print(f"\n\n{Colors.BRIGHT_YELLOW}👋 中断，退出中...{Colors.RESET}\n")
                print_stats(self.agent, self.session_start)
                break
            except Exception as e:
                print(f"\n{Colors.RED}❌ 错误：{e}{Colors.RESET}")
                print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

        await cleanup_mcp()
