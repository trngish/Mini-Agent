"""工具调用授权审批管理器。

处理 AGENT 模式下工具调用的用户审批流程，
支持可配置的超时时间和安全默认值。
线程安全的实现，带有正确的锁机制。
"""

from __future__ import annotations

import asyncio
import os
import threading
from threading import Lock

from ..schema import AgentMode
from ..utils import Colors


class ApprovalManager:
    """管理 AGENT 模式下的工具调用审批。

    在 AGENT 模式下，每个工具调用都需要用户明确审批。
    在 YOLO 模式下，所有调用都会被自动批准。
    在 PLAN 模式下，写操作会被完全阻止。

    安全默认值：如果无法获得审批（超时、错误），
    则该工具调用将被拒绝（默认拒绝）。

    线程安全：使用锁来保护并发访问。
    """

    DEFAULT_TIMEOUT = 10

    def __init__(self, mode: AgentMode = AgentMode.YOLO, write_tools: set[str] | frozenset[str] | None = None):
        self._mode = mode
        self._write_tools = write_tools or set()
        self._timeout = int(os.environ.get("MINI_AGENT_APPROVAL_TIMEOUT", str(self.DEFAULT_TIMEOUT)))
        self._lock = Lock()  # 实例级锁，用于保证线程安全

    @property
    def mode(self) -> AgentMode:
        return self._mode

    @mode.setter
    def mode(self, value: AgentMode) -> None:
        with self._lock:
            self._mode = value

    def is_approved(self, function_name: str) -> bool:
        """检查工具调用是否已批准（线程安全）。

        参数:
            function_name: 被调用工具的名称

        返回:
            True 表示已批准，False 表示被拒绝
        """
        if self._mode != AgentMode.AGENT:
            return True

        with self._lock:
            return self._get_approval_sync(function_name)

    def _get_approval_sync(self, function_name: str) -> bool:
        """同步审批检查（必须在持有锁的情况下调用）。

        参数:
            function_name: 被调用工具的名称

        返回:
            True 表示已批准，False 表示被拒绝
        """
        try:
            result: list[str | None] = [None]

            def get_input() -> None:
                result[0] = (
                    input(f"  {Colors.BRIGHT_YELLOW}Approve {function_name}? [Y/n/q]{Colors.RESET} ").strip().lower()
                )

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=self._timeout)

            if result[0] is None:
                return False
            if result[0] in ("q", "quit"):
                return False
            return result[0] not in ("n", "no")
        except (EOFError, OSError):
            return False
        except Exception:
            return False

    async def is_approved_async(self, function_name: str) -> bool:
        """使用执行器的异步审批检查版本。

        参数:
            function_name: 被调用工具的名称

        返回:
            True 表示已批准，False 表示被拒绝
        """
        if self._mode != AgentMode.AGENT:
            return True

        loop = asyncio.get_running_loop()
        # P1 修复: self._lock 是 threading.Lock，不是 Executor。
        # 使用 None 来使用默认的线程池执行器。
        return await loop.run_in_executor(None, self._get_approval_sync, function_name)

    def is_write_tool(self, function_name: str) -> bool:
        """检查工具是否为写操作。"""
        with self._lock:
            return function_name in self._write_tools

    def set_write_tools(self, write_tools: set[str] | frozenset[str]) -> None:
        """更新写工具集合（线程安全）。"""
        with self._lock:
            self._write_tools = write_tools
