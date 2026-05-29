"""Approval manager for tool call authorization.

Handles user approval flow for tool calls in AGENT mode,
with configurable timeout and security defaults.
Thread-safe implementation with proper locking.
"""

from __future__ import annotations

import asyncio
import os
import threading
from threading import Lock

from ..schema import AgentMode
from ..utils import Colors


class ApprovalManager:
    """Manages tool call approval in AGENT mode.

    In AGENT mode, each tool call requires explicit user approval.
    In YOLO mode, all calls are auto-approved.
    In PLAN mode, write operations are blocked entirely.

    Security default: if approval cannot be obtained (timeout, error),
    the tool call is REJECTED (deny by default).

    Thread-safe: uses Lock for concurrent access protection.
    """

    DEFAULT_TIMEOUT = 10

    def __init__(self, mode: AgentMode = AgentMode.YOLO, write_tools: set[str] | frozenset[str] | None = None):
        self._mode = mode
        self._write_tools = write_tools or set()
        self._timeout = int(os.environ.get("MINI_AGENT_APPROVAL_TIMEOUT", str(self.DEFAULT_TIMEOUT)))
        self._lock = Lock()  # Instance-level lock for thread safety

    @property
    def mode(self) -> AgentMode:
        return self._mode

    @mode.setter
    def mode(self, value: AgentMode) -> None:
        with self._lock:
            self._mode = value

    def is_approved(self, function_name: str) -> bool:
        """Check if a tool call is approved (thread-safe).

        Args:
            function_name: Name of the tool being called

        Returns:
            True if approved, False if rejected
        """
        if self._mode != AgentMode.AGENT:
            return True

        with self._lock:
            return self._get_approval_sync(function_name)

    def _get_approval_sync(self, function_name: str) -> bool:
        """Synchronous approval check (must be called with lock held).

        Args:
            function_name: Name of the tool being called

        Returns:
            True if approved, False if rejected
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
        """Async version of approval check using executor.

        Args:
            function_name: Name of the tool being called

        Returns:
            True if approved, False if rejected
        """
        if self._mode != AgentMode.AGENT:
            return True

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._lock, self._get_approval_sync, function_name)

    def is_write_tool(self, function_name: str) -> bool:
        """Check if a tool is a write operation."""
        with self._lock:
            return function_name in self._write_tools

    def set_write_tools(self, write_tools: set[str] | frozenset[str]) -> None:
        """Update the set of write tools (thread-safe)."""
        with self._lock:
            self._write_tools = write_tools