"""Approval manager for tool call authorization.

Handles user approval flow for tool calls in AGENT mode,
with configurable timeout and security defaults.
"""

from __future__ import annotations

import os
import threading

from ..schema import AgentMode
from ..utils import Colors


class ApprovalManager:
    """Manages tool call approval in AGENT mode.

    In AGENT mode, each tool call requires explicit user approval.
    In YOLO mode, all calls are auto-approved.
    In PLAN mode, write operations are blocked entirely.

    Security default: if approval cannot be obtained (timeout, error),
    the tool call is REJECTED (deny by default).
    """

    DEFAULT_TIMEOUT = 10

    def __init__(self, mode: AgentMode = AgentMode.YOLO, write_tools: set[str] | frozenset[str] | None = None):
        self._mode = mode
        self._write_tools = write_tools or set()
        self._timeout = int(os.environ.get("MINI_AGENT_APPROVAL_TIMEOUT", str(self.DEFAULT_TIMEOUT)))

    @property
    def mode(self) -> AgentMode:
        return self._mode

    @mode.setter
    def mode(self, value: AgentMode) -> None:
        self._mode = value

    def is_approved(self, function_name: str) -> bool:
        """Check if a tool call is approved.

        Args:
            function_name: Name of the tool being called

        Returns:
            True if approved, False if rejected
        """
        if self._mode != AgentMode.AGENT:
            return True

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
        except Exception:
            return False

    def is_write_tool(self, function_name: str) -> bool:
        """Check if a tool is a write operation."""
        return function_name in self._write_tools
