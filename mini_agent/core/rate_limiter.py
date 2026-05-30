"""工具执行速率限制器，防止资源耗尽。

提供按工具和全局的速率限制，支持可配置的阈值。
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class RateLimiter:
    """线程安全的工具执行速率限制器。

    支持按工具和全局的速率限制，使用滑动窗口算法。当超过限制时，
    调用方会收到明确的错误消息，说明触发了哪个限制以及何时重试。
    """

    DEFAULT_GLOBAL_LIMIT = 100
    DEFAULT_GLOBAL_WINDOW = 60
    DEFAULT_PER_TOOL_LIMIT = 30
    DEFAULT_PER_TOOL_WINDOW = 60
    DEFAULT_INPUT_MAX_LENGTH = 1_000_000

    def __init__(
        self,
        global_limit: int = DEFAULT_GLOBAL_LIMIT,
        global_window: int = DEFAULT_GLOBAL_WINDOW,
        per_tool_limit: int = DEFAULT_PER_TOOL_LIMIT,
        per_tool_window: int = DEFAULT_PER_TOOL_WINDOW,
        input_max_length: int = DEFAULT_INPUT_MAX_LENGTH,
    ):
        self._global_limit = global_limit
        self._global_window = global_window
        self._per_tool_limit = per_tool_limit
        self._per_tool_window = per_tool_window
        self._input_max_length = input_max_length
        self._global_timestamps: list[float] = []
        self._tool_timestamps: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check_rate(self, tool_name: str) -> tuple[bool, str]:
        """检查工具调用是否在速率限制范围内。

        Args:
            tool_name: 被调用工具的名称

        Returns:
            元组 (allowed, message)。如果 allowed 为 False，message
            包含原因和重试时间提示。
        """
        now = time.monotonic()

        with self._lock:
            self._global_timestamps = [t for t in self._global_timestamps if now - t < self._global_window]
            self._tool_timestamps[tool_name] = [
                t for t in self._tool_timestamps[tool_name] if now - t < self._per_tool_window
            ]

            if len(self._global_timestamps) >= self._global_limit:
                oldest = self._global_timestamps[0]
                retry_after = int(self._global_window - (now - oldest)) + 1
                return False, (
                    f"Global rate limit exceeded: "
                    f"{self._global_limit} calls per {self._global_window}s. "
                    f"Retry after {retry_after}s."
                )

            if len(self._tool_timestamps[tool_name]) >= self._per_tool_limit:
                oldest = self._tool_timestamps[tool_name][0]
                retry_after = int(self._per_tool_window - (now - oldest)) + 1
                return False, (
                    f"Rate limit for '{tool_name}' exceeded: "
                    f"{self._per_tool_limit} calls per {self._per_tool_window}s. "
                    f"Retry after {retry_after}s."
                )

            self._global_timestamps.append(now)
            self._tool_timestamps[tool_name].append(now)

        return True, ""

    def validate_input_length(self, tool_name: str, arguments: dict[str, object]) -> tuple[bool, str]:
        """验证工具调用参数是否超过大小限制。

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数

        Returns:
            元组 (valid, message)。如果 valid 为 False，message
            包含哪个参数超过了限制。
        """
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > self._input_max_length:
                return False, (
                    f"Input too long for '{tool_name}.{key}': "
                    f"{len(value)} chars exceeds limit of {self._input_max_length}."
                )
        return True, ""

    def reset(self) -> None:
        """重置所有速率限制计数器。"""
        with self._lock:
            self._global_timestamps.clear()
            self._tool_timestamps.clear()
