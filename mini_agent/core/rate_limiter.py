"""Rate limiter for tool execution to prevent resource exhaustion.

Provides per-tool and global rate limiting with configurable thresholds.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class RateLimiter:
    """Thread-safe rate limiter for tool execution.

    Supports both per-tool and global rate limits using a sliding window
    algorithm. When a limit is exceeded, the caller receives a clear
    error message indicating which limit was hit and when to retry.
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
        """Check if a tool call is within rate limits.

        Args:
            tool_name: Name of the tool being called

        Returns:
            Tuple of (allowed, message). If allowed is False, message
            contains the reason and retry-after hint.
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
        """Validate that tool call arguments don't exceed size limits.

        Args:
            tool_name: Name of the tool
            arguments: Tool call arguments

        Returns:
            Tuple of (valid, message). If valid is False, message
            contains which argument exceeded the limit.
        """
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > self._input_max_length:
                return False, (
                    f"Input too long for '{tool_name}.{key}': "
                    f"{len(value)} chars exceeds limit of {self._input_max_length}."
                )
        return True, ""

    def reset(self) -> None:
        """Reset all rate limit counters."""
        with self._lock:
            self._global_timestamps.clear()
            self._tool_timestamps.clear()
