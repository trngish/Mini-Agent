"""Unified retry handler for tool execution.

Consolidates retry logic from:
- ErrorRecoveryManager (should_retry, get_backoff_delay)
- tool_execution.is_transient_error

This provides a single, consistent retry interface for all tool executions.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


# Transient error patterns that warrant retry
TRANSIENT_PATTERNS = [
    # 网络超时
    "timeout",
    "timed out",
    "timed_out",
    # 网络连接
    "connection",
    "econreset",
    "etimedout",
    "enotfound",
    "econnrefused",
    "econnaborted",
    # 服务状态
    "temporary",
    "unavailable",
    "overloaded",
    "backpressure",
    "service overloaded",
    "busy",
    "degraded",
    # 速率限制
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttl",
    "quota exceeded",
    # 可重试标识
    "retry",
    "retry after",
    "retry_after",
    "please retry",
    "try again",
    # HTTP 状态码
    "429",
    "503",
    "502",
    "504",
    # 云服务错误
    "server error",
    "internal error",
    "maintenance",
]


class RetryHandler:
    """Unified retry handler for tool execution.

    Provides:
    - Transient error detection
    - Retry decision making
    - Exponential backoff calculation
    """

    def __init__(self, agent: "Agent", max_retries: int = 3, base_delay: float = 0.5):
        """Initialize RetryHandler.

        Args:
            agent: The agent instance (for accessing config)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        """
        self._agent = agent
        self._max_retries = max_retries
        self._base_delay = base_delay

    def should_retry(self, error: str | Exception, attempt: int) -> bool:
        """Check if an error should trigger a retry.

        Args:
            error: Error message or exception
            attempt: Current attempt number (0-indexed)

        Returns:
            True if the error is transient and retries remain
        """
        if attempt >= self._max_retries:
            return False

        error_str = str(error).lower()
        return self.is_transient_error(error_str)

    def is_transient_error(self, error: str) -> bool:
        """Check if an error is transient (worth retrying).

        Args:
            error: Error message (will be lowercased internally)

        Returns:
            True if error is transient
        """
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in TRANSIENT_PATTERNS)

    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        return float(self._base_delay * (2**attempt))

    def get_max_retries(self) -> int:
        """Get maximum retry attempts."""
        return self._max_retries


def create_retry_handler(agent: "Agent") -> RetryHandler:
    """Factory function to create a RetryHandler from agent config.

    Args:
        agent: The agent instance

    Returns:
        RetryHandler configured from agent's settings
    """
    # Try to get config from agent, fallback to defaults
    max_retries = 3
    base_delay = 0.5

    # Check for M27 config
    if hasattr(agent, "m27_config") and agent.m27_config:
        max_retries = agent.m27_config.get("max_tool_retries", 3)

    # Check for retry config on agent
    if hasattr(agent, "_retry_config"):
        max_retries = agent._retry_config.max_retries
        base_delay = agent._retry_config.initial_delay

    return RetryHandler(agent, max_retries=max_retries, base_delay=base_delay)
