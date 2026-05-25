"""Error recovery and pattern learning for agent self-improvement.

Provides error tracking, pattern analysis, and actionable suggestions.
"""

import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..agent import Agent


class ErrorPattern:
    """Represents a detected error pattern."""

    def __init__(self, tool_name: str, count: int, recent_errors: list[dict]):
        self.tool_name = tool_name
        self.count = count
        self.recent_errors = recent_errors

    @property
    def severity(self) -> str:
        """Get severity level based on count."""
        if self.count >= 5:
            return "critical"
        elif self.count >= 3:
            return "warning"
        return "info"


class ErrorRecoveryManager:
    """Manages error patterns, recovery strategies, and suggestions.

    Features:
    - Track errors by tool name
    - Keep recent error history for analysis
    - Provide actionable suggestions based on patterns
    - Configurable recovery strategies
    """

    MAX_ERROR_HISTORY = 20
    MAX_CONSECUTIVE_FAILURES = 3

    # Recovery strategies
    STRATEGIES = {
        "retry": {"max_attempts": 3, "backoff": "exponential"},
        "fallback": {"use_alternative_tool": True},
        "skip": {"log_and_continue": True},
    }

    def __init__(self, agent: "Agent"):
        self._agent = agent
        self._error_patterns: dict[str, int] = {}
        self._error_history: list[dict] = []
        self._consecutive_failures = 0

    def record_error(self, error: str, context: str) -> None:
        """Record an error for pattern learning.

        Args:
            error: Error message
            context: Context where error occurred (e.g., "tool_name(args)")
        """
        # Extract tool name from context
        tool_name = context.split('(')[0] if '(' in context else "unknown"

        # Track by tool name
        self._error_patterns[tool_name] = self._error_patterns.get(tool_name, 0) + 1

        # Keep error history
        self._error_history.append({
            "error": error[:200],
            "context": context[:100],
            "tool": tool_name,
        })
        if len(self._error_history) > self.MAX_ERROR_HISTORY:
            self._error_history = self._error_history[-self.MAX_ERROR_HISTORY:]

        # Also record via note tool if available
        self._agent.record_context(
            content=f"Error with {tool_name}: {error[:150]}",
            category="error_pattern"
        )

    def record_success(self) -> None:
        """Record a successful operation (reset consecutive failures)."""
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._consecutive_failures += 1

    def get_patterns(self) -> dict:
        """Get error pattern analysis.

        Returns:
            Dict with error patterns by tool and recent error history
        """
        return {
            "error_counts_by_tool": self._error_patterns.copy(),
            "recent_errors": self._error_history.copy(),
            "total_consecutive_failures": self._consecutive_failures,
        }

    def get_top_failing_tools(self, limit: int = 5) -> list[tuple[str, int]]:
        """Get top tools by failure count.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of (tool_name, count) tuples sorted by count
        """
        return sorted(
            self._error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

    def get_suggestions(self) -> list[str]:
        """Get suggestions based on current agent state.

        Analyzes agent status and provides actionable suggestions.

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Check error patterns
        if self._consecutive_failures >= 2:
            suggestions.append(f"Consider reviewing recent errors with get_error_patterns()")

        # Check token usage
        try:
            tokens = self._agent._estimate_tokens()
            if tokens > self._agent.token_limit * 0.8:
                suggestions.append("Token usage is high - consider summarizing messages earlier")
        except Exception:
            pass

        # Check for repeated tool failures
        for tool, count in self._error_patterns.items():
            if count >= 3:
                suggestions.append(f"Tool '{tool}' has failed {count} times - may need investigation")

        # Check session age
        steps = len([m for m in self._agent.messages if m.role == "user"])
        if steps > 30 and not suggestions:
            suggestions.append("Long session detected - consider saving session and starting fresh")

        return suggestions

    def get_recovery_strategy(self, tool_name: str) -> dict:
        """Get recovery strategy for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Recovery strategy dict
        """
        # Could be extended to have per-tool strategies
        return self.STRATEGIES["retry"].copy()

    def should_retry(self, tool_name: str, attempt: int) -> bool:
        """Check if should retry a failed tool.

        Args:
            tool_name: Name of the tool
            attempt: Current attempt number

        Returns:
            True if should retry
        """
        strategy = self.get_recovery_strategy(tool_name)
        max_attempts = strategy.get("max_attempts", 3)
        return attempt < max_attempts

    def get_backoff_delay(self, attempt: int, base_delay: float = 0.5) -> float:
        """Calculate backoff delay for retry.

        Args:
            attempt: Current attempt number
            base_delay: Base delay in seconds

        Returns:
            Delay in seconds
        """
        return base_delay * (2 ** attempt)

    @property
    def consecutive_failures(self) -> int:
        """Get current consecutive failure count."""
        return self._consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        """Set consecutive failure count."""
        self._consecutive_failures = value