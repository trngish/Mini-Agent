"""Error recovery and pattern learning for agent self-improvement.

Provides error tracking, pattern analysis, and actionable suggestions.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_context import AgentContext


class ErrorPattern:
    """Represents a detected error pattern."""

    def __init__(self, tool_name: str, count: int, recent_errors: list[dict[str, Any]]):
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
    STRATEGIES: dict[str, dict[str, Any]] = {
        "retry": {"max_attempts": 3, "backoff": "exponential"},
        "fallback": {"use_alternative_tool": True},
        "skip": {"log_and_continue": True},
    }

    def __init__(self, context: "AgentContext"):
        self._context = context
        self._error_patterns: dict[str, int] = {}
        self._error_history: list[dict[str, Any]] = []

    @property
    def _consecutive_failures(self) -> int:
        """Get consecutive failures from AgentContext (single source of truth)."""
        return self._context.consecutive_failures if self._context else 0

    @_consecutive_failures.setter
    def _consecutive_failures(self, value: int) -> None:
        """Set consecutive failures via AgentContext."""
        if self._context:
            self._context.consecutive_failures = value

    def record_error(self, error: str, context: str) -> None:
        """Record an error for pattern learning.

        Args:
            error: Error message
            context: Context where error occurred (e.g., "tool_name(args)")
        """
        # Extract tool name from context
        tool_name = context.split("(")[0] if "(" in context else "unknown"

        # Track by tool name
        self._error_patterns[tool_name] = self._error_patterns.get(tool_name, 0) + 1

        # Keep error history
        self._error_history.append(
            {
                "error": error[:200],
                "context": context[:100],
                "tool": tool_name,
            }
        )
        if len(self._error_history) > self.MAX_ERROR_HISTORY:
            self._error_history = self._error_history[-self.MAX_ERROR_HISTORY :]

        # Note: Context recording via record_context_fn removed to break circular dependency
        # Errors are still tracked in _error_history for pattern analysis

    def record_success(self) -> None:
        """Record a successful operation (reset consecutive failures)."""
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._consecutive_failures += 1

    def get_error_patterns(self) -> dict[str, int]:
        """Get error patterns dictionary.

        Returns:
            Copy of error patterns dict mapping tool name to count
        """
        return self._error_patterns.copy()

    def get_error_history(self) -> list[dict[str, Any]]:
        """Get error history list.

        Returns:
            Copy of recent error history
        """
        return self._error_history.copy()

    def get_patterns(self) -> dict[str, Any]:
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
        return sorted(self._error_patterns.items(), key=lambda x: x[1], reverse=True)[:limit]

    def get_suggestions(self) -> list[str]:
        """Get suggestions based on current agent state.

        Analyzes agent status and provides actionable suggestions.

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Check error patterns and provide specific suggestions
        top_tools = self.get_top_failing_tools(limit=3)
        for tool, count in top_tools:
            if count >= 3:
                # Provide tool-specific suggestions
                if tool in ("read_file", "multi_read"):
                    suggestions.append(
                        f"'{tool}' failing frequently - check file paths and permissions"
                    )
                elif tool in ("edit_file", "multi_edit"):
                    suggestions.append(
                        f"'{tool}' failing frequently - verify content format and encoding"
                    )
                elif tool in ("bash", "multi_bash"):
                    suggestions.append(
                        f"'{tool}' failing frequently - review command syntax and shell availability"
                    )
                elif tool == "grep":
                    suggestions.append(
                        f"'{tool}' failing frequently - check regex syntax and search paths"
                    )
                else:
                    suggestions.append(
                        f"'{tool}' has failed {count} times - may need parameter adjustment"
                    )

        # Check for consecutive failures patterns
        if self._consecutive_failures >= 2:
            suggestions.append("Multiple consecutive failures detected - consider /debug for details")
            if self._consecutive_failures >= 3:
                suggestions.append("Critical: consecutive failures may indicate systemic issue")

        # Check token usage
        try:
            tokens = self._context.estimate_tokens()
            limit = self._context.token_limit
            if tokens > limit * 0.9:
                suggestions.append("Token usage critical - consider summarizing or saving session")
            elif tokens > limit * 0.8:
                suggestions.append("Token usage high - plan for summarization soon")
        except Exception:
            pass

        # Check session age
        steps = len([m for m in self._context.get_messages() if m.role == "user"])
        if steps > 30:
            suggestions.append("Long session detected - consider /save and starting fresh")
        elif steps > 20 and not suggestions:
            suggestions.append("Extended session - save with /save periodically")

        # Check for specific error patterns in history
        error_history = self.get_error_history()
        if len(error_history) >= 5:
            # Analyze recent errors for patterns
            recent_contexts = [e.get("context", "") for e in error_history[-5:]]
            # Check for common patterns
            if any("permission" in ctx.lower() for ctx in recent_contexts):
                suggestions.append("Permission errors detected - check file/directory access rights")
            if any("not found" in ctx.lower() or "no such" in ctx.lower() for ctx in recent_contexts):
                suggestions.append("Path errors detected - verify file and directory paths")
            if any("timeout" in ctx.lower() for ctx in recent_contexts):
                suggestions.append("Timeout errors detected - consider increasing timeout settings")

        return suggestions

    def get_recovery_strategy(self, _tool_name: str) -> dict[str, Any]:  # noqa: ARG002
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
        max_attempts = int(strategy.get("max_attempts", 3))
        return attempt < max_attempts

    def get_backoff_delay(self, attempt: int, base_delay: float = 0.5) -> float:
        """Calculate backoff delay for retry.

        Args:
            attempt: Current attempt number
            base_delay: Base delay in seconds

        Returns:
            Delay in seconds
        """
        return float(base_delay * (2**attempt))

    @property
    def consecutive_failures(self) -> int:
        """Get current consecutive failure count."""
        return self._consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        """Set consecutive failure count."""
        self._consecutive_failures = value
