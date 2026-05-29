"""Health check and diagnostics for agent self-monitoring.

Provides self-health checks after each step to detect and warn about issues.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_context import AgentContext


class HealthCheckResult:
    """Result of a health check."""

    def __init__(self, issues: list[str]):
        self.issues = issues

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any("critical" in issue.lower() or "high" in issue.lower() for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.issues) > 0


class HealthChecker:
    """Self-health checker for agent state.

    Performs health checks after each step to detect:
    - Token usage critical/high
    - Multiple consecutive tool failures
    - Message history consistency issues
    """

    # Thresholds
    TOKEN_WARNING_THRESHOLD = 0.8  # 80% of limit
    TOKEN_CRITICAL_THRESHOLD = 0.9  # 90% of limit
    CONSECUTIVE_FAILURE_WARNING = 2
    CONSECUTIVE_FAILURE_CRITICAL = 3
    MIN_MESSAGE_COUNT = 2
    THINKING_RATIO_WARNING = 0.4  # Thinking > 40% of tokens is suspicious
    THINKING_RATIO_CRITICAL = 0.6  # Thinking > 60% is critical

    def __init__(self, context: "AgentContext"):
        self._context = context

    def check(self) -> HealthCheckResult:
        """Perform health check.

        Returns:
            HealthCheckResult with list of issues found
        """
        issues = []

        # Check token usage - use AgentContext's token tracker directly for accuracy
        # AgentContext.estimate_tokens() already uses tiktoken for accurate counting
        # but provides fallback for edge cases
        try:
            tokens = self._context.estimate_tokens()
            limit = self._context.token_limit

            if tokens > limit * self.TOKEN_CRITICAL_THRESHOLD:
                issues.append(f"Token usage critical: {tokens:,} / {limit:,}")
            elif tokens > limit * self.TOKEN_WARNING_THRESHOLD:
                issues.append(f"Token usage high: {tokens:,} / {limit:,}")
        except Exception:
            pass

        # Check consecutive failures
        failures = self._context.consecutive_failures
        if failures >= self.CONSECUTIVE_FAILURE_CRITICAL:
            issues.append(f"Multiple consecutive tool failures: {failures}")
        elif failures >= self.CONSECUTIVE_FAILURE_WARNING:
            issues.append(f"Possible tool issue: {failures} consecutive failures")

        # Check for message consistency
        messages = self._context.get_messages()
        if len(messages) < self.MIN_MESSAGE_COUNT:
            issues.append("Message history seems incomplete")

        # Check thinking content ratio
        thinking_issues = self._check_thinking_ratio(messages)
        issues.extend(thinking_issues)

        return HealthCheckResult(issues)

    def _check_thinking_ratio(self, messages: list[Any]) -> list[str]:
        """Check if thinking content占比过高.

        Returns:
            List of issues found
        """
        issues = []
        try:
            total_tokens = self._context.estimate_tokens()
            if total_tokens == 0:
                return issues

            # Calculate thinking tokens
            thinking_tokens = 0
            for msg in messages:
                if hasattr(msg, "thinking") and msg.thinking:
                    # Rough estimate: ~4 chars per token
                    thinking_tokens += len(msg.thinking) // 4

            thinking_ratio = thinking_tokens / total_tokens if total_tokens > 0 else 0

            if thinking_ratio > self.THINKING_RATIO_CRITICAL:
                issues.append(
                    f"Thinking ratio critical: {thinking_ratio:.0%} ({thinking_tokens:,} / {total_tokens:,} tokens)"
                )
            elif thinking_ratio > self.THINKING_RATIO_WARNING:
                issues.append(
                    f"Thinking ratio high: {thinking_ratio:.0%} ({thinking_tokens:,} / {total_tokens:,} tokens)"
                )
        except Exception:
            pass

        return issues

    def get_status(self) -> dict[str, Any]:
        """Get agent status for diagnostics.

        Returns:
            Dict with current state information
        """
        return {
            "token_usage": self._context.estimate_tokens(),
            "token_limit": self._context.token_limit,
            "api_call_count": self._context.api_call_count,
            "session_age_steps": len([m for m in self._context.get_messages() if m.role == "user"]),
            "consecutive_failures": self._context.consecutive_failures,
            "auto_save_enabled": self._context.auto_save,
            "last_auto_save_step": self._context.last_auto_save_step,
            "thinking_budget": self._context.thinking_budget,
            "mode": self._context.mode.value,
        }

    def get_status_report(self) -> str:
        """Generate a human-readable status report."""
        from ..utils.display import Colors

        status = self.get_status()

        lines = [
            f"{Colors.BOLD}📊 Agent Status Report{Colors.RESET}",
            f"{'=' * 40}",
            f"Token usage: {status['token_usage']:,} / {status['token_limit']:,}",
            f"API calls: {status['api_call_count']}",
            f"Session steps: {status['session_age_steps']}",
            f"Mode: {status['mode']}",
            f"Thinking budget: {status['thinking_budget']:,}",
            f"Consecutive failures: {status['consecutive_failures']}",
            f"Auto-save: {status['auto_save_enabled']} (last: step {status['last_auto_save_step']})",
        ]

        # Add warning indicators
        if status["token_usage"] > status["token_limit"] * 0.8:
            lines.append(f"{Colors.YELLOW}⚠️  Token usage high{Colors.RESET}")
        if status["consecutive_failures"] >= 2:
            lines.append(f"{Colors.YELLOW}⚠️  Multiple recent failures{Colors.RESET}")

        return "\n".join(lines)
