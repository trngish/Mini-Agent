"""Health check and diagnostics for agent self-monitoring.

Provides self-health checks after each step to detect and warn about issues.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


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

    def __init__(self, agent: "Agent"):
        self._agent = agent

    def check(self) -> HealthCheckResult:
        """Perform health check.

        Returns:
            HealthCheckResult with list of issues found
        """
        issues = []

        # Check token usage
        try:
            tokens = self._agent._estimate_tokens()
            limit = self._agent.token_limit

            if tokens > limit * self.TOKEN_CRITICAL_THRESHOLD:
                issues.append(f"Token usage critical: {tokens:,} / {limit:,}")
            elif tokens > limit * self.TOKEN_WARNING_THRESHOLD:
                issues.append(f"Token usage high: {tokens:,} / {limit:,}")
        except Exception:
            pass

        # Check consecutive failures
        failures = self._agent._consecutive_failures
        if failures >= self.CONSECUTIVE_FAILURE_CRITICAL:
            issues.append(f"Multiple consecutive tool failures: {failures}")
        elif failures >= self.CONSECUTIVE_FAILURE_WARNING:
            issues.append(f"Possible tool issue: {failures} consecutive failures")

        # Check for message consistency
        if len(self._agent.messages) < self.MIN_MESSAGE_COUNT:
            issues.append("Message history seems incomplete")

        return HealthCheckResult(issues)

    def get_status(self) -> dict:
        """Get agent status for diagnostics.

        Returns:
            Dict with current state information
        """
        return {
            "token_usage": self._agent._estimate_tokens(),
            "token_limit": self._agent.token_limit,
            "api_call_count": self._agent.api_call_count,
            "session_age_steps": len([m for m in self._agent.messages if m.role == "user"]),
            "consecutive_failures": self._agent._consecutive_failures,
            "auto_save_enabled": self._agent.auto_save,
            "last_auto_save_step": self._agent._last_auto_save_step,
            "thinking_budget": self._agent.thinking_budget,
            "mode": self._agent.mode.value,
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
        if status['token_usage'] > status['token_limit'] * 0.8:
            lines.append(f"{Colors.YELLOW}⚠️  Token usage high{Colors.RESET}")
        if status['consecutive_failures'] >= 2:
            lines.append(f"{Colors.YELLOW}⚠️  Multiple recent failures{Colors.RESET}")

        return '\n'.join(lines)