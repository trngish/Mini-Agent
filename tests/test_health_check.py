"""Tests for HealthChecker."""

from pathlib import Path

from mini_agent.core.health_check import HealthChecker, HealthCheckResult
from mini_agent.core.agent_context import AgentContext
from mini_agent.schema import AgentMode


class MockMessage:
    """Mock message for testing."""

    def __init__(self, role="user"):
        self.role = role


class MockTokenTracker:
    """Mock token tracker for testing."""

    def __init__(self, estimated_tokens=50000):
        self._estimated_tokens = estimated_tokens

    def estimate_tokens(self, messages):
        return self._estimated_tokens


def create_mock_context(token_limit=100000, messages_count=5, estimated_tokens=50000, consecutive_failures=0):
    """Create a mock AgentContext for testing."""
    messages = [MockMessage("user")] * max(1, messages_count)
    return AgentContext(
        messages=messages,
        mode=AgentMode.YOLO,
        max_steps=50,
        workspace_dir=Path("."),
        token_limit=token_limit,
        api_call_count=0,
        api_total_tokens=0,
        is_m27=False,
        thinking_budget=16384,
        auto_save=True,
        token_tracker=MockTokenTracker(estimated_tokens),
    )


class TestHealthCheckResult:
    """Test HealthCheckResult class."""

    def test_has_critical_issues_with_critical(self):
        """Test has_critical_issues returns True for critical issues."""
        result = HealthCheckResult(["Token usage critical: 90000 / 100000"])
        assert result.has_critical_issues is True

    def test_has_critical_issues_with_high(self):
        """Test has_critical_issues returns True for high issues."""
        result = HealthCheckResult(["Token usage high: 85000 / 100000"])
        assert result.has_critical_issues is True

    def test_has_critical_issues_false(self):
        """Test has_critical_issues returns False when no critical issues."""
        result = HealthCheckResult(["Minor warning"])
        assert result.has_critical_issues is False

    def test_has_warnings_true(self):
        """Test has_warnings returns True when issues exist."""
        result = HealthCheckResult(["Some warning"])
        assert result.has_warnings is True

    def test_has_warnings_false(self):
        """Test has_warnings returns False when no issues."""
        result = HealthCheckResult([])
        assert result.has_warnings is False

    def test_empty_issues(self):
        """Test HealthCheckResult with empty issues list."""
        result = HealthCheckResult([])
        result.issues == []
        assert result.has_critical_issues is False
        assert result.has_warnings is False


class TestHealthChecker:
    """Test HealthChecker functionality."""

    def test_check_token_warning(self):
        """Test health check detects token usage warning."""
        context = create_mock_context(token_limit=100000, estimated_tokens=85000)  # 85% of limit

        checker = HealthChecker(context)
        result = checker.check()

        assert result.has_warnings is True
        assert any("high" in issue.lower() for issue in result.issues)

    def test_check_token_critical(self):
        """Test health check detects token usage critical."""
        context = create_mock_context(token_limit=100000, estimated_tokens=95000)  # 95% of limit

        checker = HealthChecker(context)
        result = checker.check()

        assert result.has_critical_issues is True
        assert any("critical" in issue.lower() for issue in result.issues)

    def test_check_token_ok(self):
        """Test health check passes when token usage is ok."""
        context = create_mock_context(token_limit=100000, estimated_tokens=50000)  # 50% of limit

        checker = HealthChecker(context)
        result = checker.check()

        assert result.has_warnings is False

    def test_check_consecutive_failures_warning(self):
        """Test health check detects consecutive failures warning."""
        context = create_mock_context(consecutive_failures=2)
        context._consecutive_failures = 2

        checker = HealthChecker(context)
        result = checker.check()

        assert result.has_warnings is True
        assert any("failure" in issue.lower() for issue in result.issues)

    def test_check_consecutive_failures_critical(self):
        """Test health check detects critical consecutive failures."""
        context = create_mock_context(consecutive_failures=3)
        context._consecutive_failures = 3

        checker = HealthChecker(context)
        result = checker.check()

        # Should detect consecutive failures issue
        assert result.has_warnings is True
        assert any("failure" in issue.lower() for issue in result.issues)

    def test_check_message_consistency(self):
        """Test health check detects incomplete message history."""
        context = create_mock_context(messages_count=1)  # Less than MIN_MESSAGE_COUNT

        checker = HealthChecker(context)
        result = checker.check()

        assert any("incomplete" in issue.lower() or "message" in issue.lower() for issue in result.issues)

    def test_check_no_issues(self):
        """Test health check passes when no issues."""
        context = create_mock_context(estimated_tokens=50000, consecutive_failures=0)
        context._consecutive_failures = 0

        checker = HealthChecker(context)
        result = checker.check()

        assert result.has_warnings is False
        assert result.has_critical_issues is False

    def test_get_status(self):
        """Test get_status returns correct structure."""
        context = create_mock_context(estimated_tokens=50000, consecutive_failures=2)
        context._consecutive_failures = 2

        checker = HealthChecker(context)
        status = checker.get_status()

        assert "token_usage" in status
        assert "token_limit" in status
        assert "api_call_count" in status
        assert "session_age_steps" in status
        assert "consecutive_failures" in status
        assert "auto_save_enabled" in status
        assert "last_auto_save_step" in status
        assert "thinking_budget" in status
        assert "mode" in status

    def test_get_status_report(self):
        """Test get_status_report generates human readable output."""
        context = create_mock_context(estimated_tokens=50000, consecutive_failures=0)
        context._consecutive_failures = 0

        checker = HealthChecker(context)
        report = checker.get_status_report()

        assert "Status Report" in report
        assert "Token usage" in report
        assert "API calls" in report

    def test_thresholds(self):
        """Test health checker's threshold constants."""
        assert HealthChecker.TOKEN_WARNING_THRESHOLD == 0.8
        assert HealthChecker.TOKEN_CRITICAL_THRESHOLD == 0.9
        assert HealthChecker.CONSECUTIVE_FAILURE_WARNING == 2
        assert HealthChecker.CONSECUTIVE_FAILURE_CRITICAL == 3
        assert HealthChecker.MIN_MESSAGE_COUNT == 2

    def test_check_handles_token_estimation_error(self):
        """Test health check handles token estimation errors gracefully."""
        context = create_mock_context()

        class RaisingTokenTracker:
            def estimate_tokens(self, messages):
                raise Exception("Token estimation failed")

        context.token_tracker = RaisingTokenTracker()

        checker = HealthChecker(context)
        result = checker.check()

        # Should not raise, just continue without token warnings
        assert result is not None

    def test_multiple_issues_detected(self):
        """Test that multiple issues are detected and accumulated."""
        context = create_mock_context(estimated_tokens=95000, messages_count=1, consecutive_failures=3)
        context._consecutive_failures = 3

        checker = HealthChecker(context)
        result = checker.check()

        # Should detect multiple issues
        assert len(result.issues) >= 2