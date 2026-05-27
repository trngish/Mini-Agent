"""Tests for ErrorRecoveryManager."""

import pytest

from mini_agent.core.error_recovery import ErrorPattern, ErrorRecoveryManager


class MockAgent:
    """Mock agent for testing."""

    def __init__(self):
        self.messages = []
        self.token_limit = 100000
        self.thinking_budget = 16384
        self.mode = type("Mode", (), {"value": "YOLO"})()
        self._error_patterns = {}
        self._error_history = []
        self._consecutive_failures = 0
        self.api_call_count = 0
        self.auto_save = True
        self._last_auto_save_step = 0

    def _estimate_tokens(self):
        return 50000

    def record_context(self, content, category="auto"):
        pass


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def error_recovery(mock_agent):
    return ErrorRecoveryManager(mock_agent)


class TestErrorRecoveryManager:
    """Test ErrorRecoveryManager functionality."""

    def test_record_error_updates_patterns(self, error_recovery):
        """Test that record_error updates error patterns by tool."""
        error_recovery.record_error("File not found", "read_file(/path/to/file)")
        assert error_recovery._error_patterns["read_file"] == 1

        error_recovery.record_error("Permission denied", "bash(git status)")
        assert error_recovery._error_patterns["bash"] == 1

        error_recovery.record_error("Another read error", "read_file(another/path)")
        assert error_recovery._error_patterns["read_file"] == 2

    def test_record_error_trims_history(self, error_recovery):
        """Test that error history is trimmed to MAX_ERROR_HISTORY."""
        for i in range(25):
            error_recovery.record_error(f"Error {i}", f"tool_{i % 5}(args)")

        assert len(error_recovery._error_history) == ErrorRecoveryManager.MAX_ERROR_HISTORY

    def test_record_success_resets_consecutive_failures(self, error_recovery):
        """Test that record_success resets consecutive failures."""
        error_recovery.record_failure()
        error_recovery.record_failure()
        error_recovery.record_failure()
        assert error_recovery.consecutive_failures == 3

        error_recovery.record_success()
        assert error_recovery.consecutive_failures == 0

    def test_record_failure_increments_counter(self, error_recovery):
        """Test that record_failure increments consecutive failures."""
        assert error_recovery.consecutive_failures == 0
        error_recovery.record_failure()
        assert error_recovery.consecutive_failures == 1
        error_recovery.record_failure()
        assert error_recovery.consecutive_failures == 2

    def test_should_retry_within_limit(self, error_recovery):
        """Test should_retry returns True when within max_attempts."""
        assert error_recovery.should_retry("read_file", 0) is True
        assert error_recovery.should_retry("read_file", 1) is True
        assert error_recovery.should_retry("read_file", 2) is True

    def test_should_retry_exceeds_limit(self, error_recovery):
        """Test should_retry returns False when attempt >= max_attempts."""
        assert error_recovery.should_retry("read_file", 3) is False
        assert error_recovery.should_retry("read_file", 4) is False

    def test_get_backoff_delay_calculation(self, error_recovery):
        """Test exponential backoff delay calculation."""
        # base_delay = 0.5, exponential_base = 2.0
        # delay = 0.5 * (2^attempt)
        assert error_recovery.get_backoff_delay(0, 0.5) == 0.5
        assert error_recovery.get_backoff_delay(1, 0.5) == 1.0
        assert error_recovery.get_backoff_delay(2, 0.5) == 2.0
        assert error_recovery.get_backoff_delay(3, 0.5) == 4.0

    def test_get_patterns(self, error_recovery):
        """Test get_patterns returns correct structure."""
        error_recovery.record_error("Error 1", "tool1(args)")
        error_recovery.record_error("Error 2", "tool1(args)")
        error_recovery.record_error("Error 3", "tool2(args)")
        error_recovery.record_failure()
        error_recovery.record_failure()

        patterns = error_recovery.get_patterns()

        assert "error_counts_by_tool" in patterns
        assert "recent_errors" in patterns
        assert "total_consecutive_failures" in patterns
        assert patterns["error_counts_by_tool"]["tool1"] == 2
        assert patterns["error_counts_by_tool"]["tool2"] == 1
        assert patterns["total_consecutive_failures"] == 2

    def test_get_top_failing_tools(self, error_recovery):
        """Test get_top_failing_tools returns sorted results."""
        error_recovery.record_error("Error", "tool_a(args)")
        for _ in range(5):
            error_recovery.record_error("Error", "tool_b(args)")
        for _ in range(3):
            error_recovery.record_error("Error", "tool_c(args)")

        top = error_recovery.get_top_failing_tools(limit=3)

        assert top[0] == ("tool_b", 5)
        assert top[1] == ("tool_c", 3)
        assert top[2] == ("tool_a", 1)

    def test_get_top_failing_tools_respects_limit(self, error_recovery):
        """Test get_top_failing_tools respects the limit parameter."""
        for i in range(5):
            error_recovery.record_error("Error", f"tool_{i}(args)")

        top = error_recovery.get_top_failing_tools(limit=2)
        assert len(top) == 2

    def test_get_suggestions_when_failures_high(self, error_recovery, mock_agent):
        """Test suggestions generated when consecutive failures are high."""
        error_recovery.record_failure()
        error_recovery.record_failure()

        suggestions = error_recovery.get_suggestions()
        assert any("review" in s.lower() for s in suggestions)

    def test_get_suggestions_token_high(self, error_recovery, mock_agent):
        """Test suggestion when token usage is high."""
        # mock_agent._estimate_tokens returns 50000, token_limit is 100000
        # 50000 > 100000 * 0.8 = 80000 is False, so no suggestion
        suggestions = error_recovery.get_suggestions()
        assert not any("token" in s.lower() for s in suggestions)

        # Now set token_limit lower to trigger warning
        mock_agent.token_limit = 50000
        suggestions = error_recovery.get_suggestions()
        assert any("token" in s.lower() for s in suggestions)

    def test_get_suggestions_tool_failures(self, error_recovery):
        """Test suggestion when a tool has many failures."""
        for _ in range(3):
            error_recovery.record_error("Error", "failing_tool(args)")

        suggestions = error_recovery.get_suggestions()
        assert any("failing_tool" in s for s in suggestions)

    def test_get_suggestions_long_session(self, error_recovery, mock_agent):
        """Test suggestion for long sessions."""
        # Add 31 user messages
        for _ in range(31):
            mock_agent.messages.append(type("Message", (), {"role": "user"})())

        suggestions = error_recovery.get_suggestions()
        assert any("Long session" in s for s in suggestions)

    def test_get_recovery_strategy(self, error_recovery):
        """Test get_recovery_strategy returns retry strategy."""
        strategy = error_recovery.get_recovery_strategy("any_tool")
        assert "max_attempts" in strategy
        assert "backoff" in strategy
        assert strategy["max_attempts"] == 3
        assert strategy["backoff"] == "exponential"

    def test_consecutive_failures_property(self, error_recovery):
        """Test consecutive_failures property getter and setter."""
        assert error_recovery.consecutive_failures == 0

        error_recovery.consecutive_failures = 5
        assert error_recovery.consecutive_failures == 5

        error_recovery.consecutive_failures = 0
        assert error_recovery.consecutive_failures == 0


class TestErrorPattern:
    """Test ErrorPattern class."""

    def test_severity_critical(self):
        """Test critical severity level."""
        pattern = ErrorPattern("tool", 5, [])
        assert pattern.severity == "critical"

        pattern = ErrorPattern("tool", 10, [])
        assert pattern.severity == "critical"

    def test_severity_warning(self):
        """Test warning severity level."""
        pattern = ErrorPattern("tool", 3, [])
        assert pattern.severity == "warning"

        pattern = ErrorPattern("tool", 4, [])
        assert pattern.severity == "warning"

    def test_severity_info(self):
        """Test info severity level."""
        pattern = ErrorPattern("tool", 1, [])
        assert pattern.severity == "info"

        pattern = ErrorPattern("tool", 2, [])
        assert pattern.severity == "info"

    def test_error_pattern_properties(self):
        """Test ErrorPattern stores values correctly."""
        recent_errors = [{"error": "test", "context": "tool(args)"}]
        pattern = ErrorPattern("test_tool", 3, recent_errors)

        assert pattern.tool_name == "test_tool"
        assert pattern.count == 3
        assert pattern.recent_errors == recent_errors
