"""Tests for PerformanceMetrics."""

import pytest

from mini_agent.core.agent_context import AgentContext
from mini_agent.core.metrics import PerformanceMetrics
from mini_agent.schema import AgentMode, Message


@pytest.fixture
def mock_agent_context():
    """Create AgentContext for testing."""
    return AgentContext(
        messages=[Message(role="system", content="You are a test agent.")],
        mode=AgentMode.YOLO,
        max_steps=10,
        token_limit=100000,
        api_call_count=0,
        api_total_tokens=0,
        is_m27=False,
        thinking_budget=16384,
    )


@pytest.fixture
def metrics(mock_agent_context):
    return PerformanceMetrics(mock_agent_context)


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""

    def test_record_step_duration(self, metrics):
        """Test recording step duration."""
        metrics.record_step_duration(1.5)
        metrics.record_step_duration(2.0)
        metrics.record_step_duration(0.5)

        assert len(metrics.step_durations) == 3
        assert 1.5 in metrics.step_durations
        assert 2.0 in metrics.step_durations
        assert 0.5 in metrics.step_durations

    def test_record_step_duration_respects_max_history(self, metrics):
        """Test that step history is capped at MAX_STEP_HISTORY."""
        for i in range(60):
            metrics.record_step_duration(float(i))

        assert len(metrics.step_durations) == PerformanceMetrics.MAX_STEP_HISTORY
        # Should have the most recent 50
        assert 59.0 in metrics.step_durations
        assert 9.0 not in metrics.step_durations

    def test_record_tool_duration(self, metrics):
        """Test recording tool duration."""
        metrics.record_tool_duration("read_file", 0.5)
        metrics.record_tool_duration("read_file", 0.3)
        metrics.record_tool_duration("bash", 1.0)

        assert "read_file" in metrics.tool_execution_times
        assert "bash" in metrics.tool_execution_times
        assert len(metrics.tool_execution_times["read_file"]) == 2
        assert len(metrics.tool_execution_times["bash"]) == 1

    def test_record_tool_duration_respects_max_history(self, metrics):
        """Test that per-tool history is capped at MAX_TOOL_HISTORY."""
        for i in range(25):
            metrics.record_tool_duration("read_file", float(i))

        assert len(metrics.tool_execution_times["read_file"]) == PerformanceMetrics.MAX_TOOL_HISTORY

    def test_record_api_latency(self, metrics):
        """Test recording API latency."""
        metrics.record_api_latency(0.1)
        metrics.record_api_latency(0.2)
        metrics.record_api_latency(0.15)

        # API latencies are not exposed via property, but get_metrics includes them
        metrics_data = metrics.get_metrics()
        assert "api_metrics" in metrics_data
        assert metrics_data["api_metrics"]["count"] == 3

    def test_get_metrics_step_stats(self, metrics):
        """Test step metrics calculation."""
        metrics.record_step_duration(1.0)
        metrics.record_step_duration(2.0)
        metrics.record_step_duration(3.0)

        result = metrics.get_metrics()

        assert "step_metrics" in result
        step_stats = result["step_metrics"]
        assert step_stats["count"] == 3
        assert step_stats["avg_seconds"] == 2.0
        assert step_stats["min_seconds"] == 1.0
        assert step_stats["max_seconds"] == 3.0
        assert step_stats["total_seconds"] == 6.0

    def test_get_metrics_tool_stats(self, metrics):
        """Test tool metrics calculation."""
        metrics.record_tool_duration("read_file", 0.5)
        metrics.record_tool_duration("read_file", 1.5)
        metrics.record_tool_duration("bash", 2.0)

        result = metrics.get_metrics()

        assert "tool_metrics" in result
        tool_stats = result["tool_metrics"]

        assert "read_file" in tool_stats
        assert tool_stats["read_file"]["calls"] == 2
        assert tool_stats["read_file"]["avg_seconds"] == 1.0
        assert tool_stats["read_file"]["total_seconds"] == 2.0

        assert "bash" in tool_stats
        assert tool_stats["bash"]["calls"] == 1
        assert tool_stats["bash"]["avg_seconds"] == 2.0

    def test_get_metrics_api_stats(self, metrics):
        """Test API metrics calculation."""
        metrics.record_api_latency(0.1)
        metrics.record_api_latency(0.3)
        metrics.record_api_latency(0.2)

        result = metrics.get_metrics()

        assert "api_metrics" in result
        api_stats = result["api_metrics"]
        assert api_stats["count"] == 3
        assert api_stats["avg_seconds"] == pytest.approx(0.2)
        assert api_stats["min_seconds"] == 0.1
        assert api_stats["max_seconds"] == 0.3

    def test_get_metrics_includes_api_call_count(self, metrics):
        """Test that get_metrics includes api_call_count from agent."""
        result = metrics.get_metrics()
        assert "api_call_count" in result
        assert result["api_call_count"] == 0

    def test_get_metrics_empty(self, metrics):
        """Test get_metrics with no recorded data."""
        result = metrics.get_metrics()

        assert "step_metrics" in result
        assert result["step_metrics"] == {}

        assert "tool_metrics" in result
        assert result["tool_metrics"] == {}

        assert "api_metrics" in result
        assert result["api_metrics"] == {}

    def test_step_durations_property(self, metrics):
        """Test step_durations property returns copy."""
        metrics.record_step_duration(1.0)
        metrics.record_step_duration(2.0)

        durations = metrics.step_durations
        durations.append(999.0)  # Should not affect internal state

        assert 999.0 not in metrics.step_durations

    def test_tool_execution_times_property(self, metrics):
        """Test tool_execution_times property returns deep copy."""
        metrics.record_tool_duration("read_file", 0.5)

        times = metrics.tool_execution_times
        times["new_tool"] = [999.0]  # Should not affect internal state

        assert "new_tool" not in metrics.tool_execution_times

    def test_get_metrics_single_call(self, metrics):
        """Test metrics with single recording."""
        metrics.record_step_duration(5.0)
        metrics.record_tool_duration("bash", 2.0)
        metrics.record_api_latency(0.5)

        result = metrics.get_metrics()

        assert result["step_metrics"]["count"] == 1
        assert result["step_metrics"]["avg_seconds"] == 5.0
        assert result["tool_metrics"]["bash"]["calls"] == 1
        assert result["api_metrics"]["count"] == 1

    def test_max_constants(self):
        """Test MAX_STEP_HISTORY and MAX_TOOL_HISTORY constants."""
        assert PerformanceMetrics.MAX_STEP_HISTORY == 50
        assert PerformanceMetrics.MAX_TOOL_HISTORY == 20
