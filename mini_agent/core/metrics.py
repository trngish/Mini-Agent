"""Performance metrics tracking for agent sessions.

Provides detailed timing metrics for steps, tools, and API calls.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_context import AgentContext


class PerformanceMetrics:
    """Tracks and calculates performance metrics for agent sessions.

    Metrics collected:
    - Step durations (each step in seconds)
    - Tool execution times (per tool name)
    - API call latencies
    """

    MAX_STEP_HISTORY = 50
    MAX_TOOL_HISTORY = 20

    def __init__(self, context: "AgentContext"):
        self._context = context
        self._step_durations: list[float] = []
        self._tool_execution_times: dict[str, list[float]] = {}
        self._api_latencies: list[float] = []

    def record_step_duration(self, duration: float) -> None:
        """Record a step's duration.

        Args:
            duration: Duration in seconds
        """
        self._step_durations.append(duration)
        if len(self._step_durations) > self.MAX_STEP_HISTORY:
            self._step_durations = self._step_durations[-self.MAX_STEP_HISTORY :]

    def record_tool_duration(self, tool_name: str, duration: float) -> None:
        """Record a tool's execution duration.

        Args:
            tool_name: Name of the tool
            duration: Duration in seconds
        """
        if tool_name not in self._tool_execution_times:
            self._tool_execution_times[tool_name] = []
        self._tool_execution_times[tool_name].append(duration)
        if len(self._tool_execution_times[tool_name]) > self.MAX_TOOL_HISTORY:
            self._tool_execution_times[tool_name] = self._tool_execution_times[tool_name][-self.MAX_TOOL_HISTORY :]

    def record_api_latency(self, latency: float) -> None:
        """Record API call latency.

        Args:
            latency: Latency in seconds
        """
        self._api_latencies.append(latency)

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the current session.

        Returns:
            Dict with timing metrics for steps, tools, and API calls
        """
        # Calculate step duration stats
        step_stats = {}
        if self._step_durations:
            step_stats = {
                "count": len(self._step_durations),
                "avg_seconds": sum(self._step_durations) / len(self._step_durations),
                "min_seconds": min(self._step_durations),
                "max_seconds": max(self._step_durations),
                "total_seconds": sum(self._step_durations),
            }

        # Calculate tool execution stats
        tool_stats = {}
        for tool_name, durations in self._tool_execution_times.items():
            if durations:
                tool_stats[tool_name] = {
                    "calls": len(durations),
                    "avg_seconds": sum(durations) / len(durations),
                    "total_seconds": sum(durations),
                }

        # API latency stats
        api_stats = {}
        if self._api_latencies:
            api_stats = {
                "count": len(self._api_latencies),
                "avg_seconds": sum(self._api_latencies) / len(self._api_latencies),
                "min_seconds": min(self._api_latencies),
                "max_seconds": max(self._api_latencies),
            }

        return {
            "step_metrics": step_stats,
            "tool_metrics": tool_stats,
            "api_metrics": api_stats,
            "api_call_count": self._context.api_call_count,
        }

    @property
    def step_durations(self) -> list[float]:
        """Get step duration history."""
        return self._step_durations.copy()

    @property
    def tool_execution_times(self) -> dict[str, list[float]]:
        """Get tool execution time history."""
        return {k: v.copy() for k, v in self._tool_execution_times.items()}
