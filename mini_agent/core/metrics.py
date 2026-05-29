"""Performance metrics tracking for agent sessions.

Provides detailed timing metrics for steps, tools, and API calls.
"""

import json
from datetime import datetime
from pathlib import Path
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
    METRICS_LOG_DIR = Path.home() / ".mini-agent" / "metrics"
    FLUSH_INTERVAL_STEPS = 10  # Flush to disk every N steps

    def __init__(self, context: "AgentContext"):
        self._context = context
        self._step_durations: list[float] = []
        self._tool_execution_times: dict[str, list[float]] = {}
        self._api_latencies: list[float] = []
        self._session_id: str | None = None
        self._step_count_since_flush = 0
        self._tool_success_count: dict[str, int] = {}
        self._tool_failure_count: dict[str, int] = {}
        self._ensure_metrics_dir()

    def _ensure_metrics_dir(self) -> None:
        """Ensure metrics directory exists."""
        self.METRICS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID for metrics logging."""
        self._session_id = session_id

    def _persist_metrics(self) -> None:
        """Persist current metrics to disk for historical analysis."""
        if not self._session_id:
            return

        try:
            metrics_data = self.get_metrics()
            metrics_data["session_id"] = self._session_id
            metrics_data["timestamp"] = datetime.now().isoformat()

            # Save to session-specific metrics file
            metrics_file = self.METRICS_LOG_DIR / f"{self._session_id}.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)

            # Also maintain a summary file for quick access
            summary_file = self.METRICS_LOG_DIR / "latest_session.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)
        except Exception:
            pass  # Silently fail - metrics persistence is not critical

    def record_step_duration(self, duration: float) -> None:
        """Record a step's duration.

        Args:
            duration: Duration in seconds
        """
        self._step_durations.append(duration)
        if len(self._step_durations) > self.MAX_STEP_HISTORY:
            self._step_durations = self._step_durations[-self.MAX_STEP_HISTORY :]

        self._step_count_since_flush += 1
        if self._step_count_since_flush >= self.FLUSH_INTERVAL_STEPS:
            self._persist_metrics()
            self._step_count_since_flush = 0

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

    def record_tool_result(self, tool_name: str, success: bool) -> None:
        """Record tool execution result for hit rate calculation.

        Args:
            tool_name: Name of the tool
            success: Whether the tool execution succeeded
        """
        if success:
            self._tool_success_count[tool_name] = self._tool_success_count.get(tool_name, 0) + 1
        else:
            self._tool_failure_count[tool_name] = self._tool_failure_count.get(tool_name, 0) + 1

    def get_tool_hit_rate(self, tool_name: str) -> float:
        """Get hit rate (success rate) for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Success rate between 0.0 and 1.0
        """
        successes = self._tool_success_count.get(tool_name, 0)
        failures = self._tool_failure_count.get(tool_name, 0)
        total = successes + failures
        if total == 0:
            return 0.0
        return successes / total

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
                successes = self._tool_success_count.get(tool_name, 0)
                failures = self._tool_failure_count.get(tool_name, 0)
                total_calls = successes + failures
                tool_stats[tool_name] = {
                    "calls": len(durations),
                    "avg_seconds": sum(durations) / len(durations),
                    "total_seconds": sum(durations),
                    "successes": successes,
                    "failures": failures,
                    "hit_rate": successes / total_calls if total_calls > 0 else 0.0,
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
