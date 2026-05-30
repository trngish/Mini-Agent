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
    """跟踪并计算智能体会话的性能指标。

    收集的指标：
    - 步骤耗时（每个步骤的秒数）
    - 工具执行时间（按工具名称）
    - API 调用延迟
    """

    MAX_STEP_HISTORY = 50
    MAX_TOOL_HISTORY = 20
    METRICS_LOG_DIR = Path.home() / ".mini-agent" / "metrics"
    FLUSH_INTERVAL_STEPS = 10  # 每 N 步刷新到磁盘

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
        """确保指标目录存在。"""
        self.METRICS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def set_session_id(self, session_id: str) -> None:
        """设置当前会话 ID 用于指标日志记录。"""
        self._session_id = session_id

    def _persist_metrics(self) -> None:
        """将当前指标持久化到磁盘以进行历史分析。"""
        if not self._session_id:
            return

        try:
            metrics_data = self.get_metrics()
            metrics_data["session_id"] = self._session_id
            metrics_data["timestamp"] = datetime.now().isoformat()

            # 保存到会话特定的指标文件
            metrics_file = self.METRICS_LOG_DIR / f"{self._session_id}.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)

            # 同时维护一个摘要文件以便快速访问
            summary_file = self.METRICS_LOG_DIR / "latest_session.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)
        except Exception:
            pass  # 静默失败 - 指标持久化不是关键操作

    def record_step_duration(self, duration: float) -> None:
        """记录单个步骤的耗时。

        Args:
            duration: 耗时秒数
        """
        self._step_durations.append(duration)
        if len(self._step_durations) > self.MAX_STEP_HISTORY:
            self._step_durations = self._step_durations[-self.MAX_STEP_HISTORY :]

        self._step_count_since_flush += 1
        if self._step_count_since_flush >= self.FLUSH_INTERVAL_STEPS:
            self._persist_metrics()
            self._step_count_since_flush = 0

    def record_tool_duration(self, tool_name: str, duration: float) -> None:
        """记录工具的执行耗时。

        Args:
            tool_name: 工具名称
            duration: 耗时秒数
        """
        if tool_name not in self._tool_execution_times:
            self._tool_execution_times[tool_name] = []
        self._tool_execution_times[tool_name].append(duration)
        if len(self._tool_execution_times[tool_name]) > self.MAX_TOOL_HISTORY:
            self._tool_execution_times[tool_name] = self._tool_execution_times[tool_name][-self.MAX_TOOL_HISTORY :]

    def record_tool_result(self, tool_name: str, success: bool) -> None:
        """记录工具执行结果以计算命中率。

        Args:
            tool_name: 工具名称
            success: 工具执行是否成功
        """
        if success:
            self._tool_success_count[tool_name] = self._tool_success_count.get(tool_name, 0) + 1
        else:
            self._tool_failure_count[tool_name] = self._tool_failure_count.get(tool_name, 0) + 1

    def get_tool_hit_rate(self, tool_name: str) -> float:
        """获取特定工具的命中率（成功率）。

        Args:
            tool_name: 工具名称

        Returns:
            0.0 到 1.0 之间的成功率
        """
        successes = self._tool_success_count.get(tool_name, 0)
        failures = self._tool_failure_count.get(tool_name, 0)
        total = successes + failures
        if total == 0:
            return 0.0
        return successes / total

    def record_api_latency(self, latency: float) -> None:
        """记录 API 调用延迟。

        Args:
            latency: 延迟秒数
        """
        self._api_latencies.append(latency)

    def get_metrics(self) -> dict[str, Any]:
        """获取当前会话的性能指标。

        Returns:
            包含步骤、工具和 API 调用的计时指标的字典
        """
        # 计算步骤耗时统计
        step_stats = {}
        if self._step_durations:
            step_stats = {
                "count": len(self._step_durations),
                "avg_seconds": sum(self._step_durations) / len(self._step_durations),
                "min_seconds": min(self._step_durations),
                "max_seconds": max(self._step_durations),
                "total_seconds": sum(self._step_durations),
            }

        # 计算工具执行统计
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

        # API 延迟统计
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
        """获取步骤耗时历史记录。"""
        return self._step_durations.copy()

    @property
    def tool_execution_times(self) -> dict[str, list[float]]:
        """获取工具执行时间历史记录。"""
        return {k: v.copy() for k, v in self._tool_execution_times.items()}
