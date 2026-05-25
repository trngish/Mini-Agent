"""Core agent components.

This package contains modular components extracted from agent.py:
- thinking_budget: Adaptive thinking budget management for M2.7
- health_check: Self-health checks and diagnostics
- metrics: Performance metrics tracking
- error_recovery: Error pattern learning and recovery strategies
- tool_execution: Tool timeout and compression utilities
"""

from .thinking_budget import ThinkingBudgetManager
from .health_check import HealthChecker, HealthCheckResult
from .metrics import PerformanceMetrics
from .error_recovery import ErrorRecoveryManager
from .tool_execution import (
    get_tool_timeout,
    compress_tool_result,
    is_transient_error,
    execute_with_timeout,
    should_compress_result,
    DEFAULT_TOOL_TIMEOUTS,
)

__all__ = [
    "ThinkingBudgetManager",
    "HealthChecker",
    "HealthCheckResult",
    "PerformanceMetrics",
    "ErrorRecoveryManager",
    "get_tool_timeout",
    "compress_tool_result",
    "is_transient_error",
    "execute_with_timeout",
    "should_compress_result",
    "DEFAULT_TOOL_TIMEOUTS",
]