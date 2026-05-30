"""核心智能体组件。

该包包含从 agent.py 提取的模块化组件：
- thinking_budget: M2.7 自适应思考预算管理
- health_check: 自我健康检查和诊断
- metrics: 性能指标跟踪
- error_recovery: 错误模式学习和恢复策略
- tool_execution: 工具超时和压缩工具
"""

from .error_recovery import ErrorRecoveryManager
from .health_check import HealthChecker, HealthCheckResult
from .metrics import PerformanceMetrics
from .thinking_budget import ThinkingBudgetManager

__all__ = [
    "ThinkingBudgetManager",
    "HealthChecker",
    "HealthCheckResult",
    "PerformanceMetrics",
    "ErrorRecoveryManager",
]
