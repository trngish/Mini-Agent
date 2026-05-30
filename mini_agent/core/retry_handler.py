"""统一重试处理器，用于工具执行。

整合了以下模块的重试逻辑：
- ErrorRecoveryManager (should_retry, get_backoff_delay)
- tool_execution.is_transient_error

为所有工具执行提供单一、一致的重试接口。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


# 瞬态错误模式，值得重试
TRANSIENT_PATTERNS = [
    # 网络超时
    "timeout",
    "timed out",
    "timed_out",
    # 网络连接
    "connection",
    "econreset",
    "etimedout",
    "enotfound",
    "econnrefused",
    "econnaborted",
    # 服务状态
    "temporary",
    "unavailable",
    "overloaded",
    "backpressure",
    "service overloaded",
    "busy",
    "degraded",
    # 速率限制
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttl",
    "quota exceeded",
    # 可重试标识
    "retry",
    "retry after",
    "retry_after",
    "please retry",
    "try again",
    # HTTP 状态码
    "429",
    "503",
    "502",
    "504",
    # 云服务错误
    "server error",
    "internal error",
    "maintenance",
]


class RetryHandler:
    """统一重试处理器，用于工具执行。

    提供：
    - 瞬态错误检测
    - 重试决策判断
    - 指数退避计算
    """

    def __init__(self, agent: "Agent", max_retries: int = 3, base_delay: float = 0.5):
        """初始化 RetryHandler。

        Args:
            agent: 智能体实例（用于访问配置）
            max_retries: 最大重试次数
            base_delay: 指数退避基础延迟（秒）
        """
        self._agent = agent
        self._max_retries = max_retries
        self._base_delay = base_delay

    def should_retry(self, error: str | Exception, attempt: int) -> bool:
        """检查错误是否应触发重试。

        Args:
            error: 错误消息或异常
            attempt: 当前尝试次数（从0开始计数）

        Returns:
            如果错误是瞬态的且还有重试机会则返回 True
        """
        if attempt >= self._max_retries:
            return False

        error_str = str(error).lower()
        return self.is_transient_error(error_str)

    def is_transient_error(self, error: str) -> bool:
        """检查错误是否为瞬态的（值得重试）。

        Args:
            error: 错误消息（内部会自动转为小写）

        Returns:
            如果错误是瞬态的则返回 True
        """
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in TRANSIENT_PATTERNS)

    def get_delay(self, attempt: int) -> float:
        """计算指数退避延迟。

        Args:
            attempt: 当前尝试次数（从0开始计数）

        Returns:
            延迟时间（秒）
        """
        return float(self._base_delay * (2**attempt))

    def get_max_retries(self) -> int:
        """获取最大重试次数。"""
        return self._max_retries


def create_retry_handler(agent: "Agent") -> RetryHandler:
    """工厂函数，用于从智能体配置创建 RetryHandler。

    Args:
        agent: 智能体实例

    Returns:
        根据智能体设置配置好的 RetryHandler
    """
    # 尝试从智能体获取配置，降级使用默认值
    max_retries = 3
    base_delay = 0.5

    # 检查 M27 配置
    if hasattr(agent, "m27_config") and agent.m27_config:
        max_retries = agent.m27_config.get("max_tool_retries", 3)

    # 检查智能体上的重试配置
    if hasattr(agent, "_retry_config"):
        max_retries = agent._retry_config.max_retries
        base_delay = agent._retry_config.initial_delay

    return RetryHandler(agent, max_retries=max_retries, base_delay=base_delay)
