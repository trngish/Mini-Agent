"""优雅的重试机制模块

提供装饰器和工具函数以支持异步函数的重试逻辑。

特性：
- 支持指数退避策略
- 可配置重试次数和间隔
- 支持指定可重试的异常类型
- 详细日志记录
- 完全解耦，对业务代码无侵入
"""

import asyncio
import builtins
import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 不应重试的异常（致命的/不可恢复的）
NON_RETRYABLE = (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit)

# 预计算的可重试异常类型集合（在模块加载时缓存）
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] | None = None


def _get_retryable_exceptions() -> tuple[type[Exception], ...]:
    """返回默认可重试的异常（除致命类型外的所有异常）。

    为性能在模块级别缓存。
    """
    global _RETRYABLE_EXCEPTIONS
    if _RETRYABLE_EXCEPTIONS is None:
        result = []
        for name in dir(builtins):
            obj = getattr(builtins, name)
            if isinstance(obj, type) and issubclass(obj, Exception) and not issubclass(obj, NON_RETRYABLE):
                result.append(obj)
        _RETRYABLE_EXCEPTIONS = tuple(result)
    return _RETRYABLE_EXCEPTIONS


class RetryConfig:
    """重试配置类"""

    def __init__(
        self,
        enabled: bool = True,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        """
        Args:
            enabled: 是否启用重试机制
            max_retries: 最大重试次数
            initial_delay: 初始延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_base: 指数退避基数
            retryable_exceptions: 可重试异常类型元组（默认为所有非致命异常）
        """
        self.enabled = enabled
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions or _get_retryable_exceptions()

    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间（指数退避）

        Args:
            attempt: 当前尝试次数（从0开始）

        Returns:
            延迟时间（秒）
        """
        delay = self.initial_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)


class RetryExhaustedError(Exception):
    """重试耗尽异常"""

    def __init__(self, last_exception: Exception, attempts: int):
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(f"Retry failed after {attempts} attempts. Last error: {str(last_exception)}")


def async_retry(
    config: RetryConfig | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[..., Any]:
    """异步函数重试装饰器

    Args:
        config: 重试配置对象，如果为None则使用默认配置
        on_retry: 重试时的回调函数，接收异常和当前尝试次数

    Returns:
        装饰器函数

    Example:
        ```python
        @async_retry(RetryConfig(max_retries=3, initial_delay=1.0))
        async def call_api():
            # API调用代码
            pass
        ```
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except NON_RETRYABLE:
                    raise

                except config.retryable_exceptions as e:
                    if attempt >= config.max_retries:
                        logger.error(
                            f"Function {func.__name__} retry failed, reached maximum retry count {config.max_retries}"
                        )
                        raise RetryExhaustedError(e, attempt + 1) from e

                    # 计算延迟时间
                    delay = config.calculate_delay(attempt)

                    # 记录日志
                    logger.warning(
                        f"Function {func.__name__} call {attempt + 1} failed: {str(e)}, "
                        f"retrying attempt {attempt + 2} after {delay:.2f} seconds"
                    )

                    # 调用回调函数
                    if on_retry:
                        on_retry(e, attempt + 1)

                    # 等待后重试
                    await asyncio.sleep(delay)

        return wrapper

    return decorator
