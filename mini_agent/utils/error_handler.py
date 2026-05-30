"""LLM API 错误处理与分类。

提供各种 API 错误类型的详细错误分类和用户友好的错误消息。
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from typing import Any

from .display import Colors


class LLMErrorType(str, Enum):
    """LLM API 错误类型。"""

    # 认证与授权
    AUTHENTICATION_ERROR = "authentication_error"  # 401
    PERMISSION_DENIED = "permission_denied"  # 403

    # 速率限制
    RATE_LIMIT_ERROR = "rate_limit_error"  # 429
    QUOTA_EXCEEDED = "quota_exceeded"  # 429 带有特定消息

    # 服务器错误
    SERVER_ERROR = "server_error"  # 500-599
    SERVICE_UNAVAILABLE = "service_unavailable"  # 503
    GATEWAY_TIMEOUT = "gateway_timeout"  # 504

    # 客户端错误
    BAD_REQUEST = "bad_request"  # 400
    INVALID_REQUEST = "invalid_request"  # 400 带有特定消息
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"  # 400 带有特定消息
    UNPROCESSABLE_ENTITY = "unprocessable_entity"  # 422

    # 网络错误
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    CONNECTION_ERROR = "connection_error"

    # 未知
    UNKNOWN_ERROR = "unknown_error"


class LLMError(Exception):
    """LLM 错误的基类异常，包含错误分类。"""

    def __init__(
        self,
        message: str,
        error_type: LLMErrorType = LLMErrorType.UNKNOWN_ERROR,
        status_code: int | None = None,
        details: str | None = None,
        retry_after: int | None = None,
    ):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details
        self.retry_after = retry_after  # 重试前等待的秒数（用于速率限制）
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"(Status: {self.status_code})")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)

    @property
    def is_retryable(self) -> bool:
        """检查此错误类型是否应该重试。"""
        retryable_types = {
            LLMErrorType.RATE_LIMIT_ERROR,
            LLMErrorType.SERVER_ERROR,
            LLMErrorType.SERVICE_UNAVAILABLE,
            LLMErrorType.GATEWAY_TIMEOUT,
            LLMErrorType.NETWORK_ERROR,
            LLMErrorType.TIMEOUT_ERROR,
            LLMErrorType.CONNECTION_ERROR,
        }
        return self.error_type in retryable_types

    @property
    def user_guidance(self) -> str:
        """获取此错误类型的用户指导。"""
        guidance = {
            LLMErrorType.AUTHENTICATION_ERROR: (
                "Please check your API key is valid and has not expired. Verify your API key in the configuration file."
            ),
            LLMErrorType.PERMISSION_DENIED: (
                "Your API key does not have permission to perform this operation. "
                "Please check your account permissions."
            ),
            LLMErrorType.RATE_LIMIT_ERROR: (
                "Rate limit exceeded. "
                + (
                    f"Please wait {self.retry_after} seconds before retrying."
                    if self.retry_after
                    else "Please wait a moment before retrying."
                )
            ),
            LLMErrorType.QUOTA_EXCEEDED: (
                "Your API quota has been exceeded. Please check your usage limits or upgrade your plan."
            ),
            LLMErrorType.SERVER_ERROR: (
                "The server encountered an internal error. This is usually temporary. Please retry in a few moments."
            ),
            LLMErrorType.SERVICE_UNAVAILABLE: (
                "The service is temporarily unavailable. Please retry in a few moments."
            ),
            LLMErrorType.GATEWAY_TIMEOUT: ("The request timed out. Please retry with a shorter prompt or fewer tools."),
            LLMErrorType.BAD_REQUEST: ("Invalid request format. Please check your input."),
            LLMErrorType.INVALID_REQUEST: ("The request was invalid. Please check the input format."),
            LLMErrorType.CONTEXT_LENGTH_EXCEEDED: (
                "The conversation is too long and exceeded the context limit. "
                "Consider starting a new conversation or reducing the task size."
            ),
            LLMErrorType.UNPROCESSABLE_ENTITY: (
                "The request could not be processed. Please check your input format and parameters."
            ),
            LLMErrorType.NETWORK_ERROR: ("Network connection failed. Please check your internet connection."),
            LLMErrorType.TIMEOUT_ERROR: ("The request timed out. Please try again or reduce the request size."),
            LLMErrorType.CONNECTION_ERROR: (
                "Could not connect to the server. Please check your network and try again."
            ),
            LLMErrorType.UNKNOWN_ERROR: ("An unexpected error occurred. Please try again later."),
        }
        return guidance.get(self.error_type, "An unknown error occurred.")


class LLMErrorClassifier:
    """从异常对象或 HTTP 响应中对 LLM API 错误进行分类。"""

    # 特定错误消息的模式匹配器（为性能编译一次）
    _RATE_LIMIT_PATTERNS = [
        re.compile(r"rate.?limit", re.IGNORECASE),
        re.compile(r"too.?many.?requests", re.IGNORECASE),
        re.compile(r"quota.?exceeded", re.IGNORECASE),
        re.compile(r"api.?rate", re.IGNORECASE),
    ]

    _CONTEXT_LENGTH_PATTERNS = [
        re.compile(r"context.?length", re.IGNORECASE),
        re.compile(r"token.?limit", re.IGNORECASE),
        re.compile(r"too.?long", re.IGNORECASE),
        re.compile(r"maximum.?context", re.IGNORECASE),
        re.compile(r"max_tokens", re.IGNORECASE),
    ]

    _AUTH_PATTERNS = [
        re.compile(r"invalid.?api.?key", re.IGNORECASE),
        re.compile(r"authentication.?failed", re.IGNORECASE),
        re.compile(r"unauthorized", re.IGNORECASE),
        re.compile(r"api.?key.?invalid", re.IGNORECASE),
    ]

    _TIMEOUT_PATTERNS = [
        re.compile(r"timeout", re.IGNORECASE),
        re.compile(r"timed.?out", re.IGNORECASE),
        re.compile(r"request.?timeout", re.IGNORECASE),
    ]

    _CONNECTION_PATTERNS = [
        re.compile(r"connection.*refused", re.IGNORECASE),
        re.compile(r"connection.*reset", re.IGNORECASE),
        re.compile(r"network.*unreachable", re.IGNORECASE),
        re.compile(r"could.*not.*connect", re.IGNORECASE),
    ]

    @classmethod
    def classify(
        cls,
        error: Exception,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> LLMError:
        """对异常进行分类并返回 LLMError。

        Args:
            error: 原始异常
            status_code: HTTP 状态码（如果有）
            response_body: 响应正文文本（如果有）

        Returns:
            带有分类错误类型的 LLMError
        """
        error_str = str(error)
        body_str = response_body or ""

        # 合并以进行模式匹配
        combined_text = f"{error_str} {body_str}"

        # 首先按状态码分类
        if status_code:
            error_type = cls._classify_by_status(status_code, combined_text)
            if error_type != LLMErrorType.UNKNOWN_ERROR:
                retry_after = cls._extract_retry_after(response_body)
                return LLMError(
                    message=error_str,
                    error_type=error_type,
                    status_code=status_code,
                    details=body_str[:500] if body_str else None,
                    retry_after=retry_after,
                )

        # 通过错误消息模式分类
        error_type = cls._classify_by_message(combined_text)
        return LLMError(
            message=error_str,
            error_type=error_type,
            status_code=status_code,
            details=body_str[:500] if body_str else None,
        )

    @classmethod
    def _classify_by_status(cls, status_code: int, message: str) -> LLMErrorType:
        """根据 HTTP 状态码对错误进行分类。"""
        if status_code == 400:
            # 检查特定的 400 错误
            if cls._matches_patterns(message, cls._CONTEXT_LENGTH_PATTERNS):
                return LLMErrorType.CONTEXT_LENGTH_EXCEEDED
            return LLMErrorType.BAD_REQUEST

        elif status_code == 401:
            return LLMErrorType.AUTHENTICATION_ERROR

        elif status_code == 403:
            return LLMErrorType.PERMISSION_DENIED

        elif status_code == 422:
            return LLMErrorType.UNPROCESSABLE_ENTITY

        elif status_code == 429:
            if cls._matches_patterns(message, cls._RATE_LIMIT_PATTERNS):
                return LLMErrorType.RATE_LIMIT_ERROR
            return LLMErrorType.QUOTA_EXCEEDED

        elif 500 <= status_code < 600:
            if status_code == 503:
                return LLMErrorType.SERVICE_UNAVAILABLE
            elif status_code == 504:
                return LLMErrorType.GATEWAY_TIMEOUT
            return LLMErrorType.SERVER_ERROR

        return LLMErrorType.UNKNOWN_ERROR

    @classmethod
    def _classify_by_message(cls, message: str) -> LLMErrorType:
        """根据错误消息模式对错误进行分类。"""
        if cls._matches_patterns(message, cls._AUTH_PATTERNS):
            return LLMErrorType.AUTHENTICATION_ERROR

        if cls._matches_patterns(message, cls._CONTEXT_LENGTH_PATTERNS):
            return LLMErrorType.CONTEXT_LENGTH_EXCEEDED

        if cls._matches_patterns(message, cls._RATE_LIMIT_PATTERNS):
            return LLMErrorType.RATE_LIMIT_ERROR

        if cls._matches_patterns(message, cls._TIMEOUT_PATTERNS):
            return LLMErrorType.TIMEOUT_ERROR

        if cls._matches_patterns(message, cls._CONNECTION_PATTERNS):
            return LLMErrorType.CONNECTION_ERROR

        return LLMErrorType.UNKNOWN_ERROR

    @classmethod
    @lru_cache(maxsize=128)
    def _matches_patterns_cached(cls, _text: str, _patterns_hash: int) -> bool:  # noqa: ARG002, ARG003
        """缓存的模式匹配辅助函数（使用哈希存储模式）。"""
        return False  # 占位符 - 实际逻辑使用类级模式

    @classmethod
    def _matches_patterns(cls, text: str, patterns: list[re.Pattern[str]]) -> bool:
        """检查文本是否匹配任何模式。"""
        return any(p.search(text) for p in patterns)

    @classmethod
    @lru_cache(maxsize=32)
    def _extract_retry_after(cls, response_body: str) -> int | None:
        """从响应正文中提取 retry-after 值并缓存。"""
        if not response_body:
            return None

        # 尝试在 JSON 中查找 retry-after
        match = re.search(r'"retry_after"\s*:\s*(\d+)', response_body)
        if match:
            return int(match.group(1))

        # 尝试在 header 格式中查找 retry-after
        match = re.search(r"retry.?after[:\s]+(\d+)", response_body, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None


def calculate_exponential_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, exponential_base: float = 2.0, jitter: float = 0.5
) -> float:
    """计算带有指数退避和抖动的延迟。

    这比固定延迟更有效地恢复速率限制。

    Args:
        attempt: 当前尝试次数（从 0 开始）
        base_delay: 基础延迟秒数
        max_delay: 最大延迟上限
        exponential_base: 每次尝试的倍数
        jitter: 添加随机性的随机因子（0-1）

    Returns:
        下次重试前等待的延迟秒数
    """
    import random

    # 计算指数延迟
    delay = base_delay * (exponential_base**attempt)

    # 限制在最大延迟
    delay = min(delay, max_delay)

    # 添加抖动以防止雷鸣般的群体效应
    jitter_range = delay * jitter
    delay += random.uniform(-jitter_range, jitter_range)

    return max(0.1, delay)  # 至少 100ms


class RetryStrategy:
    """LLM API 调用的可配置重试策略。

    支持：
    - 带抖动的指数退避
    - 按错误类型配置
    - 最大重试限制
    """

    # 每个错误类型的默认重试配置
    DEFAULT_CONFIG = {
        "rate_limit": {"max_retries": 5, "base_delay": 2.0, "max_delay": 120.0},
        "server_error": {"max_retries": 3, "base_delay": 1.0, "max_delay": 30.0},
        "timeout": {"max_retries": 3, "base_delay": 1.0, "max_delay": 60.0},
        "network": {"max_retries": 5, "base_delay": 1.0, "max_delay": 60.0},
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self.DEFAULT_CONFIG

    def get_delay(self, error_type: str, attempt: int) -> float | None:
        """获取错误类型和尝试次数的重试延迟。

        Args:
            error_type: 错误类型（来自 LLMErrorType）
            attempt: 当前尝试次数

        Returns:
            延迟秒数，如果不应重试则返回 None
        """
        # 将错误类型规范化为配置键
        config_key = error_type.value.replace("_error", "") if hasattr(error_type, "value") else error_type

        if config_key not in self.config:
            config_key = "network"  # 默认回退

        cfg = self.config[config_key]
        if attempt >= cfg["max_retries"]:
            return None

        return calculate_exponential_backoff(
            attempt=attempt,
            base_delay=cfg["base_delay"],
            max_delay=cfg["max_delay"],
        )

    def should_retry(self, error_type: str, attempt: int) -> bool:
        """检查是否应该重试此错误类型。

        Args:
            error_type: 错误类型
            attempt: 当前尝试次数

        Returns:
            如果应该重试则返回 True，否则返回 False
        """
        return self.get_delay(error_type, attempt) is not None


def format_llm_error(error: Exception, status_code: int | None = None) -> str:
    """将 LLM 错误格式化为用户友好的消息。

    Args:
        error: 要格式化的异常
        status_code: HTTP 状态码（如果有）

    Returns:
        带指导的格式化错误消息
    """
    llm_error = LLMErrorClassifier.classify(error, status_code)

    parts = [
        f"{Colors.BRIGHT_RED}Error: {llm_error.message}{Colors.RESET}",
    ]

    if llm_error.status_code:
        parts.append(f"{Colors.DIM}Status Code: {llm_error.status_code}{Colors.RESET}")

    parts.append(f"{Colors.BRIGHT_YELLOW}Guidance: {llm_error.user_guidance}{Colors.RESET}")

    return "\n".join(parts)
