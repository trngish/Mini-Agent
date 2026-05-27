"""LLM API error handling and classification.

Provides detailed error classification and user-friendly error messages
for various API error types.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from typing import Any

from .display import Colors


class LLMErrorType(str, Enum):
    """LLM API error types."""

    # Authentication & Authorization
    AUTHENTICATION_ERROR = "authentication_error"  # 401
    PERMISSION_DENIED = "permission_denied"  # 403

    # Rate Limiting
    RATE_LIMIT_ERROR = "rate_limit_error"  # 429
    QUOTA_EXCEEDED = "quota_exceeded"  # 429 with specific message

    # Server Errors
    SERVER_ERROR = "server_error"  # 500-599
    SERVICE_UNAVAILABLE = "service_unavailable"  # 503
    GATEWAY_TIMEOUT = "gateway_timeout"  # 504

    # Client Errors
    BAD_REQUEST = "bad_request"  # 400
    INVALID_REQUEST = "invalid_request"  # 400 with specific message
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"  # 400 with specific message
    UNPROCESSABLE_ENTITY = "unprocessable_entity"  # 422

    # Network Errors
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    CONNECTION_ERROR = "connection_error"

    # Unknown
    UNKNOWN_ERROR = "unknown_error"


class LLMError(Exception):
    """Base exception for LLM errors with error classification."""

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
        self.retry_after = retry_after  # Seconds to wait before retry (for rate limit)
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
        """Check if this error type should be retried."""
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
        """Get user guidance for this error type."""
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
    """Classifies LLM API errors from exception objects or HTTP responses."""

    # Pattern matchers for specific error messages (compiled once for performance)
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
        """Classify an exception and return an LLMError.

        Args:
            error: The original exception
            status_code: HTTP status code if available
            response_body: Response body text if available

        Returns:
            LLMError with classified error type
        """
        error_str = str(error)
        body_str = response_body or ""

        # Combine for pattern matching
        combined_text = f"{error_str} {body_str}"

        # Classify by status code first
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

        # Classify by error message patterns
        error_type = cls._classify_by_message(combined_text)
        return LLMError(
            message=error_str,
            error_type=error_type,
            status_code=status_code,
            details=body_str[:500] if body_str else None,
        )

    @classmethod
    def _classify_by_status(cls, status_code: int, message: str) -> LLMErrorType:
        """Classify error by HTTP status code."""
        if status_code == 400:
            # Check for specific 400 errors
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
        """Classify error by error message patterns."""
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
        """Cached pattern matching helper (uses hash to store patterns)."""
        return False  # Placeholder - actual logic uses class-level patterns

    @classmethod
    def _matches_patterns(cls, text: str, patterns: list[re.Pattern[str]]) -> bool:
        """Check if text matches any of the patterns."""
        return any(p.search(text) for p in patterns)

    @classmethod
    @lru_cache(maxsize=32)
    def _extract_retry_after(cls, response_body: str) -> int | None:
        """Extract retry-after value from response body with caching."""
        if not response_body:
            return None

        # Try to find retry-after in JSON
        match = re.search(r'"retry_after"\s*:\s*(\d+)', response_body)
        if match:
            return int(match.group(1))

        # Try to find retry-after in header format
        match = re.search(r"retry.?after[:\s]+(\d+)", response_body, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None


def calculate_exponential_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, exponential_base: float = 2.0, jitter: float = 0.5
) -> float:
    """Calculate delay with exponential backoff and jitter.

    This is more effective for rate limit recovery than fixed delays.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Multiplier for each attempt
        jitter: Random factor (0-1) to add randomness

    Returns:
        Delay in seconds to wait before next retry
    """
    import random

    # Calculate exponential delay
    delay = base_delay * (exponential_base**attempt)

    # Cap at max delay
    delay = min(delay, max_delay)

    # Add jitter to prevent thundering herd
    jitter_range = delay * jitter
    delay += random.uniform(-jitter_range, jitter_range)

    return max(0.1, delay)  # At least 100ms


class RetryStrategy:
    """Configurable retry strategy for LLM API calls.

    Supports:
    - Exponential backoff with jitter
    - Per-error-type configuration
    - Maximum retry limits
    """

    # Default retry configuration per error type
    DEFAULT_CONFIG = {
        "rate_limit": {"max_retries": 5, "base_delay": 2.0, "max_delay": 120.0},
        "server_error": {"max_retries": 3, "base_delay": 1.0, "max_delay": 30.0},
        "timeout": {"max_retries": 3, "base_delay": 1.0, "max_delay": 60.0},
        "network": {"max_retries": 5, "base_delay": 1.0, "max_delay": 60.0},
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self.DEFAULT_CONFIG

    def get_delay(self, error_type: str, attempt: int) -> float | None:
        """Get retry delay for error type and attempt.

        Args:
            error_type: Type of error (from LLMErrorType)
            attempt: Current attempt number

        Returns:
            Delay in seconds, or None if should not retry
        """
        # Normalize error type to config key
        config_key = error_type.value.replace("_error", "") if hasattr(error_type, "value") else error_type

        if config_key not in self.config:
            config_key = "network"  # Default fallback

        cfg = self.config[config_key]
        if attempt >= cfg["max_retries"]:
            return None

        return calculate_exponential_backoff(
            attempt=attempt,
            base_delay=cfg["base_delay"],
            max_delay=cfg["max_delay"],
        )

    def should_retry(self, error_type: str, attempt: int) -> bool:
        """Check if should retry this error type.

        Args:
            error_type: Type of error
            attempt: Current attempt number

        Returns:
            True if should retry, False otherwise
        """
        return self.get_delay(error_type, attempt) is not None


def format_llm_error(error: Exception, status_code: int | None = None) -> str:
    """Format an LLM error into a user-friendly message.

    Args:
        error: The exception to format
        status_code: HTTP status code if available

    Returns:
        Formatted error message with guidance
    """
    llm_error = LLMErrorClassifier.classify(error, status_code)

    parts = [
        f"{Colors.BRIGHT_RED}Error: {llm_error.message}{Colors.RESET}",
    ]

    if llm_error.status_code:
        parts.append(f"{Colors.DIM}Status Code: {llm_error.status_code}{Colors.RESET}")

    parts.append(f"{Colors.BRIGHT_YELLOW}Guidance: {llm_error.user_guidance}{Colors.RESET}")

    return "\n".join(parts)
