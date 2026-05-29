"""Token usage tracking and estimation for conversation history."""

import logging
from typing import Any

from ..schema import Message
from ..utils.token_utils import get_encoder

logger = logging.getLogger(__name__)

DEFAULT_ENCODING_NAME = "cl100k_base"


class TokenTracker:
    """Tracks token usage for message history with incremental estimation.

    Features:
    - Incremental token counting (only encodes new messages)
    - Cache invalidation on message changes
    - Fallback estimation when tiktoken is unavailable
    """

    def __init__(self, encoding_name: str = DEFAULT_ENCODING_NAME):
        self._encoding_name = encoding_name
        self._encoder_cache: dict[str, Any] = {}
        self._cached_token_count: int = 0
        self._cached_token_index: int = 0
        self._token_cache_version: int = 0

    def _get_cached_encoder(self, encoding_name: str = DEFAULT_ENCODING_NAME) -> Any:
        if encoding_name not in self._encoder_cache:
            self._encoder_cache[encoding_name] = get_encoder(encoding_name)
        return self._encoder_cache[encoding_name]

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Accurately calculate token count for message history.
        Uses incremental estimation: only encodes new messages since last check.
        """
        if self._cached_token_index >= len(messages):
            return self._cached_token_count

        try:
            encoding = self._get_cached_encoder(self._encoding_name)
        except Exception:
            logger.debug("tiktoken unavailable, using fallback estimation")
            return self._estimate_tokens_fallback(messages)

        new_tokens = 0
        for msg in messages[self._cached_token_index :]:
            content = msg.content
            if isinstance(content, str):
                new_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        new_tokens += len(encoding.encode(str(block)))

            if msg.thinking:
                new_tokens += len(encoding.encode(msg.thinking))

            if msg.tool_calls:
                new_tokens += len(encoding.encode(str(msg.tool_calls)))

            new_tokens += 4

        self._cached_token_count += new_tokens
        self._cached_token_index = len(messages)
        return self._cached_token_count

    def _estimate_tokens_fallback(self, messages: list[Message]) -> int:
        """Fallback estimation when tiktoken is unavailable.

        Uses different ratios for different content types:
        - English: ~4 chars per token (1/4)
        - Chinese: ~2 chars per token (1/2)
        - Mixed: weighted average
        """
        total_tokens = 0

        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                total_tokens += self._estimate_string_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_tokens += self._estimate_string_tokens(str(block))

            if msg.thinking:
                total_tokens += self._estimate_string_tokens(msg.thinking)

            if msg.tool_calls:
                total_tokens += self._estimate_string_tokens(str(msg.tool_calls))

            total_tokens += 4  # Message overhead

        return total_tokens

    def _estimate_string_tokens(self, text: str) -> int:
        """Estimate tokens for a string, handling mixed languages.

        Args:
            text: String to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Detect Chinese characters (CJK Unified Ideographs)
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿" or "㐀" <= c <= "䶿")
        total_chars = len(text)

        if total_chars == 0:
            return 0

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.5:
            # Mostly Chinese: ~2 chars per token
            return int(total_chars / 2)
        elif chinese_ratio > 0.1:
            # Mixed content: weighted average
            # English part: ~4 chars/token, Chinese part: ~2 chars/token
            english_chars = total_chars - chinese_chars
            return int(english_chars / 4) + int(chinese_chars / 2)
        else:
            # Mostly English: ~4 chars per token
            return int(total_chars / 4)

    def invalidate_cache(self) -> None:
        """Reset the token cache after message list changes."""
        self._cached_token_count = 0
        self._cached_token_index = 0
        self._token_cache_version += 1

    @property
    def cache_version(self) -> int:
        return self._token_cache_version

    @property
    def cached_count(self) -> int:
        return self._cached_token_count
