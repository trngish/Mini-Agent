"""Token encoding utilities with caching for performance optimization.

This module provides cached access to tiktoken encoders to avoid
repeated initialization overhead.
"""

from functools import lru_cache
from typing import Any

import tiktoken

# Global encoder cache
_encoders_cache: dict[str, tiktoken.Encoding] = {}


def get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get or create a cached tiktoken encoder.

    Encoders are cached to avoid repeated initialization overhead,
    which can be significant when tokenizing many texts.

    Args:
        encoding_name: Name of the encoding (default: "cl100k_base" for GPT-4/Claude/M2)

    Returns:
        Cached tiktoken Encoding instance

    Examples:
        >>> encoder = get_encoder()
        >>> tokens = encoder.encode("Hello, world!")
        >>> len(tokens)
        5
    """
    if encoding_name not in _encoders_cache:
        _encoders_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoders_cache[encoding_name]


def encode_text(text: str, encoding_name: str = "cl100k_base") -> list[int]:
    """Encode text to token IDs using cached encoder.

    Args:
        text: Text to encode
        encoding_name: Name of the encoding to use

    Returns:
        List of token IDs
    """
    encoder = get_encoder(encoding_name)
    return encoder.encode(text)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in text.

    Args:
        text: Text to count tokens for
        encoding_name: Name of the encoding to use

    Returns:
        Number of tokens in the text
    """
    encoder = get_encoder(encoding_name)
    return len(encoder.encode(text))


def decode_tokens(token_ids: list[int], encoding_name: str = "cl100k_base") -> str:
    """Decode token IDs back to text.

    Args:
        token_ids: List of token IDs
        encoding_name: Name of the encoding to use

    Returns:
        Decoded text string
    """
    encoder = get_encoder(encoding_name)
    return encoder.decode(token_ids)


def get_tokens_info(text: str, encoding_name: str = "cl100k_base") -> dict[str, Any]:
    """Get detailed token information for text.

    Args:
        text: Text to analyze
        encoding_name: Name of the encoding to use

    Returns:
        Dictionary with token count and token IDs
    """
    encoder = get_encoder(encoding_name)
    token_ids = encoder.encode(text)
    return {
        "text": text,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "encoding": encoding_name,
    }


def clear_encoder_cache() -> None:
    """Clear the encoder cache to free memory.

    Use this when you've finished tokenizing large amounts of text
    and want to free up memory.
    """
    global _encoders_cache
    _encoders_cache.clear()


def get_cache_size() -> int:
    """Get the number of cached encoders.

    Returns:
        Number of encoders currently in cache
    """
    return len(_encoders_cache)


# Pre-load common encoders for faster first use
@lru_cache(maxsize=1)
def get_cl100k_base() -> tiktoken.Encoding:
    """Get cl100k_base encoder (GPT-4/Claude/M2 compatible).

    This is the most commonly used encoder and is pre-cached
    for faster access.

    Returns:
        tiktoken Encoding for cl100k_base
    """
    return get_encoder("cl100k_base")


@lru_cache(maxsize=1)
def get_o200k_base() -> tiktoken.Encoding:
    """Get o200k_base encoder (GPT-4o compatible).

    Returns:
        tiktoken Encoding for o200k_base
    """
    return get_encoder("o200k_base")


# Lazy initialization function for background pre-loading
def preload_encoders() -> None:
    """Pre-load common encoders in the background.

    Call this during application startup to ensure encoders
    are ready when first needed, avoiding first-call latency.
    """
    # Pre-load cl100k_base (most common)
    get_cl100k_base()
