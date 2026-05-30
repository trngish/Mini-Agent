"""令牌编码工具，带缓存以优化性能。

本模块提供 tiktoken 编码器的缓存访问，避免
重复初始化开销。
"""

from functools import lru_cache
from typing import Any

import tiktoken

# 全局编码器缓存
_encoders_cache: dict[str, tiktoken.Encoding] = {}


def get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """获取或创建缓存的 tiktoken 编码器。

    编码器会被缓存以避免重复初始化开销，
    这在 token 化大量文本时可能非常显著。

    Args:
        encoding_name: 编码名称（默认为 "cl100k_base"，用于 GPT-4/Claude/M2）

    Returns:
        缓存的 tiktoken Encoding 实例

    示例:
        >>> encoder = get_encoder()
        >>> tokens = encoder.encode("Hello, world!")
        >>> len(tokens)
        5
    """
    if encoding_name not in _encoders_cache:
        _encoders_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoders_cache[encoding_name]


def encode_text(text: str, encoding_name: str = "cl100k_base") -> list[int]:
    """使用缓存的编码器将文本编码为 token ID。

    Args:
        text: 要编码的文本
        encoding_name: 要使用的编码名称

    Returns:
        token ID 列表
    """
    encoder = get_encoder(encoding_name)
    return encoder.encode(text)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """计算文本中的 token 数量。

    Args:
        text: 要计算 token 的文本
        encoding_name: 要使用的编码名称

    Returns:
        文本中的 token 数量
    """
    encoder = get_encoder(encoding_name)
    return len(encoder.encode(text))


def decode_tokens(token_ids: list[int], encoding_name: str = "cl100k_base") -> str:
    """将 token ID 解码回文本。

    Args:
        token_ids: token ID 列表
        encoding_name: 要使用的编码名称

    Returns:
        解码后的文本字符串
    """
    encoder = get_encoder(encoding_name)
    return encoder.decode(token_ids)


def get_tokens_info(text: str, encoding_name: str = "cl100k_base") -> dict[str, Any]:
    """获取文本的详细 token 信息。

    Args:
        text: 要分析的文本
        encoding_name: 要使用的编码名称

    Returns:
        包含 token 数量和 token ID 的字典
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
    """清除编码器缓存以释放内存。

    当你完成大量文本的 token 化后调用此函数
    可以释放内存。
    """
    global _encoders_cache
    _encoders_cache.clear()


def get_cache_size() -> int:
    """获取缓存的编码器数量。

    Returns:
        当前缓存中的编码器数量
    """
    return len(_encoders_cache)


# 预加载常用编码器以加快首次使用速度
@lru_cache(maxsize=1)
def get_cl100k_base() -> tiktoken.Encoding:
    """获取 cl100k_base 编码器（兼容 GPT-4/Claude/M2）。

    这是最常用的编码器，已被预缓存
    以加快访问速度。

    Returns:
        cl100k_base 的 tiktoken Encoding
    """
    return get_encoder("cl100k_base")


@lru_cache(maxsize=1)
def get_o200k_base() -> tiktoken.Encoding:
    """获取 o200k_base 编码器（兼容 GPT-4o）。

    Returns:
        o200k_base 的 tiktoken Encoding
    """
    return get_encoder("o200k_base")


# 后台预加载的延迟初始化函数
def preload_encoders() -> None:
    """在后台预加载常用编码器。

    在应用程序启动时调用此函数以确保编码器
    在首次需要时已准备就绪，避免首次调用延迟。
    """
    # 预加载 cl100k_base（最常用）
    get_cl100k_base()
