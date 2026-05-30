"""终端显示工具，用于正确的文本对齐。

此模块提供用于计算文本在终端中可见宽度的工具，
能正确处理ANSI转义码、emoji以及东亚语言字符。
"""

import re
import unicodedata

# 为提升性能，在模块级别编译一次正则表达式
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Emoji的Unicode范围
EMOJI_START = 0x1F300
EMOJI_END = 0x1FAFF


def calculate_display_width(text: str) -> int:
    """计算文本在终端列中的可见宽度。

    此函数能正确处理：
    - ANSI转义码（不计入宽度）
    - Emoji字符（计为2列）
    - 东亚宽/全角字符（计为2列）
    - 组合字符（计为0列）
    - 常规ASCII字符（计为1列）

    Args:
        text: 输入文本，可能包含ANSI码、emoji或unicode字符

    Returns:
        文本显示时占据的终端列数

    Examples:
        >>> calculate_display_width("Hello")
        5
        >>> calculate_display_width("World")
        4
        >>> calculate_display_width("🤖")
        2
        >>> calculate_display_width("\033[31mRed\033[0m")
        3
    """
    # 移除ANSI转义码（它们不占据显示空间）
    clean_text = ANSI_ESCAPE_RE.sub("", text)

    width = 0
    for char in clean_text:
        # 跳过组合字符（零宽度）
        if unicodedata.combining(char):
            continue

        code_point = ord(char)

        # Emoji范围（最常见的emoji，计为2列）
        if EMOJI_START <= code_point <= EMOJI_END:
            width += 2
            continue

        # 东亚宽度属性
        # W = 宽字符，F = 全角字符（均占据2列）
        eaw = unicodedata.east_asian_width(char)
        if eaw in ("W", "F"):
            width += 2
        else:
            width += 1

    return width


def truncate_with_ellipsis(text: str, max_width: int, ellipsis: str = "…") -> str:
    """截断文本以适应max_width，必要时添加省略号。

    Args:
        text: 要截断的文本（ANSI码会被保留但不计宽度）
        max_width: 最大可见宽度（终端列数）
        ellipsis: 省略号字符（默认："…"）

    Returns:
        截断后的文本，必要时添加省略号

    Examples:
        >>> truncate_with_ellipsis("Hello World", 8)
        'Hello W…'
        >>> truncate_with_ellipsis("Good morning", 5)
        'Good…'
    """
    if max_width <= 0:
        return ""

    current_width = calculate_display_width(text)

    # 无需截断
    if current_width <= max_width:
        return text

    # 截断时移除ANSI码（会失去颜色，但这是预期的）
    plain_text = ANSI_ESCAPE_RE.sub("", text)

    # 如果max_width对于省略号来说太小
    ellipsis_width = calculate_display_width(ellipsis)
    if max_width <= ellipsis_width:
        return plain_text[:max_width]

    # 找到截断点
    available_width = max_width - ellipsis_width
    truncated = ""
    current_width = 0

    for char in plain_text:
        char_width = calculate_display_width(char)
        if current_width + char_width > available_width:
            break
        truncated += char
        current_width += char_width

    return truncated + ellipsis


def pad_to_width(text: str, target_width: int, align: str = "left", fill_char: str = " ") -> str:
    """填充文本至目标宽度，支持多种对齐方式。

    Args:
        text: 要填充的文本（可能包含ANSI码）
        target_width: 目标宽度（终端列数）
        align: 对齐模式 - "left"、"right"或"center"
        fill_char: 用于填充的字符（默认：空格）

    Returns:
        填充后的文本

    Examples:
        >>> pad_to_width("Hello", 10)
        'Hello     '
        >>> pad_to_width("World", 10)
        'World     '
        >>> pad_to_width("Test", 10, align="center")
        '   Test   '
    """
    current_width = calculate_display_width(text)

    if current_width >= target_width:
        return text

    padding_needed = target_width - current_width

    if align == "left":
        return text + (fill_char * padding_needed)
    elif align == "right":
        return (fill_char * padding_needed) + text
    elif align == "center":
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        return (fill_char * left_padding) + text + (fill_char * right_padding)
    else:
        raise ValueError(f"Invalid align value: {align}. Must be 'left', 'right', or 'center'")
