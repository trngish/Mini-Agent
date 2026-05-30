#!/usr/bin/env python3
"""
配色方案 - 专业、和谐的GIF色彩方案。

使用一致、设计精良的配色方案可以让GIF看起来更加专业和精致，
而不是随机和业余的。
"""

import colorsys

# 专业配色方案 - 精选用于GIF压缩和视觉吸引

VIBRANT = {
    "primary": (255, 68, 68),  # 亮红色
    "secondary": (255, 168, 0),  # 亮橙色
    "accent": (0, 168, 255),  # 亮蓝色
    "success": (68, 255, 68),  # 亮绿色
    "background": (240, 248, 255),  # 爱丽丝蓝
    "text": (30, 30, 30),  # 近黑色
    "text_light": (255, 255, 255),  # 白色
}

PASTEL = {
    "primary": (255, 179, 186),  # 粉红色
    "secondary": (255, 223, 186),  # 桃色
    "accent": (186, 225, 255),  # 淡蓝色
    "success": (186, 255, 201),  # 淡绿色
    "background": (255, 250, 240),  # 花白色
    "text": (80, 80, 80),  # 深灰色
    "text_light": (255, 255, 255),  # 白色
}

DARK = {
    "primary": (255, 100, 100),  # 柔和红色
    "secondary": (100, 200, 255),  # 柔和蓝色
    "accent": (255, 200, 100),  # 柔和金色
    "success": (100, 255, 150),  # 柔和绿色
    "background": (30, 30, 35),  # 近黑色
    "text": (220, 220, 220),  # 浅灰色
    "text_light": (255, 255, 255),  # 白色
}

NEON = {
    "primary": (255, 16, 240),  # 霓虹粉
    "secondary": (0, 255, 255),  # 青色
    "accent": (255, 255, 0),  # 黄色
    "success": (57, 255, 20),  # 霓虹绿
    "background": (20, 20, 30),  # 深蓝黑
    "text": (255, 255, 255),  # 白色
    "text_light": (255, 255, 255),  # 白色
}

PROFESSIONAL = {
    "primary": (0, 122, 255),  # 系统蓝
    "secondary": (88, 86, 214),  # 系统紫
    "accent": (255, 149, 0),  # 系统橙
    "success": (52, 199, 89),  # 系统绿
    "background": (255, 255, 255),  # 白色
    "text": (0, 0, 0),  # 黑色
    "text_light": (255, 255, 255),  # 白色
}

WARM = {
    "primary": (255, 107, 107),  # 珊瑚红
    "secondary": (255, 159, 64),  # 橙色
    "accent": (255, 218, 121),  # 黄色
    "success": (106, 176, 76),  # 橄榄绿
    "background": (255, 246, 229),  # 暖白色
    "text": (51, 51, 51),  # 炭灰色
    "text_light": (255, 255, 255),  # 白色
}

COOL = {
    "primary": (107, 185, 240),  # 天蓝色
    "secondary": (130, 202, 157),  # 薄荷绿
    "accent": (162, 155, 254),  # 薰衣草紫
    "success": (86, 217, 150),  # 浅绿松石色
    "background": (240, 248, 255),  # 爱丽丝蓝
    "text": (45, 55, 72),  # 深蓝灰色
    "text_light": (255, 255, 255),  # 白色
}

MONOCHROME = {
    "primary": (80, 80, 80),  # 深灰色
    "secondary": (130, 130, 130),  # 中灰色
    "accent": (180, 180, 180),  # 浅灰色
    "success": (100, 100, 100),  # 灰色
    "background": (245, 245, 245),  # 灰白色
    "text": (30, 30, 30),  # 近黑色
    "text_light": (255, 255, 255),  # 白色
}

# 配色方案名称映射
PALETTES = {
    "vibrant": VIBRANT,
    "pastel": PASTEL,
    "dark": DARK,
    "neon": NEON,
    "professional": PROFESSIONAL,
    "warm": WARM,
    "cool": COOL,
    "monochrome": MONOCHROME,
}


def get_palette(name: str = "vibrant") -> dict:
    """
    根据名称获取配色方案。

    Args:
        name: 配色方案名称 (vibrant, pastel, dark, neon, professional, warm, cool, monochrome)

    Returns:
        颜色角色到RGB元组的字典
    """
    return PALETTES.get(name.lower(), VIBRANT)


def get_text_color_for_background(bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    获取给定背景的最佳文本颜色（黑色或白色）。

    使用亮度计算确保可读性。

    Args:
        bg_color: 背景RGB颜色

    Returns:
        对比度良好的文本颜色（黑色或白色）
    """
    # 计算相对亮度
    r, g, b = bg_color
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    # 浅色背景返回黑色，深色背景返回白色
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)


def get_complementary_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    获取色轮上互补（相对）的颜色。

    Args:
        color: RGB颜色元组

    Returns:
        互补的RGB颜色
    """
    # 转换为HSV
    r, g, b = [x / 255.0 for x in color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # 将色相旋转180度（0-1尺度上为0.5）
    h_comp = (h + 0.5) % 1.0

    # 转换回RGB
    r_comp, g_comp, b_comp = colorsys.hsv_to_rgb(h_comp, s, v)
    return (int(r_comp * 255), int(g_comp * 255), int(b_comp * 255))


def lighten_color(color: tuple[int, int, int], amount: float = 0.3) -> tuple[int, int, int]:
    """
    按指定量提亮颜色。

    Args:
        color: RGB颜色元组
        amount: 提亮量 (0.0-1.0)

    Returns:
        提亮后的RGB颜色
    """
    r, g, b = color
    r = min(255, int(r + (255 - r) * amount))
    g = min(255, int(g + (255 - g) * amount))
    b = min(255, int(b + (255 - b) * amount))
    return (r, g, b)


def darken_color(color: tuple[int, int, int], amount: float = 0.3) -> tuple[int, int, int]:
    """
    按指定量加深颜色。

    Args:
        color: RGB颜色元组
        amount: 加深量 (0.0-1.0)

    Returns:
        加深后的RGB颜色
    """
    r, g, b = color
    r = max(0, int(r * (1 - amount)))
    g = max(0, int(g * (1 - amount)))
    b = max(0, int(b * (1 - amount)))
    return (r, g, b)


def blend_colors(
    color1: tuple[int, int, int], color2: tuple[int, int, int], ratio: float = 0.5
) -> tuple[int, int, int]:
    """
    混合两种颜色。

    Args:
        color1: 第一个RGB颜色
        color2: 第二个RGB颜色
        ratio: 混合比例 (0.0 = 全部color1, 1.0 = 全部color2)

    Returns:
        混合后的RGB颜色
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    r = int(r1 * (1 - ratio) + r2 * ratio)
    g = int(g1 * (1 - ratio) + g2 * ratio)
    b = int(b1 * (1 - ratio) + b2 * ratio)

    return (r, g, b)


def create_gradient_colors(
    start_color: tuple[int, int, int], end_color: tuple[int, int, int], steps: int
) -> list[tuple[int, int, int]]:
    """
    在两种颜色之间创建渐变。

    Args:
        start_color: 起始RGB颜色
        end_color: 结束RGB颜色
        steps: 渐变步数

    Returns:
        形成渐变的RGB颜色列表
    """
    colors = []
    for i in range(steps):
        ratio = i / (steps - 1) if steps > 1 else 0
        colors.append(blend_colors(start_color, end_color, ratio))
    return colors


# 跨配色方案效果好的冲击/强调颜色
IMPACT_COLORS = {
    "flash": (255, 255, 240),  # 明亮闪光（米色）
    "explosion": (255, 150, 0),  # 橙色爆炸
    "electricity": (100, 200, 255),  # 电蓝色
    "fire": (255, 100, 0),  # 火橙色
    "success": (50, 255, 100),  # 成功绿色
    "error": (255, 50, 50),  # 错误红色
    "warning": (255, 200, 0),  # 警告黄色
    "magic": (200, 100, 255),  # 魔法紫色
}


def get_impact_color(effect_type: str = "flash") -> tuple[int, int, int]:
    """
    获取冲击/强调效果的颜色。

    Args:
        effect_type: 效果类型 (flash, explosion, electricity 等)

    Returns:
        效果对应的RGB颜色
    """
    return IMPACT_COLORS.get(effect_type, IMPACT_COLORS["flash"])


# 兼容表情符号的配色方案（适用于128x128、32-64色的场景）
EMOJI_PALETTES = {
    "simple": [
        (255, 255, 255),  # 白色
        (0, 0, 0),  # 黑色
        (255, 100, 100),  # 红色
        (100, 255, 100),  # 绿色
        (100, 100, 255),  # 蓝色
        (255, 255, 100),  # 黄色
    ],
    "vibrant_emoji": [
        (255, 255, 255),  # 白色
        (30, 30, 30),  # 黑色
        (255, 68, 68),  # 红色
        (68, 255, 68),  # 绿色
        (68, 68, 255),  # 蓝色
        (255, 200, 68),  # 金色
        (255, 68, 200),  # 粉色
        (68, 255, 200),  # 青色
    ],
}


def get_emoji_palette(name: str = "simple") -> list[tuple[int, int, int]]:
    """
    获取专为表情符号GIF优化的有限配色方案（<64KB）。

    Args:
        name: 配色方案名称 (simple, vibrant_emoji)

    Returns:
        RGB颜色列表（6-8种颜色）
    """
    return EMOJI_PALETTES.get(name, EMOJI_PALETTES["simple"])
