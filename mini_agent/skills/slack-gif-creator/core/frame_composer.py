#!/usr/bin/env python3
"""
帧合成器 - 将视觉元素合成到帧中的工具。

提供用于绘制形状、文本、表情符号和将元素合成在一起来创建动画帧的函数。
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_blank_frame(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    创建纯色背景的空白帧。

    Args:
        width: 帧宽度
        height: 帧高度
        color: RGB颜色元组（默认：白色）

    Returns:
        PIL图像
    """
    return Image.new("RGB", (width, height), color)


def draw_circle(
    frame: Image.Image,
    center: tuple[int, int],
    radius: int,
    fill_color: tuple[int, int, int] | None = None,
    outline_color: tuple[int, int, int] | None = None,
    outline_width: int = 1,
) -> Image.Image:
    """
    在帧上绘制圆形。

    Args:
        frame: 要绘制的PIL图像
        center: (x, y) 中心位置
        radius: 圆半径
        fill_color: RGB填充颜色（None表示无填充）
        outline_color: RGB轮廓颜色（None表示无轮廓）
        outline_width: 轮廓宽度（像素）

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=outline_width)
    return frame


def draw_rectangle(
    frame: Image.Image,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    fill_color: tuple[int, int, int] | None = None,
    outline_color: tuple[int, int, int] | None = None,
    outline_width: int = 1,
) -> Image.Image:
    """
    在帧上绘制矩形。

    Args:
        frame: 要绘制的PIL图像
        top_left: (x, y) 左上角
        bottom_right: (x, y) 右下角
        fill_color: RGB填充颜色（None表示无填充）
        outline_color: RGB轮廓颜色（None表示无轮廓）
        outline_width: 轮廓宽度（像素）

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    draw.rectangle([top_left, bottom_right], fill=fill_color, outline=outline_color, width=outline_width)
    return frame


def draw_line(
    frame: Image.Image,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 2,
) -> Image.Image:
    """
    在帧上绘制线条。

    Args:
        frame: 要绘制的PIL图像
        start: (x, y) 起始位置
        end: (x, y) 结束位置
        color: RGB线条颜色
        width: 线条宽度（像素）

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    draw.line([start, end], fill=color, width=width)
    return frame


def draw_text(
    frame: Image.Image,
    text: str,
    position: tuple[int, int],
    font_size: int = 40,
    color: tuple[int, int, int] = (0, 0, 0),
    centered: bool = False,
) -> Image.Image:
    """
    在帧上绘制文本。

    Args:
        frame: 要绘制的PIL图像
        text: 要绘制的文本
        position: (x, y) 位置（除非centered=True，否则为左上角）
        font_size: 字体大小（像素）
        color: RGB文本颜色
        centered: 如果为True，在位置处居中文本

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)

    # 尝试使用默认字体，如果不可用则回退到基本字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    if centered:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = position[0] - text_width // 2
        y = position[1] - text_height // 2
        position = (x, y)

    draw.text(position, text, fill=color, font=font)
    return frame


def draw_emoji(frame: Image.Image, emoji: str, position: tuple[int, int], size: int = 60) -> Image.Image:
    """
    在帧上绘制表情符号文本（需要系统表情符号支持）。

    Args:
        frame: 要绘制的PIL图像
        emoji: 表情符号字符
        position: (x, y) 位置
        size: 表情符号大小（像素）

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)

    # 在macOS上使用Apple Color Emoji字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", size)
    except:
        # 回退到基于文本的表情符号
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)

    draw.text(position, emoji, font=font, embedded_color=True)
    return frame


def composite_layers(
    base: Image.Image, overlay: Image.Image, position: tuple[int, int] = (0, 0), alpha: float = 1.0
) -> Image.Image:
    """
    将一幅图像合成到另一幅图像上方。

    Args:
        base: 基础图像
        overlay: 要叠加在最上方的图像
        position: 放置叠加图像的 (x, y) 位置
        alpha: 叠加的不透明度 (0.0 = 透明, 1.0 = 不透明)

    Returns:
        合成后的图像
    """
    # 转换为RGBA以支持透明度
    base_rgba = base.convert("RGBA")
    overlay_rgba = overlay.convert("RGBA")

    # 应用alpha
    if alpha < 1.0:
        overlay_rgba = overlay_rgba.copy()
        overlay_rgba.putalpha(int(255 * alpha))

    # 将叠加图像粘贴到基础图像上
    base_rgba.paste(overlay_rgba, position, overlay_rgba)

    # 转换回RGB
    return base_rgba.convert("RGB")


def draw_stick_figure(
    frame: Image.Image,
    position: tuple[int, int],
    scale: float = 1.0,
    color: tuple[int, int, int] = (0, 0, 0),
    line_width: int = 3,
) -> Image.Image:
    """
    绘制简单的火柴人。

    Args:
        frame: 要绘制的PIL图像
        position: (x, y) 头部中心位置
        scale: 大小倍数
        color: RGB线条颜色
        line_width: 线条宽度（像素）

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    x, y = position

    # 缩放尺寸
    head_radius = int(15 * scale)
    body_length = int(40 * scale)
    arm_length = int(25 * scale)
    leg_length = int(35 * scale)
    leg_spread = int(15 * scale)

    # 头部
    draw.ellipse([x - head_radius, y - head_radius, x + head_radius, y + head_radius], outline=color, width=line_width)

    # 身体
    body_start = y + head_radius
    body_end = body_start + body_length
    draw.line([(x, body_start), (x, body_end)], fill=color, width=line_width)

    # 手臂
    arm_y = body_start + int(body_length * 0.3)
    draw.line([(x - arm_length, arm_y), (x + arm_length, arm_y)], fill=color, width=line_width)

    # 腿
    draw.line([(x, body_end), (x - leg_spread, body_end + leg_length)], fill=color, width=line_width)
    draw.line([(x, body_end), (x + leg_spread, body_end + leg_length)], fill=color, width=line_width)

    return frame


def create_gradient_background(
    width: int, height: int, top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]
) -> Image.Image:
    """
    创建垂直渐变背景。

    Args:
        width: 帧宽度
        height: 帧高度
        top_color: 顶部的RGB颜色
        bottom_color: 底部的RGB颜色

    Returns:
        带渐变的PIL图像
    """
    frame = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(frame)

    # 计算每行的颜色步进
    r1, g1, b1 = top_color
    r2, g2, b2 = bottom_color

    for y in range(height):
        # 插值颜色
        ratio = y / height
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)

        # 绘制水平线
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return frame


def draw_emoji_enhanced(
    frame: Image.Image,
    emoji: str,
    position: tuple[int, int],
    size: int = 60,
    shadow: bool = True,
    shadow_offset: tuple[int, int] = (2, 2),
) -> Image.Image:
    """
    绘制带有可选阴影的表情符号以获得更好的视觉效果。

    Args:
        frame: 要绘制的PIL图像
        emoji: 表情符号字符
        position: (x, y) 位置
        size: 表情符号大小（像素，最小12）
        shadow: 是否添加投影
        shadow_offset: 阴影偏移

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)

    # 确保最小尺寸以避免字体渲染错误
    size = max(12, size)

    # 在macOS上使用Apple Color Emoji字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", size)
    except:
        # 回退到基于文本的表情符号
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except:
            font = ImageFont.load_default()

    # 如果启用则首先绘制阴影
    if shadow and size >= 20:  # 仅对较大的表情符号绘制阴影
        shadow_pos = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
        # 通过多次绘制来模拟半透明阴影
        for offset in range(1, 3):
            try:
                draw.text(
                    (shadow_pos[0] + offset, shadow_pos[1] + offset),
                    emoji,
                    font=font,
                    embedded_color=True,
                    fill=(0, 0, 0, 100),
                )
            except:
                pass  # 如果阴影失败则跳过

    # 绘制主表情符号
    try:
        draw.text(position, emoji, font=font, embedded_color=True)
    except:
        # 如果嵌入颜色失败则回退到基本绘制
        draw.text(position, emoji, font=font, fill=(0, 0, 0))

    return frame


def draw_circle_with_shadow(
    frame: Image.Image,
    center: tuple[int, int],
    radius: int,
    fill_color: tuple[int, int, int],
    shadow_offset: tuple[int, int] = (3, 3),
    shadow_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    绘制带投影的圆形。

    Args:
        frame: 要绘制的PIL图像
        center: (x, y) 中心位置
        radius: 圆半径
        fill_color: RGB填充颜色
        shadow_offset: (x, y) 阴影偏移
        shadow_color: RGB阴影颜色

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    x, y = center

    # 绘制阴影
    shadow_center = (x + shadow_offset[0], y + shadow_offset[1])
    shadow_bbox = [
        shadow_center[0] - radius,
        shadow_center[1] - radius,
        shadow_center[0] + radius,
        shadow_center[1] + radius,
    ]
    draw.ellipse(shadow_bbox, fill=shadow_color)

    # 绘制主圆形
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill_color)

    return frame


def draw_rounded_rectangle(
    frame: Image.Image,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    radius: int,
    fill_color: tuple[int, int, int] | None = None,
    outline_color: tuple[int, int, int] | None = None,
    outline_width: int = 1,
) -> Image.Image:
    """
    绘制圆角矩形。

    Args:
        frame: 要绘制的PIL图像
        top_left: (x, y) 左上角
        bottom_right: (x, y) 右下角
        radius: 角半径
        fill_color: RGB填充颜色（None表示无填充）
        outline_color: RGB轮廓颜色（None表示无轮廓）
        outline_width: 轮廓宽度

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    x1, y1 = top_left
    x2, y2 = bottom_right

    # 使用PIL内置方法绘制圆角矩形
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill_color, outline=outline_color, width=outline_width)

    return frame


def add_vignette(frame: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    为帧添加晕影效果（边缘变暗）。

    Args:
        frame: PIL图像
        strength: 晕影强度 (0.0-1.0)

    Returns:
        带晕影的帧
    """
    width, height = frame.size

    # 创建径向渐变遮罩
    center_x, center_y = width // 2, height // 2
    max_dist = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5

    # 创建叠加层
    overlay = Image.new("RGB", (width, height), (0, 0, 0))
    pixels = overlay.load()

    for y in range(height):
        for x in range(width):
            # 计算到中心的距离
            dx = x - center_x
            dy = y - center_y
            dist = (dx**2 + dy**2) ** 0.5

            # 计算晕影值
            vignette = min(1, (dist / max_dist) * strength)
            value = int(255 * (1 - vignette))
            pixels[x, y] = (value, value, value)

    # 使用正片叠底与原始图像混合
    frame_array = np.array(frame, dtype=np.float32) / 255
    overlay_array = np.array(overlay, dtype=np.float32) / 255

    result = frame_array * overlay_array
    result = (result * 255).astype(np.uint8)

    return Image.fromarray(result)


def draw_star(
    frame: Image.Image,
    center: tuple[int, int],
    size: int,
    fill_color: tuple[int, int, int],
    outline_color: tuple[int, int, int] | None = None,
    outline_width: int = 1,
) -> Image.Image:
    """
    绘制五角星。

    Args:
        frame: 要绘制的PIL图像
        center: (x, y) 中心位置
        size: 星星大小（外半径）
        fill_color: RGB填充颜色
        outline_color: RGB轮廓颜色（None表示无轮廓）
        outline_width: 轮廓宽度

    Returns:
        修改后的帧
    """
    import math

    draw = ImageDraw.Draw(frame)
    x, y = center

    # 计算星形点
    points = []
    for i in range(10):
        angle = (i * 36 - 90) * math.pi / 180  # 每个点36度，从顶部开始
        radius = size if i % 2 == 0 else size * 0.4  # 外点和内点交替
        px = x + radius * math.cos(angle)
        py = y + radius * math.sin(angle)
        points.append((px, py))

    # 绘制星星
    draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)

    return frame
