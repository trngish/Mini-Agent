#!/usr/bin/env python3
"""
缩放动画 - 戏剧性地缩放对象以增强效果。

创建放大、缩小和戏剧性缩放效果。
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.easing import interpolate
from core.frame_composer import create_blank_frame, draw_emoji_enhanced
from core.gif_builder import GIFBuilder
from PIL import Image, ImageFilter


def create_zoom_animation(
    object_type: str = "emoji",
    object_data: dict | None = None,
    num_frames: int = 30,
    zoom_type: str = "in",  # 'in', 'out', 'in_out', 'punch' - 放大、缩小、放大缩小、猛击
    scale_range: tuple[float, float] = (0.1, 2.0),
    easing: str = "ease_out",
    add_motion_blur: bool = False,
    center_pos: tuple[int, int] = (240, 240),
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    创建缩放动画。

    参数:
        object_type: 'emoji', 'text', 'image'
        object_data: 对象配置
        num_frames: 帧数
        zoom_type: 缩放效果类型
        scale_range: (起始缩放, 结束缩放) 元组
        easing: 缓动函数
        add_motion_blur: 添加模糊以增强速度效果
        center_pos: 中心位置
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    frames = []

    # 默认对象数据
    if object_data is None and object_type == "emoji":
        object_data = {"emoji": "🔍", "size": 100}

    base_size = object_data.get("size", 100) if object_type == "emoji" else object_data.get("font_size", 60)
    start_scale, end_scale = scale_range

    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0

        # 根据缩放类型计算缩放比例
        if zoom_type == "in":
            scale = interpolate(start_scale, end_scale, t, easing)
        elif zoom_type == "out":
            scale = interpolate(end_scale, start_scale, t, easing)
        elif zoom_type == "in_out":
            if t < 0.5:
                scale = interpolate(start_scale, end_scale, t * 2, easing)
            else:
                scale = interpolate(end_scale, start_scale, (t - 0.5) * 2, easing)
        elif zoom_type == "punch":
            # 快速放大带过冲然后稳定
            if t < 0.3:
                scale = interpolate(start_scale, end_scale * 1.2, t / 0.3, "ease_out")
            else:
                scale = interpolate(end_scale * 1.2, end_scale, (t - 0.3) / 0.7, "elastic_out")
        else:
            scale = interpolate(start_scale, end_scale, t, easing)

        # Create frame
        frame = create_blank_frame(frame_width, frame_height, bg_color)

        if object_type == "emoji":
            current_size = int(base_size * scale)

            # 限制大小在合理范围内
            current_size = max(12, min(current_size, frame_width * 2))

            # 在透明背景上创建表情包
            canvas_size = max(frame_width, frame_height, current_size) * 2
            emoji_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

            draw_emoji_enhanced(
                emoji_canvas,
                emoji=object_data["emoji"],
                position=(canvas_size // 2 - current_size // 2, canvas_size // 2 - current_size // 2),
                size=current_size,
                shadow=False,
            )

            # 快速缩放时的可选运动模糊
            if add_motion_blur and abs(scale - 1.0) > 0.5:
                blur_amount = min(5, int(abs(scale - 1.0) * 3))
                emoji_canvas = emoji_canvas.filter(ImageFilter.GaussianBlur(blur_amount))

            # 裁剪到以中心为准的帧尺寸
            left = (canvas_size - frame_width) // 2
            top = (canvas_size - frame_height) // 2
            emoji_cropped = emoji_canvas.crop((left, top, left + frame_width, top + frame_height))

            # 合成
            frame_rgba = frame.convert("RGBA")
            frame = Image.alpha_composite(frame_rgba, emoji_cropped)
            frame = frame.convert("RGB")

        elif object_type == "text":
            from core.typography import draw_text_with_outline

            current_size = int(base_size * scale)
            current_size = max(10, min(current_size, 500))

            # 为大文本创建超大画布
            canvas_size = max(frame_width, frame_height, current_size * 10)
            text_canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)

            draw_text_with_outline(
                text_canvas,
                text=object_data.get("text", "ZOOM"),
                position=(canvas_size // 2, canvas_size // 2),
                font_size=current_size,
                text_color=object_data.get("text_color", (0, 0, 0)),
                outline_color=object_data.get("outline_color", (255, 255, 255)),
                outline_width=max(2, int(current_size * 0.05)),
                centered=True,
            )

            # 裁剪到帧
            left = (canvas_size - frame_width) // 2
            top = (canvas_size - frame_height) // 2
            frame = text_canvas.crop((left, top, left + frame_width, top + frame_height))

        frames.append(frame)

    return frames


def create_explosion_zoom(
    emoji: str = "💥",
    num_frames: int = 20,
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    创建戏剧性的爆炸缩放效果。

    参数:
        emoji: 要爆炸的表情
        num_frames: 帧数
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    frames = []

    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0

        # 指数缩放
        scale = 0.1 * math.exp(t * 5)

        # 添加旋转以增强戏剧效果
        angle = t * 360 * 2

        frame = create_blank_frame(frame_width, frame_height, bg_color)

        current_size = int(100 * scale)
        current_size = max(12, min(current_size, frame_width * 3))

        # 创建表情包
        canvas_size = max(frame_width, frame_height, current_size) * 2
        emoji_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

        draw_emoji_enhanced(
            emoji_canvas,
            emoji=emoji,
            position=(canvas_size // 2 - current_size // 2, canvas_size // 2 - current_size // 2),
            size=current_size,
            shadow=False,
        )

        # 旋转
        emoji_canvas = emoji_canvas.rotate(angle, center=(canvas_size // 2, canvas_size // 2), resample=Image.BICUBIC)

        # 为后面的帧添加运动模糊
        if t > 0.5:
            blur_amount = int((t - 0.5) * 10)
            emoji_canvas = emoji_canvas.filter(ImageFilter.GaussianBlur(blur_amount))

        # 裁剪并合成
        left = (canvas_size - frame_width) // 2
        top = (canvas_size - frame_height) // 2
        emoji_cropped = emoji_canvas.crop((left, top, left + frame_width, top + frame_height))

        frame_rgba = frame.convert("RGBA")
        frame = Image.alpha_composite(frame_rgba, emoji_cropped)
        frame = frame.convert("RGB")

        frames.append(frame)

    return frames


def create_mind_blown_zoom(
    emoji: str = "🤯",
    num_frames: int = 30,
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    创建"大脑爆炸"戏剧性缩放带摇晃效果。

    参数:
        emoji: 要使用的表情
        num_frames: 帧数
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    frames = []

    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0

        # 先放大然后摇晃
        if t < 0.5:
            scale = interpolate(0.3, 1.2, t * 2, "ease_out")
            shake_x = 0
            shake_y = 0
        else:
            scale = 1.2
            # 摇晃增强
            shake_intensity = (t - 0.5) * 40
            shake_x = int(math.sin(t * 50) * shake_intensity)
            shake_y = int(math.cos(t * 45) * shake_intensity)

        frame = create_blank_frame(frame_width, frame_height, bg_color)

        current_size = int(100 * scale)
        center_x = frame_width // 2 + shake_x
        center_y = frame_height // 2 + shake_y

        emoji_canvas = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
        draw_emoji_enhanced(
            emoji_canvas,
            emoji=emoji,
            position=(center_x - current_size // 2, center_y - current_size // 2),
            size=current_size,
            shadow=False,
        )

        frame_rgba = frame.convert("RGBA")
        frame = Image.alpha_composite(frame_rgba, emoji_canvas)
        frame = frame.convert("RGB")

        frames.append(frame)

    return frames


# 示例用法
if __name__ == "__main__":
    print("Creating zoom animations...")

    builder = GIFBuilder(width=480, height=480, fps=20)

    # 示例 1: 放大
    frames = create_zoom_animation(
        object_type="emoji",
        object_data={"emoji": "🔍", "size": 100},
        num_frames=30,
        zoom_type="in",
        scale_range=(0.1, 1.5),
        easing="ease_out",
    )
    builder.add_frames(frames)
    builder.save("zoom_in.gif", num_colors=128)

    # 示例 2: 爆炸缩放
    builder.clear()
    frames = create_explosion_zoom(emoji="💥", num_frames=20)
    builder.add_frames(frames)
    builder.save("zoom_explosion.gif", num_colors=128)

    # 示例 3: 大脑爆炸
    builder.clear()
    frames = create_mind_blown_zoom(emoji="🤯", num_frames=30)
    builder.add_frames(frames)
    builder.save("zoom_mind_blown.gif", num_colors=128)

    print("Created zoom animations!")
