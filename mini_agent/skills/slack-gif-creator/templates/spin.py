#!/usr/bin/env python3
"""
旋转动画 - 持续或变化地旋转对象。

创建旋转、摆动效果。
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.easing import interpolate
from core.frame_composer import create_blank_frame, draw_emoji_enhanced
from core.gif_builder import GIFBuilder
from PIL import Image


def create_spin_animation(
    object_type: str = "emoji",
    object_data: dict | None = None,
    num_frames: int = 30,
    rotation_type: str = "clockwise",  # 'clockwise'、'counterclockwise'、'wobble'、'pendulum'
    full_rotations: float = 1.0,
    easing: str = "linear",
    center_pos: tuple[int, int] = (240, 240),
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    创建旋转动画。

    参数:
        object_type: 'emoji'、'image'、'text'
        object_data: 对象配置
        num_frames: 帧数
        rotation_type: 旋转类型
        full_rotations: 完整 360° 旋转的次数
        easing: 旋转速度的缓动函数
        center_pos: 旋转中心位置
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    frames = []

    # 默认对象数据
    if object_data is None and object_type == "emoji":
        object_data = {"emoji": "🔄", "size": 100}

    for i in range(num_frames):
        frame = create_blank_frame(frame_width, frame_height, bg_color)
        t = i / (num_frames - 1) if num_frames > 1 else 0

        # 计算旋转角度
        if rotation_type == "clockwise":
            angle = interpolate(0, 360 * full_rotations, t, easing)
        elif rotation_type == "counterclockwise":
            angle = interpolate(0, -360 * full_rotations, t, easing)
        elif rotation_type == "wobble":
            # 来回旋转
            angle = math.sin(t * full_rotations * 2 * math.pi) * 45
        elif rotation_type == "pendulum":
            # 平滑钟摆运动
            angle = math.sin(t * full_rotations * 2 * math.pi) * 90
        else:
            angle = interpolate(0, 360 * full_rotations, t, easing)

        # 在透明背景上创建对象以便旋转
        if object_type == "emoji":
            # 对于 emoji，需要创建更大的画布以避免旋转时被裁剪
            emoji_size = object_data["size"]
            canvas_size = int(emoji_size * 1.5)
            emoji_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

            # 在画布中心绘制 emoji
            from core.frame_composer import draw_emoji_enhanced

            draw_emoji_enhanced(
                emoji_canvas,
                emoji=object_data["emoji"],
                position=(canvas_size // 2 - emoji_size // 2, canvas_size // 2 - emoji_size // 2),
                size=emoji_size,
                shadow=False,
            )

            # 旋转画布
            rotated = emoji_canvas.rotate(angle, resample=Image.BICUBIC, expand=False)

            # 粘贴到帧上
            paste_x = center_pos[0] - canvas_size // 2
            paste_y = center_pos[1] - canvas_size // 2
            frame.paste(rotated, (paste_x, paste_y), rotated)

        elif object_type == "text":
            from core.typography import draw_text_with_outline

            # 类似方法 - 创建画布、绘制文字、旋转
            text = object_data.get("text", "SPIN!")
            font_size = object_data.get("font_size", 50)

            canvas_size = max(frame_width, frame_height)
            text_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

            # 绘制文字
            text_canvas_rgb = text_canvas.convert("RGB")
            text_canvas_rgb.paste(bg_color, (0, 0, canvas_size, canvas_size))
            draw_text_with_outline(
                text_canvas_rgb,
                text,
                position=(canvas_size // 2, canvas_size // 2),
                font_size=font_size,
                text_color=object_data.get("text_color", (0, 0, 0)),
                outline_color=object_data.get("outline_color", (255, 255, 255)),
                outline_width=3,
                centered=True,
            )

            # 转换回 RGBA 以便旋转
            text_canvas = text_canvas_rgb.convert("RGBA")

            # 使背景透明
            data = text_canvas.getdata()
            new_data = []
            for item in data:
                if item[:3] == bg_color:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            text_canvas.putdata(new_data)

            # 旋转
            rotated = text_canvas.rotate(angle, resample=Image.BICUBIC, expand=False)

            # 合成到帧上
            frame_rgba = frame.convert("RGBA")
            frame_rgba = Image.alpha_composite(frame_rgba, rotated)
            frame = frame_rgba.convert("RGB")

        frames.append(frame)

    return frames


def create_loading_spinner(
    num_frames: int = 20,
    spinner_type: str = "dots",  # 'dots'、'arc'、'emoji'
    size: int = 100,
    color: tuple[int, int, int] = (100, 150, 255),
    frame_width: int = 128,
    frame_height: int = 128,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    创建加载动画。

    参数:
        num_frames: 帧数
        spinner_type: 加载器类型
        size: 加载器大小
        color: 加载器颜色
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    from PIL import ImageDraw

    frames = []
    center = (frame_width // 2, frame_height // 2)

    for i in range(num_frames):
        frame = create_blank_frame(frame_width, frame_height, bg_color)
        draw = ImageDraw.Draw(frame)

        angle_offset = (i / num_frames) * 360

        if spinner_type == "dots":
            # 圆形点
            num_dots = 8
            for j in range(num_dots):
                angle = (j / num_dots * 360 + angle_offset) * math.pi / 180
                x = center[0] + size * 0.4 * math.cos(angle)
                y = center[1] + size * 0.4 * math.sin(angle)

                # 根据位置淡出
                alpha = 1.0 - (j / num_dots)
                dot_color = tuple(int(c * alpha) for c in color)
                dot_radius = int(size * 0.1)

                draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=dot_color)

        elif spinner_type == "arc":
            # 旋转弧线
            start_angle = angle_offset
            end_angle = angle_offset + 270
            arc_width = int(size * 0.15)

            bbox = [center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2]
            draw.arc(bbox, start_angle, end_angle, fill=color, width=arc_width)

        elif spinner_type == "emoji":
            # 旋转 emoji 加载器
            angle = angle_offset
            emoji_canvas = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
            draw_emoji_enhanced(
                emoji_canvas,
                emoji="⏳",
                position=(center[0] - size // 2, center[1] - size // 2),
                size=size,
                shadow=False,
            )
            rotated = emoji_canvas.rotate(angle, center=center, resample=Image.BICUBIC)
            frame.paste(rotated, (0, 0), rotated)

        frames.append(frame)

    return frames


# 示例用法
if __name__ == "__main__":
    print("正在创建旋转动画...")

    builder = GIFBuilder(width=480, height=480, fps=20)

    # 示例 1: 顺时针旋转
    frames = create_spin_animation(
        object_type="emoji",
        object_data={"emoji": "🔄", "size": 100},
        num_frames=30,
        rotation_type="clockwise",
        full_rotations=2,
    )
    builder.add_frames(frames)
    builder.save("spin_clockwise.gif", num_colors=128)

    # 示例 2: 摆动效果
    builder.clear()
    frames = create_spin_animation(
        object_type="emoji",
        object_data={"emoji": "🎯", "size": 100},
        num_frames=30,
        rotation_type="wobble",
        full_rotations=3,
    )
    builder.add_frames(frames)
    builder.save("spin_wobble.gif", num_colors=128)

    # 示例 3: 加载动画
    builder = GIFBuilder(width=128, height=128, fps=15)
    frames = create_loading_spinner(num_frames=20, spinner_type="dots")
    builder.add_frames(frames)
    builder.save("loading_spinner.gif", num_colors=64, optimize_for_emoji=True)

    print("已创建旋转动画!")
