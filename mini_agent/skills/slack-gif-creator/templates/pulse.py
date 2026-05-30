#!/usr/bin/env python3
"""
脉冲动画 - 有节奏地缩放对象以增强效果。

创建脉冲、心跳和跳动效果。
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.easing import interpolate
from core.frame_composer import create_blank_frame, draw_circle, draw_emoji_enhanced
from core.gif_builder import GIFBuilder
from PIL import Image


def create_pulse_animation(
    object_type: str = "emoji",
    object_data: dict | None = None,
    num_frames: int = 30,
    pulse_type: str = "smooth",  # 'smooth', 'heartbeat', 'throb', 'pop' - 平滑、心跳、跳动、弹出
    scale_range: tuple[float, float] = (0.8, 1.2),
    pulses: float = 2.0,
    center_pos: tuple[int, int] = (240, 240),
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    创建脉冲/缩放动画。

    Args:
        object_type: 'emoji', 'circle', 'text'
        object_data: Object configuration
        num_frames: Number of frames
        pulse_type: Type of pulsing motion
        scale_range: (min_scale, max_scale) tuple
        pulses: Number of pulses in animation
        center_pos: Center position
        frame_width: Frame width
        frame_height: Frame height
        bg_color: Background color

    Returns:
        List of frames
    """
    frames = []

    # 默认对象数据
    if object_data is None:
        if object_type == "emoji":
            object_data = {"emoji": "❤️", "size": 100}
        elif object_type == "circle":
            object_data = {"radius": 50, "color": (255, 100, 100)}

    min_scale, max_scale = scale_range

    for i in range(num_frames):
        frame = create_blank_frame(frame_width, frame_height, bg_color)
        t = i / (num_frames - 1) if num_frames > 1 else 0

        # 根据脉冲类型计算缩放比例
        if pulse_type == "smooth":
            # 简单的正弦脉冲
            scale = min_scale + (max_scale - min_scale) * (0.5 + 0.5 * math.sin(t * pulses * 2 * math.pi - math.pi / 2))

        elif pulse_type == "heartbeat":
            # 像心跳一样的双泵
            phase = (t * pulses) % 1.0
            if phase < 0.15:
                # 第一泵
                scale = interpolate(min_scale, max_scale, phase / 0.15, "ease_out")
            elif phase < 0.25:
                # 第一次释放
                scale = interpolate(max_scale, min_scale, (phase - 0.15) / 0.10, "ease_in")
            elif phase < 0.35:
                # 第二泵（较小）
                scale = interpolate(min_scale, (min_scale + max_scale) / 2, (phase - 0.25) / 0.10, "ease_out")
            elif phase < 0.45:
                # 第二次释放
                scale = interpolate((min_scale + max_scale) / 2, min_scale, (phase - 0.35) / 0.10, "ease_in")
            else:
                # 静止期
                scale = min_scale

        elif pulse_type == "throb":
            # 快速脉冲快速返回
            phase = (t * pulses) % 1.0
            if phase < 0.2:
                scale = interpolate(min_scale, max_scale, phase / 0.2, "ease_out")
            else:
                scale = interpolate(max_scale, min_scale, (phase - 0.2) / 0.8, "ease_in")

        elif pulse_type == "pop":
            # 弹出并返回，带有过冲
            phase = (t * pulses) % 1.0
            if phase < 0.3:
                # 带过冲的弹出
                scale = interpolate(min_scale, max_scale * 1.1, phase / 0.3, "elastic_out")
            else:
                # 稳定回缩
                scale = interpolate(max_scale * 1.1, min_scale, (phase - 0.3) / 0.7, "ease_out")

        else:
            scale = min_scale + (max_scale - min_scale) * (0.5 + 0.5 * math.sin(t * pulses * 2 * math.pi))

        # 以计算的缩放比例绘制对象
        if object_type == "emoji":
            base_size = object_data["size"]
            current_size = int(base_size * scale)
            draw_emoji_enhanced(
                frame,
                emoji=object_data["emoji"],
                position=(center_pos[0] - current_size // 2, center_pos[1] - current_size // 2),
                size=current_size,
                shadow=object_data.get("shadow", True),
            )

        elif object_type == "circle":
            base_radius = object_data["radius"]
            current_radius = int(base_radius * scale)
            draw_circle(frame, center=center_pos, radius=current_radius, fill_color=object_data["color"])

        elif object_type == "text":
            from core.typography import draw_text_with_outline

            base_size = object_data.get("font_size", 50)
            current_size = int(base_size * scale)
            draw_text_with_outline(
                frame,
                text=object_data.get("text", "PULSE"),
                position=center_pos,
                font_size=current_size,
                text_color=object_data.get("text_color", (255, 100, 100)),
                outline_color=object_data.get("outline_color", (0, 0, 0)),
                outline_width=3,
                centered=True,
            )

        frames.append(frame)

    return frames


def create_attention_pulse(
    emoji: str = "⚠️", num_frames: int = 20, frame_size: int = 128, bg_color: tuple[int, int, int] = (255, 255, 255)
) -> list[Image.Image]:
    """
    创建吸引注意力的脉冲动画（适合表情包GIF）。

    参数:
        emoji: 要脉冲的表情
        num_frames: 帧数
        frame_size: 帧尺寸（正方形）
        bg_color: 背景颜色

    返回:
        针对表情包尺寸优化的帧列表
    """
    return create_pulse_animation(
        object_type="emoji",
        object_data={"emoji": emoji, "size": 80, "shadow": False},
        num_frames=num_frames,
        pulse_type="throb",
        scale_range=(0.85, 1.15),
        pulses=2,
        center_pos=(frame_size // 2, frame_size // 2),
        frame_width=frame_size,
        frame_height=frame_size,
        bg_color=bg_color,
    )


def create_breathing_animation(
    object_type: str = "emoji",
    object_data: dict | None = None,
    num_frames: int = 60,
    breaths: float = 2.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (240, 248, 255),
) -> list[Image.Image]:
    """
    创建缓慢、平静的呼吸动画（吸气和呼气）。

    参数:
        object_type: 对象类型
        object_data: 对象配置
        num_frames: 帧数
        breaths: 呼吸周期数
        scale_range: 最小/最大缩放
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    if object_data is None:
        object_data = {"emoji": "😌", "size": 100}

    return create_pulse_animation(
        object_type=object_type,
        object_data=object_data,
        num_frames=num_frames,
        pulse_type="smooth",
        scale_range=scale_range,
        pulses=breaths,
        center_pos=(frame_width // 2, frame_height // 2),
        frame_width=frame_width,
        frame_height=frame_height,
        bg_color=bg_color,
    )


# 示例用法
if __name__ == "__main__":
    print("Creating pulse animations...")

    builder = GIFBuilder(width=480, height=480, fps=20)

    # 示例 1: 平滑脉冲
    frames = create_pulse_animation(
        object_type="emoji",
        object_data={"emoji": "❤️", "size": 100},
        num_frames=40,
        pulse_type="smooth",
        scale_range=(0.8, 1.2),
        pulses=2,
    )
    builder.add_frames(frames)
    builder.save("pulse_smooth.gif", num_colors=128)

    # 示例 2: 心跳
    builder.clear()
    frames = create_pulse_animation(
        object_type="emoji",
        object_data={"emoji": "💓", "size": 100},
        num_frames=60,
        pulse_type="heartbeat",
        scale_range=(0.85, 1.2),
        pulses=3,
    )
    builder.add_frames(frames)
    builder.save("pulse_heartbeat.gif", num_colors=128)

    # 示例 3: 注意脉冲（表情包尺寸）
    builder = GIFBuilder(width=128, height=128, fps=15)
    frames = create_attention_pulse(emoji="⚠️", num_frames=20)
    builder.add_frames(frames)
    builder.save("pulse_attention.gif", num_colors=48, optimize_for_emoji=True)

    print("Created pulse animations!")
