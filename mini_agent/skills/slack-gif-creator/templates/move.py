#!/usr/bin/env python3
"""
移动动画 - 让对象沿路径移动，支持多种运动类型。

提供灵活的运动基元，支持对象沿直线、弧线或自定义路径移动。
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.easing import calculate_arc_motion, interpolate
from core.frame_composer import create_blank_frame, draw_circle, draw_emoji_enhanced
from core.gif_builder import GIFBuilder


def create_move_animation(
    object_type: str = "emoji",
    object_data: dict | None = None,
    start_pos: tuple[int, int] = (50, 240),
    end_pos: tuple[int, int] = (430, 240),
    num_frames: int = 30,
    motion_type: str = "linear",  # 'linear'、'arc'、'bezier'、'circle'、'wave'
    easing: str = "ease_out",
    motion_params: dict | None = None,
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list:
    """
    创建对象沿路径移动的帧序列。

    参数:
        object_type: 'circle'、'emoji' 或 'custom'
        object_data: 对象数据
        start_pos: 起始 (x, y) 位置
        end_pos: 结束 (x, y) 位置
        num_frames: 帧数
        motion_type: 运动路径类型
        easing: 缓动函数名称
        motion_params: 运动附加参数（例如，{'arc_height': 100}）
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    frames = []

    # 默认对象数据
    if object_data is None:
        if object_type == "circle":
            object_data = {"radius": 30, "color": (100, 150, 255)}
        elif object_type == "emoji":
            object_data = {"emoji": "🚀", "size": 60}

    # 默认运动参数
    if motion_params is None:
        motion_params = {}

    for i in range(num_frames):
        frame = create_blank_frame(frame_width, frame_height, bg_color)

        t = i / (num_frames - 1) if num_frames > 1 else 0

        # 根据运动类型计算位置
        if motion_type == "linear":
            # 带缓动的直线运动
            x = interpolate(start_pos[0], end_pos[0], t, easing)
            y = interpolate(start_pos[1], end_pos[1], t, easing)

        elif motion_type == "arc":
            # 抛物线运动
            arc_height = motion_params.get("arc_height", 100)
            x, y = calculate_arc_motion(start_pos, end_pos, arc_height, t)

        elif motion_type == "circle":
            # 围绕中心点的圆周运动
            center = motion_params.get("center", (frame_width // 2, frame_height // 2))
            radius = motion_params.get("radius", 150)
            start_angle = motion_params.get("start_angle", 0)
            angle_range = motion_params.get("angle_range", 360)  # 完整圆周

            angle = start_angle + (angle_range * t)
            angle_rad = math.radians(angle)

            x = center[0] + radius * math.cos(angle_rad)
            y = center[1] + radius * math.sin(angle_rad)

        elif motion_type == "wave":
            # 沿直线移动并添加波动效果
            wave_amplitude = motion_params.get("wave_amplitude", 50)
            wave_frequency = motion_params.get("wave_frequency", 2)

            # 基础线性运动
            base_x = interpolate(start_pos[0], end_pos[0], t, easing)
            base_y = interpolate(start_pos[1], end_pos[1], t, easing)

            # 添加与运动方向垂直的波动偏移
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length > 0:
                # 垂直方向
                perp_x = -dy / length
                perp_y = dx / length

                # 波动偏移
                wave_offset = math.sin(t * wave_frequency * 2 * math.pi) * wave_amplitude

                x = base_x + perp_x * wave_offset
                y = base_y + perp_y * wave_offset
            else:
                x, y = base_x, base_y

        elif motion_type == "bezier":
            # 二次贝塞尔曲线
            control_point = motion_params.get(
                "control_point", ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2 - 100)
            )

            # 二次贝塞尔公式: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
            x = (1 - t) ** 2 * start_pos[0] + 2 * (1 - t) * t * control_point[0] + t**2 * end_pos[0]
            y = (1 - t) ** 2 * start_pos[1] + 2 * (1 - t) * t * control_point[1] + t**2 * end_pos[1]

        else:
            # 默认为线性运动
            x = interpolate(start_pos[0], end_pos[0], t, easing)
            y = interpolate(start_pos[1], end_pos[1], t, easing)

        # 在计算出的位置绘制对象
        x, y = int(x), int(y)

        if object_type == "circle":
            draw_circle(frame, center=(x, y), radius=object_data["radius"], fill_color=object_data["color"])
        elif object_type == "emoji":
            draw_emoji_enhanced(
                frame,
                emoji=object_data["emoji"],
                position=(x - object_data["size"] // 2, y - object_data["size"] // 2),
                size=object_data["size"],
                shadow=object_data.get("shadow", True),
            )

        frames.append(frame)

    return frames


def create_path_from_points(
    points: list[tuple[int, int]], num_frames: int = 60, easing: str = "ease_in_out"
) -> list[tuple[int, int]]:
    """
    创建穿过多个点的平滑路径。

    参数:
        points: (x, y) 航点列表
        num_frames: 总帧数
        easing: 点之间的缓动

    返回:
        每帧的 (x, y) 位置列表
    """
    if len(points) < 2:
        return points * num_frames

    path = []
    frames_per_segment = num_frames // (len(points) - 1)

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        # 最后一个片段获得剩余的帧
        segment_frames = num_frames - len(path) if i == len(points) - 2 else frames_per_segment

        for j in range(segment_frames):
            t = j / segment_frames if segment_frames > 0 else 0
            x = interpolate(start[0], end[0], t, easing)
            y = interpolate(start[1], end[1], t, easing)
            path.append((int(x), int(y)))

    return path


def apply_trail_effect(frames: list, trail_length: int = 5, fade_alpha: float = 0.3) -> list:
    """
    为移动对象添加运动拖尾效果。

    参数:
        frames: 包含移动对象的帧列表
        trail_length: 要混合的先前帧数
        fade_alpha: 拖尾帧的不透明度

    返回:
        带有拖尾效果的帧列表
    """
    import numpy as np
    from PIL import Image

    trailed_frames = []

    for i, frame in enumerate(frames):
        # 从当前帧开始
        result = frame.copy()

        # 混合先前的帧
        for j in range(1, min(trail_length + 1, i + 1)):
            prev_frame = frames[i - j]

            # 计算淡出
            alpha = fade_alpha**j

            # 混合
            result_array = np.array(result, dtype=np.float32)
            prev_array = np.array(prev_frame, dtype=np.float32)

            blended = result_array * (1 - alpha) + prev_array * alpha
            result = Image.fromarray(blended.astype(np.uint8))

        trailed_frames.append(result)

    return trailed_frames


# 示例用法
if __name__ == "__main__":
    print("正在创建移动示例...")

    # 示例 1: 线性移动
    builder = GIFBuilder(width=480, height=480, fps=20)
    frames = create_move_animation(
        object_type="emoji",
        object_data={"emoji": "🚀", "size": 60},
        start_pos=(50, 240),
        end_pos=(430, 240),
        num_frames=30,
        motion_type="linear",
        easing="ease_out",
    )
    builder.add_frames(frames)
    builder.save("move_linear.gif", num_colors=128)

    # 示例 2: 弧线移动
    builder.clear()
    frames = create_move_animation(
        object_type="emoji",
        object_data={"emoji": "⚽", "size": 60},
        start_pos=(50, 350),
        end_pos=(430, 350),
        num_frames=30,
        motion_type="arc",
        motion_params={"arc_height": 150},
        easing="linear",
    )
    builder.add_frames(frames)
    builder.save("move_arc.gif", num_colors=128)

    # 示例 3: 圆周移动
    builder.clear()
    frames = create_move_animation(
        object_type="emoji",
        object_data={"emoji": "🌍", "size": 50},
        start_pos=(0, 0),  # 圆周运动时忽略
        end_pos=(0, 0),  # 圆周运动时忽略
        num_frames=40,
        motion_type="circle",
        motion_params={"center": (240, 240), "radius": 120, "start_angle": 0, "angle_range": 360},
        easing="linear",
    )
    builder.add_frames(frames)
    builder.save("move_circle.gif", num_colors=128)

    print("已创建移动示例!")
