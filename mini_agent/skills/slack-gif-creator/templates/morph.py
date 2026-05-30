#!/usr/bin/env python3
"""
Morph Animation - Transform between different emojis or shapes.

Creates smooth transitions and transformations.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.easing import interpolate
from core.frame_composer import create_blank_frame, draw_circle, draw_emoji_enhanced
from core.gif_builder import GIFBuilder
from PIL import Image


def create_morph_animation(
    object1_data: dict,
    object2_data: dict,
    num_frames: int = 30,
    morph_type: str = "crossfade",  # 'crossfade': 交叉淡入淡出, 'scale': 缩放过渡, 'spin_morph': 旋转变形
    easing: str = "ease_in_out",
    object_type: str = "emoji",
    center_pos: tuple[int, int] = (240, 240),
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    创建两个物体之间的变形动画。

    参数:
        object1_data: 第一个物体配置
        object2_data: 第二个物体配置
        num_frames: 帧数
        morph_type: 变形效果类型
        easing: 缓动函数
        object_type: 物体类型
        center_pos: 中心位置
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    frames = []

    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        frame = create_blank_frame(frame_width, frame_height, bg_color)

        if morph_type == "crossfade":
            # 两个物体之间的简单交叉淡入淡出
            opacity1 = interpolate(1, 0, t, easing)
            opacity2 = interpolate(0, 1, t, easing)

            if object_type == "emoji":
                # 创建第一个 emoji
                emoji1_canvas = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
                size1 = object1_data["size"]
                draw_emoji_enhanced(
                    emoji1_canvas,
                    emoji=object1_data["emoji"],
                    position=(center_pos[0] - size1 // 2, center_pos[1] - size1 // 2),
                    size=size1,
                    shadow=False,
                )

                # 应用不透明度
                from templates.fade import apply_opacity

                emoji1_canvas = apply_opacity(emoji1_canvas, opacity1)

                # 创建第二个 emoji
                emoji2_canvas = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
                size2 = object2_data["size"]
                draw_emoji_enhanced(
                    emoji2_canvas,
                    emoji=object2_data["emoji"],
                    position=(center_pos[0] - size2 // 2, center_pos[1] - size2 // 2),
                    size=size2,
                    shadow=False,
                )

                emoji2_canvas = apply_opacity(emoji2_canvas, opacity2)

                # 合成两者
                frame_rgba = frame.convert("RGBA")
                frame_rgba = Image.alpha_composite(frame_rgba, emoji1_canvas)
                frame_rgba = Image.alpha_composite(frame_rgba, emoji2_canvas)
                frame = frame_rgba.convert("RGB")

            elif object_type == "circle":
                # 在两个圆之间变形
                radius1 = object1_data["radius"]
                radius2 = object2_data["radius"]
                color1 = object1_data["color"]
                color2 = object2_data["color"]

                # 插值属性
                current_radius = int(interpolate(radius1, radius2, t, easing))
                current_color = tuple(int(interpolate(color1[i], color2[i], t, easing)) for i in range(3))

                draw_circle(frame, center_pos, current_radius, fill_color=current_color)

        elif morph_type == "scale":
            # 第一个物体缩小，第二个物体放大
            if object_type == "emoji":
                scale1 = interpolate(1.0, 0.0, t, easing)
                scale2 = interpolate(0.0, 1.0, t, easing)

                # 绘制第一个 emoji（正在缩小）
                if scale1 > 0.05:
                    size1 = int(object1_data["size"] * scale1)
                    size1 = max(12, size1)
                    emoji1_canvas = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
                    draw_emoji_enhanced(
                        emoji1_canvas,
                        emoji=object1_data["emoji"],
                        position=(center_pos[0] - size1 // 2, center_pos[1] - size1 // 2),
                        size=size1,
                        shadow=False,
                    )

                    frame_rgba = frame.convert("RGBA")
                    frame = Image.alpha_composite(frame_rgba, emoji1_canvas)
                    frame = frame.convert("RGB")

                # 绘制第二个 emoji（正在放大）
                if scale2 > 0.05:
                    size2 = int(object2_data["size"] * scale2)
                    size2 = max(12, size2)
                    emoji2_canvas = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
                    draw_emoji_enhanced(
                        emoji2_canvas,
                        emoji=object2_data["emoji"],
                        position=(center_pos[0] - size2 // 2, center_pos[1] - size2 // 2),
                        size=size2,
                        shadow=False,
                    )

                    frame_rgba = frame.convert("RGBA")
                    frame = Image.alpha_composite(frame_rgba, emoji2_canvas)
                    frame = frame.convert("RGB")

        elif morph_type == "spin_morph":
            # 旋转同时变形（类似翻转）
            import math

            # 计算旋转角度（0 到 180 度）
            angle = interpolate(0, 180, t, easing)
            scale_factor = abs(math.cos(math.radians(angle)))

            # 判断显示哪个物体
            current_object = object1_data if angle < 90 else object2_data

            # 侧面朝前时跳过
            if scale_factor < 0.05:
                frames.append(frame)
                continue

            if object_type == "emoji":
                size = current_object["size"]
                canvas_size = size * 2
                emoji_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

                draw_emoji_enhanced(
                    emoji_canvas,
                    emoji=current_object["emoji"],
                    position=(canvas_size // 2 - size // 2, canvas_size // 2 - size // 2),
                    size=size,
                    shadow=False,
                )

                # 为旋转效果水平缩放
                new_width = max(1, int(canvas_size * scale_factor))
                emoji_scaled = emoji_canvas.resize((new_width, canvas_size), Image.LANCZOS)

                paste_x = center_pos[0] - new_width // 2
                paste_y = center_pos[1] - canvas_size // 2

                frame_rgba = frame.convert("RGBA")
                frame_rgba.paste(emoji_scaled, (paste_x, paste_y), emoji_scaled)
                frame = frame_rgba.convert("RGB")

        frames.append(frame)

    return frames


def create_reaction_morph(
    emoji_start: str, emoji_end: str, num_frames: int = 20, frame_size: int = 128
) -> list[Image.Image]:
    """
    创建快速 emoji 反应变形（用于 emoji GIF）。

    参数:
        emoji_start: 起始 emoji
        emoji_end: 结束 emoji
        num_frames: 帧数
        frame_size: 帧大小（正方形）

    返回:
        帧列表
    """
    return create_morph_animation(
        object1_data={"emoji": emoji_start, "size": 80},
        object2_data={"emoji": emoji_end, "size": 80},
        num_frames=num_frames,
        morph_type="crossfade",
        easing="ease_in_out",
        object_type="emoji",
        center_pos=(frame_size // 2, frame_size // 2),
        frame_width=frame_size,
        frame_height=frame_size,
        bg_color=(255, 255, 255),
    )


def create_shape_morph(
    shapes: list[dict],
    num_frames: int = 60,
    frames_per_shape: int = 20,
    frame_width: int = 480,
    frame_height: int = 480,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Image.Image]:
    """
    在一系列形状之间变形。

    参数:
        shapes: 形状字典列表，包含 'radius' 和 'color'
        num_frames: 总帧数
        frames_per_shape: 每个变形的帧数
        frame_width: 帧宽度
        frame_height: 帧高度
        bg_color: 背景颜色

    返回:
        帧列表
    """
    frames = []
    center = (frame_width // 2, frame_height // 2)

    for i in range(num_frames):
        # 确定我们正在变形的形状
        cycle_progress = (i % (frames_per_shape * len(shapes))) / frames_per_shape
        shape_idx = int(cycle_progress) % len(shapes)
        next_shape_idx = (shape_idx + 1) % len(shapes)

        # 这两个形状之间的进度
        t = cycle_progress - shape_idx

        shape1 = shapes[shape_idx]
        shape2 = shapes[next_shape_idx]

        # 插值属性
        radius = int(interpolate(shape1["radius"], shape2["radius"], t, "ease_in_out"))
        color = tuple(int(interpolate(shape1["color"][j], shape2["color"][j], t, "ease_in_out")) for j in range(3))

        # 绘制帧
        frame = create_blank_frame(frame_width, frame_height, bg_color)
        draw_circle(frame, center, radius, fill_color=color)

        frames.append(frame)

    return frames


# 示例用法
if __name__ == "__main__":
    print("正在创建变形动画...")

    builder = GIFBuilder(width=480, height=480, fps=20)

    # 示例 1: 交叉淡入淡出变形
    frames = create_morph_animation(
        object1_data={"emoji": "😊", "size": 100},
        object2_data={"emoji": "😂", "size": 100},
        num_frames=30,
        morph_type="crossfade",
        object_type="emoji",
    )
    builder.add_frames(frames)
    builder.save("morph_crossfade.gif", num_colors=128)

    # 示例 2: 缩放变形
    builder.clear()
    frames = create_morph_animation(
        object1_data={"emoji": "🌙", "size": 100},
        object2_data={"emoji": "☀️", "size": 100},
        num_frames=40,
        morph_type="scale",
        object_type="emoji",
    )
    builder.add_frames(frames)
    builder.save("morph_scale.gif", num_colors=128)

    # 示例 3: 形状变形循环
    builder.clear()
    from core.color_palettes import get_palette

    palette = get_palette("vibrant")

    shapes = [
        {"radius": 60, "color": palette["primary"]},
        {"radius": 80, "color": palette["secondary"]},
        {"radius": 50, "color": palette["accent"]},
        {"radius": 70, "color": palette["success"]},
    ]
    frames = create_shape_morph(shapes, num_frames=80, frames_per_shape=20)
    builder.add_frames(frames)
    builder.save("morph_shapes.gif", num_colors=64)

    print("变形动画创建完成!")
