#!/usr/bin/env python3
"""
Kaleidoscope Effect - Create mirror/rotation effects.

Apply kaleidoscope effects to frames or objects for psychedelic visuals.
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def apply_kaleidoscope(frame: Image.Image, segments: int = 8, center: tuple[int, int] | None = None) -> Image.Image:
    """
    通过镜像/旋转帧区域来应用万花筒效果。

    参数:
        frame: 输入帧
        segments: 镜像段数（4、6、8、12 效果较好）
        center: 效果中心点（None 表示帧中心）

    返回:
        应用了万花筒效果的帧
    """
    width, height = frame.size

    if center is None:
        center = (width // 2, height // 2)

    # 创建输出帧
    Image.new("RGB", (width, height))

    # 计算每段的 angle
    angle_per_segment = 360 / segments

    # 为简单起见，我们创建一个径向镜像效果
    # 完整的实现会正确地旋转和镜像
    # 这是创建一个有趣图案的简化版本

    # 转换为 numpy 数组以便操作
    frame_array = np.array(frame)
    output_array = np.zeros_like(frame_array)

    center_x, center_y = center

    # 创建楔形 mask 并镜像
    for y in range(height):
        for x in range(width):
            # 计算相对于中心的角度
            dx = x - center_x
            dy = y - center_y

            angle = (math.degrees(math.atan2(dy, dx)) + 180) % 360
            distance = math.sqrt(dx * dx + dy * dy)

            # 这个像素属于哪一段？
            segment = int(angle / angle_per_segment)

            # 镜像段内的角度
            segment_angle = angle % angle_per_segment
            if segment % 2 == 1:  # 每隔一段镜像一次
                segment_angle = angle_per_segment - segment_angle

            # 计算源位置
            source_angle = segment_angle + (segment // 2) * angle_per_segment * 2
            source_angle_rad = math.radians(source_angle - 180)

            source_x = int(center_x + distance * math.cos(source_angle_rad))
            source_y = int(center_y + distance * math.sin(source_angle_rad))

            # 边界检查
            if 0 <= source_x < width and 0 <= source_y < height:
                output_array[y, x] = frame_array[source_y, source_x]
            else:
                output_array[y, x] = frame_array[y, x]

    return Image.fromarray(output_array)


def apply_simple_mirror(frame: Image.Image, mode: str = "quad") -> Image.Image:
    """
    应用简单镜像效果（比完整万花筒快）。

    参数:
        frame: 输入帧
        mode: 模式，可选 'horizontal'、'vertical'、'quad'（4向）、'radial'

    返回:
        镜像后的帧
    """
    width, height = frame.size
    center_x, center_y = width // 2, height // 2

    if mode == "horizontal":
        # 将左半部分镜像到右边
        left_half = frame.crop((0, 0, center_x, height))
        left_flipped = ImageOps.mirror(left_half)
        result = frame.copy()
        result.paste(left_flipped, (center_x, 0))
        return result

    elif mode == "vertical":
        # 将上半部分镜像到底部
        top_half = frame.crop((0, 0, width, center_y))
        top_flipped = ImageOps.flip(top_half)
        result = frame.copy()
        result.paste(top_flipped, (0, center_y))
        return result

    elif mode == "quad":
        # 4向镜像（左上角镜像到所有象限）
        quad = frame.crop((0, 0, center_x, center_y))

        result = Image.new("RGB", (width, height))

        # 左上角（原始）
        result.paste(quad, (0, 0))

        # 右上角（水平镜像）
        result.paste(ImageOps.mirror(quad), (center_x, 0))

        # 左下角（垂直镜像）
        result.paste(ImageOps.flip(quad), (0, center_y))

        # 右下角（水平和垂直镜像）
        result.paste(ImageOps.flip(ImageOps.mirror(quad)), (center_x, center_y))

        return result

    else:
        return frame


def create_kaleidoscope_animation(
    base_frame: Image.Image | None = None,
    num_frames: int = 30,
    segments: int = 8,
    rotation_speed: float = 1.0,
    width: int = 480,
    height: int = 480,
) -> list[Image.Image]:
    """
    创建动态万花筒效果。

    参数:
        base_frame: 要应用效果的帧（None 表示使用演示图案）
        num_frames: 帧数
        segments: 万花筒段数
        rotation_speed: 图案旋转速度（0.5-2.0）
        width: 如果生成演示图案时的帧宽度
        height: 如果生成演示图案时的帧高度

    返回:
        应用了万花筒效果的帧列表
    """
    frames = []

    # 如果没有基础帧则创建演示图案
    if base_frame is None:
        base_frame = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(base_frame)

        # 绘制一些彩色形状
        from core.color_palettes import get_palette

        palette = get_palette("vibrant")

        colors = [palette["primary"], palette["secondary"], palette["accent"]]

        for i, color in enumerate(colors):
            x = width // 2 + int(100 * math.cos(i * 2 * math.pi / 3))
            y = height // 2 + int(100 * math.sin(i * 2 * math.pi / 3))
            draw.ellipse([x - 40, y - 40, x + 40, y + 40], fill=color)

    # 旋转基础帧并应用万花筒效果
    for i in range(num_frames):
        angle = (i / num_frames) * 360 * rotation_speed

        # 旋转基础帧
        rotated = base_frame.rotate(angle, resample=Image.BICUBIC)

        # 应用万花筒效果
        kaleido_frame = apply_kaleidoscope(rotated, segments=segments)

        frames.append(kaleido_frame)

    return frames


# 示例用法
if __name__ == "__main__":
    from core.gif_builder import GIFBuilder

    print("正在创建万花筒 GIF...")

    builder = GIFBuilder(width=480, height=480, fps=20)

    # 创建万花筒动画
    frames = create_kaleidoscope_animation(num_frames=40, segments=8, rotation_speed=0.5)

    builder.add_frames(frames)
    builder.save("kaleidoscope_test.gif", num_colors=128)
