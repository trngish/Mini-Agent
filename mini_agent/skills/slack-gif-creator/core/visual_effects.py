#!/usr/bin/env python3
"""
视觉效果 - 粒子系统、运动模糊、冲击效果及其他GIF特效。

本模块提供高质量的视觉效果，使动画更具专业感和动感，
同时保持合理的文件大小。
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class Particle:
    """粒子系统中的单个粒子。"""

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        lifetime: float,
        color: tuple[int, int, int],
        size: int = 3,
        shape: str = "circle",
    ):
        """
        初始化粒子。

        参数:
            x, y: 起始位置
            vx, vy: 速度分量
            lifetime: 粒子存活时间（以帧为单位）
            color: RGB颜色
            size: 粒子大小（像素）
            shape: 'circle'（圆形）、'square'（方形）或'star'（星形）
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color
        self.size = size
        self.shape = shape
        self.gravity = 0.5  # 每帧平方的像素加速度
        self.drag = 0.98  # 每帧的速度倍数

    def update(self):
        """更新粒子位置和生命周期。"""
        # 应用物理效果
        self.vy += self.gravity
        self.vx *= self.drag
        self.vy *= self.drag

        # 更新位置
        self.x += self.vx
        self.y += self.vy

        # 减少生命周期
        self.lifetime -= 1

    def is_alive(self) -> bool:
        """检查粒子是否仍然存活。"""
        return self.lifetime > 0

    def get_alpha(self) -> float:
        """根据生命周期获取粒子透明度。"""
        return max(0, min(1, self.lifetime / self.max_lifetime))

    def render(self, frame: Image.Image):
        """
        将粒子渲染到帧。

        参数:
            frame: 要绘制其上的PIL图像
        """
        if not self.is_alive():
            return

        draw = ImageDraw.Draw(frame)
        alpha = self.get_alpha()

        # 计算淡化的颜色
        color = tuple(int(c * alpha) for c in self.color)

        # 根据形状绘制
        x, y = int(self.x), int(self.y)
        size = max(1, int(self.size * alpha))

        if self.shape == "circle":
            bbox = [x - size, y - size, x + size, y + size]
            draw.ellipse(bbox, fill=color)
        elif self.shape == "square":
            bbox = [x - size, y - size, x + size, y + size]
            draw.rectangle(bbox, fill=color)
        elif self.shape == "star":
            # 简单的4点星形
            points = [
                (x, y - size),
                (x - size // 2, y),
                (x, y),
                (x, y + size),
                (x, y),
                (x + size // 2, y),
            ]
            draw.line(points, fill=color, width=2)


class ParticleSystem:
    """管理粒子集合。"""

    def __init__(self):
        """初始化粒子系统。"""
        self.particles: list[Particle] = []

    def emit(
        self,
        x: int,
        y: int,
        count: int = 10,
        spread: float = 2.0,
        speed: float = 5.0,
        color: tuple[int, int, int] = (255, 200, 0),
        lifetime: float = 20.0,
        size: int = 3,
        shape: str = "circle",
    ):
        """
        发射一波粒子。

        参数:
            x, y: 发射位置
            count: 发射粒子数量
            spread: 角度扩散（弧度）
            speed: 初始速度
            color: 粒子颜色
            lifetime: 粒子生命周期（帧）
            size: 粒子大小
            shape: 粒子形状
        """
        for _ in range(count):
            # 随机角度和速度
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(speed * 0.5, speed * 1.5)
            vx = math.cos(angle) * vel_mag
            vy = math.sin(angle) * vel_mag

            # 随机生命周期变化
            life = random.uniform(lifetime * 0.7, lifetime * 1.3)

            particle = Particle(x, y, vx, vy, life, color, size, shape)
            self.particles.append(particle)

    def emit_confetti(self, x: int, y: int, count: int = 20, colors: list[tuple[int, int, int]] | None = None):
        """
        发射彩纸粒子（彩色、下落）。

        参数:
            x, y: 发射位置
            count: 彩纸数量
            colors: 颜色列表（如果为None则随机）
        """
        if colors is None:
            colors = [
                (255, 107, 107),
                (255, 159, 64),
                (255, 218, 121),
                (107, 185, 240),
                (162, 155, 254),
                (255, 182, 193),
            ]

        for _ in range(count):
            color = random.choice(colors)
            vx = random.uniform(-3, 3)
            vy = random.uniform(-8, -2)
            shape = random.choice(["square", "circle"])
            size = random.randint(2, 4)
            lifetime = random.uniform(40, 60)

            particle = Particle(x, y, vx, vy, lifetime, color, size, shape)
            particle.gravity = 0.3  # 彩纸的重力较小
            self.particles.append(particle)

    def emit_sparkles(self, x: int, y: int, count: int = 15):
        """
        发射闪烁粒子（闪烁的星星）。

        参数:
            x, y: 发射位置
            count: 闪烁数量
        """
        colors = [(255, 255, 200), (255, 255, 255), (255, 255, 150)]

        for _ in range(count):
            color = random.choice(colors)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.uniform(15, 30)

            particle = Particle(x, y, vx, vy, lifetime, color, 2, "star")
            particle.gravity = 0
            particle.drag = 0.95
            self.particles.append(particle)

    def update(self):
        """更新所有粒子。"""
        # 更新存活粒子
        for particle in self.particles:
            particle.update()

        # 移除死亡粒子
        self.particles = [p for p in self.particles if p.is_alive()]

    def render(self, frame: Image.Image):
        """将所有粒子渲染到帧。"""
        for particle in self.particles:
            particle.render(frame)

    def get_particle_count(self) -> int:
        """获取活跃粒子数量。"""
        return len(self.particles)


def add_motion_blur(frame: Image.Image, prev_frame: Image.Image | None, blur_amount: float = 0.5) -> Image.Image:
    """
    通过与前一帧混合添加运动模糊。

    参数:
        frame: 当前帧
        prev_frame: 上一帧（第一帧时为None）
        blur_amount: 模糊量（0.0-1.0）

    返回:
        应用运动模糊后的帧
    """
    if prev_frame is None:
        return frame

    # 将当前帧与上一帧混合
    frame_array = np.array(frame, dtype=np.float32)
    prev_array = np.array(prev_frame, dtype=np.float32)

    blended = frame_array * (1 - blur_amount) + prev_array * blur_amount
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def create_impact_flash(
    frame: Image.Image, position: tuple[int, int], radius: int = 100, intensity: float = 0.7
) -> Image.Image:
    """
    在冲击点创建明亮的闪光效果。

    参数:
        frame: 要绘制其上的PIL图像
        position: 闪光中心
        radius: 闪光半径
        intensity: 闪光强度（0.0-1.0）

    返回:
        修改后的帧
    """
    # 创建叠加层
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = position

    # 绘制同心圆，透明度递减
    num_circles = 5
    for i in range(num_circles):
        alpha = int(255 * intensity * (1 - i / num_circles))
        r = radius * (1 - i / num_circles)
        color = (255, 255, 240, alpha)  # 暖白色

        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill=color)

    # 合成到帧上
    frame_rgba = frame.convert("RGBA")
    frame_rgba = Image.alpha_composite(frame_rgba, overlay)
    return frame_rgba.convert("RGB")


def create_shockwave_rings(
    frame: Image.Image,
    position: tuple[int, int],
    radii: list[int],
    color: tuple[int, int, int] = (255, 200, 0),
    width: int = 3,
) -> Image.Image:
    """
    创建扩展环效果。

    参数:
        frame: 要绘制其上的PIL图像
        position: 环中心
        radii: 环半径列表
        color: 环颜色
        width: 环宽度

    返回:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    x, y = position

    for radius in radii:
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=color, width=width)

    return frame


def create_explosion_effect(
    frame: Image.Image,
    position: tuple[int, int],
    radius: int,
    progress: float,
    color: tuple[int, int, int] = (255, 150, 0),
) -> Image.Image:
    """
    创建扩展并淡出的爆炸效果。

    参数:
        frame: 要绘制其上的PIL图像
        position: 爆炸中心
        radius: 最大半径
        progress: 动画进度（0.0-1.0）
        color: 爆炸颜色

    返回:
        修改后的帧
    """
    current_radius = int(radius * progress)
    fade = 1 - progress

    # 创建叠加层
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = position

    # 绘制带淡出的扩展圆
    alpha = int(255 * fade)
    r, g, b = color
    circle_color = (r, g, b, alpha)

    bbox = [x - current_radius, y - current_radius, x + current_radius, y + current_radius]
    draw.ellipse(bbox, fill=circle_color)

    # 合成
    frame_rgba = frame.convert("RGBA")
    frame_rgba = Image.alpha_composite(frame_rgba, overlay)
    return frame_rgba.convert("RGB")


def add_glow_effect(
    frame: Image.Image, mask_color: tuple[int, int, int], glow_color: tuple[int, int, int], blur_radius: int = 10
) -> Image.Image:
    """
    为特定颜色的区域添加发光效果。

    参数:
        frame: PIL图像
        mask_color: 要创建发光效果的颜色
        glow_color: 发光颜色
        blur_radius: 模糊量

    返回:
        带发光效果的帧
    """
    # 创建目标颜色的蒙版
    frame_array = np.array(frame)
    mask = np.all(frame_array == mask_color, axis=-1)

    # 创建发光层
    glow = Image.new("RGB", frame.size, (0, 0, 0))
    glow_array = np.array(glow)
    glow_array[mask] = glow_color
    glow = Image.fromarray(glow_array)

    # 模糊发光层
    glow = glow.filter(ImageFilter.GaussianBlur(blur_radius))

    # 与原图混合
    blended = Image.blend(frame, glow, 0.5)
    return blended


def add_drop_shadow(
    frame: Image.Image,
    object_bounds: tuple[int, int, int, int],
    shadow_offset: tuple[int, int] = (5, 5),
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    blur: int = 5,
) -> Image.Image:
    """
    为对象添加投影效果。

    参数:
        frame: PIL图像
        object_bounds: 对象的(x1, y1, x2, y2)边界
        shadow_offset: 阴影的(x, y)偏移量
        shadow_color: 阴影颜色
        blur: 阴影模糊量

    返回:
        带阴影的帧
    """
    # 提取对象
    x1, y1, x2, y2 = object_bounds
    obj = frame.crop((x1, y1, x2, y2))

    # 创建阴影
    shadow = Image.new("RGBA", obj.size, (*shadow_color, 180))

    # 创建带透明度的帧
    frame_rgba = frame.convert("RGBA")

    # 粘贴阴影
    shadow_pos = (x1 + shadow_offset[0], y1 + shadow_offset[1])
    frame_rgba.paste(shadow, shadow_pos, shadow)

    # 在顶部粘贴对象
    frame_rgba.paste(obj, (x1, y1))

    return frame_rgba.convert("RGB")


def create_speed_lines(
    frame: Image.Image,
    position: tuple[int, int],
    direction: float,
    length: int = 50,
    count: int = 5,
    color: tuple[int, int, int] = (200, 200, 200),
) -> Image.Image:
    """
    为运动效果创建速度线。

    参数:
        frame: 要绘制其上的PIL图像
        position: 中心位置
        direction: 角度（弧度，0=向右，pi/2=向下）
        length: 线长度
        count: 线条数量
        color: 线条颜色

    返回:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    x, y = position

    # 反方向（线条拖在后面）
    trail_angle = direction + math.pi

    for _i in range(count):
        # 与中心的偏移
        offset_angle = trail_angle + random.uniform(-0.3, 0.3)
        offset_dist = random.uniform(10, 30)
        start_x = x + math.cos(offset_angle) * offset_dist
        start_y = y + math.sin(offset_angle) * offset_dist

        # 终点
        line_length = random.uniform(length * 0.7, length * 1.3)
        end_x = start_x + math.cos(trail_angle) * line_length
        end_y = start_y + math.sin(trail_angle) * line_length

        # 绘制具有不同透明度的线条
        random.randint(100, 200)
        width = random.randint(1, 3)

        # 简单线条（模拟完全不透明）
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)

    return frame


def create_screen_shake_offset(intensity: int, frame_index: int) -> tuple[int, int]:
    """
    计算帧的画面震动偏移量。

    参数:
        intensity: 震动强度（像素）
        frame_index: 当前帧编号

    返回:
        (x, y) 偏移量元组
    """
    # 使用帧索引来获得确定性的但看起来随机地震动
    random.seed(frame_index)
    offset_x = random.randint(-intensity, intensity)
    offset_y = random.randint(-intensity, intensity)
    random.seed()  # 重置随机种子
    return (offset_x, offset_y)


def apply_screen_shake(frame: Image.Image, intensity: int, frame_index: int) -> Image.Image:
    """
    对整个帧应用画面震动效果。

    参数:
        frame: PIL图像
        intensity: 震动强度
        frame_index: 当前帧编号

    返回:
        震动后的帧
    """
    offset_x, offset_y = create_screen_shake_offset(intensity, frame_index)

    # 创建带背景的新帧
    shaken = Image.new("RGB", frame.size, (0, 0, 0))

    # 用偏移量粘贴原始帧
    shaken.paste(frame, (offset_x, offset_y))

    return shaken
