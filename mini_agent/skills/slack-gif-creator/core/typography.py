"""
排版系统 - 带有轮廓、阴影和效果的专业文本渲染。

本模块提供高质量的文本渲染，在GIF中看起来清晰而专业，
具有轮廓以提高可读性和效果以增强视觉冲击力。
"""

from PIL import Image, ImageDraw, ImageFont

# 排版比例 - 比例 sizing 系统
TYPOGRAPHY_SCALE = {
    "h1": 60,  # 大标题
    "h2": 48,  # 中等标题
    "h3": 36,  # 小标题
    "title": 50,  # 标题文本
    "body": 28,  # 正文文本
    "small": 20,  # 小文本
    "tiny": 16,  # 极小文本
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    获取支持回退的字体。

    Args:
        size: 字体大小（像素）
        bold: 如果可用则使用粗体变体

    Returns:
        ImageFont 对象
    """
    # 尝试多个字体路径以支持跨平台
    font_paths = [
        # macOS 字体
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SF-Pro.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        # Linux 字体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Windows 字体
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue

    # 最终回退
    return ImageFont.load_default()


def draw_text_with_outline(
    frame: Image.Image,
    text: str,
    position: tuple[int, int],
    font_size: int = 40,
    text_color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 3,
    centered: bool = False,
    bold: bool = True,
) -> Image.Image:
    """
    绘制带轮廓的文本以获得最大可读性。

    这是GIF中专业外观文本最重要的函数。
    轮廓确保文本在任何背景上都可读。

    Args:
        frame: 要绘制的PIL图像
        text: 要绘制的文本
        position: (x, y) 位置
        font_size: 字体大小（像素）
        text_color: 文本填充的RGB颜色
        outline_color: 轮廓的RGB颜色
        outline_width: 轮廓宽度（像素）（建议2-4）
        centered: 如果为True，在位置处居中文本
        bold: 使用粗体字体变体

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    font = get_font(font_size, bold=bold)

    # 计算居中的位置
    if centered:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = position[0] - text_width // 2
        y = position[1] - text_height // 2
        position = (x, y)

    # 通过在所有方向偏移多次绘制文本来绘制轮廓
    x, y = position
    for offset_x in range(-outline_width, outline_width + 1):
        for offset_y in range(-outline_width, outline_width + 1):
            if offset_x != 0 or offset_y != 0:
                draw.text((x + offset_x, y + offset_y), text, fill=outline_color, font=font)

    # 在上面绘制主文本
    draw.text(position, text, fill=text_color, font=font)

    return frame


def draw_text_with_shadow(
    frame: Image.Image,
    text: str,
    position: tuple[int, int],
    font_size: int = 40,
    text_color: tuple[int, int, int] = (255, 255, 255),
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    shadow_offset: tuple[int, int] = (3, 3),
    centered: bool = False,
    bold: bool = True,
) -> Image.Image:
    """
    绘制带投影的文本以增加深度。

    Args:
        frame: 要绘制的PIL图像
        text: 要绘制的文本
        position: (x, y) 位置
        font_size: 字体大小（像素）
        text_color: 文本的RGB颜色
        shadow_color: 阴影的RGB颜色
        shadow_offset: 阴影的 (x, y) 偏移
        centered: 如果为True，在位置处居中文本
        bold: 使用粗体字体变体

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    font = get_font(font_size, bold=bold)

    # 计算居中的位置
    if centered:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = position[0] - text_width // 2
        y = position[1] - text_height // 2
        position = (x, y)

    # 绘制阴影
    shadow_pos = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
    draw.text(shadow_pos, text, fill=shadow_color, font=font)

    # 绘制主文本
    draw.text(position, text, fill=text_color, font=font)

    return frame


def draw_text_with_glow(
    frame: Image.Image,
    text: str,
    position: tuple[int, int],
    font_size: int = 40,
    text_color: tuple[int, int, int] = (255, 255, 255),
    glow_color: tuple[int, int, int] = (255, 200, 0),
    glow_radius: int = 5,
    centered: bool = False,
    bold: bool = True,
) -> Image.Image:
    """
    绘制带发光效果的文本以增强强调。

    Args:
        frame: 要绘制的PIL图像
        text: 要绘制的文本
        position: (x, y) 位置
        font_size: 字体大小（像素）
        text_color: 文本的RGB颜色
        glow_color: 发光的RGB颜色
        glow_radius: 发光效果的半径
        centered: 如果为True，在位置处居中文本
        bold: 使用粗体字体变体

    Returns:
        修改后的帧
    """
    draw = ImageDraw.Draw(frame)
    font = get_font(font_size, bold=bold)

    # 计算居中的位置
    if centered:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = position[0] - text_width // 2
        y = position[1] - text_height // 2
        position = (x, y)

    # 使用在不同偏移处相同颜色绘制发光层来模拟降低的不透明度
    x, y = position
    for radius in range(glow_radius, 0, -1):
        for offset_x in range(-radius, radius + 1):
            for offset_y in range(-radius, radius + 1):
                if offset_x != 0 or offset_y != 0:
                    draw.text((x + offset_x, y + offset_y), text, fill=glow_color, font=font)

    # 绘制主文本
    draw.text(position, text, fill=text_color, font=font)

    return frame


def draw_text_in_box(
    frame: Image.Image,
    text: str,
    position: tuple[int, int],
    font_size: int = 40,
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (0, 0, 0),
    box_alpha: float = 0.7,
    padding: int = 10,
    centered: bool = True,
    bold: bool = True,
) -> Image.Image:
    """
    在半透明框中绘制文本以保证可读性。

    Args:
        frame: 要绘制的PIL图像
        text: 要绘制的文本
        position: (x, y) 位置
        font_size: 字体大小（像素）
        text_color: 文本的RGB颜色
        box_color: 背景框的RGB颜色
        box_alpha: 框的不透明度 (0.0-1.0)
        padding: 文本周围的填充（像素）
        centered: 如果为True，在位置处居中
        bold: 使用粗体字体变体

    Returns:
        修改后的帧
    """
    # 为带alpha的框创建一个单独的层
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(frame)

    font = get_font(font_size, bold=bold)

    # 获取文本尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 计算框的位置
    if centered:
        box_x = position[0] - (text_width + padding * 2) // 2
        box_y = position[1] - (text_height + padding * 2) // 2
        text_x = position[0] - text_width // 2
        text_y = position[1] - text_height // 2
    else:
        box_x = position[0] - padding
        box_y = position[1] - padding
        text_x = position[0]
        text_y = position[1]

    # 绘制半透明框
    box_coords = [box_x, box_y, box_x + text_width + padding * 2, box_y + text_height + padding * 2]
    alpha_value = int(255 * box_alpha)
    draw_overlay.rectangle(box_coords, fill=(*box_color, alpha_value))

    # 将叠加层合成到帧上
    frame_rgba = frame.convert("RGBA")
    frame_rgba = Image.alpha_composite(frame_rgba, overlay)
    frame = frame_rgba.convert("RGB")

    # 在上面绘制文本
    draw = ImageDraw.Draw(frame)
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    return frame


def get_text_size(text: str, font_size: int, bold: bool = True) -> tuple[int, int]:
    """
    获取文本的尺寸而不绘制它。

    Args:
        text: 要测量的文本
        font_size: 字体大小（像素）
        bold: 使用粗体字体变体

    Returns:
        (width, height) 元组
    """
    font = get_font(font_size, bold=bold)
    # 创建临时图像以进行测量
    temp_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (width, height)


def get_optimal_font_size(text: str, max_width: int, max_height: int, start_size: int = 60) -> int:
    """
    找到在给定尺寸内适应的最大字体大小。

    Args:
        text: 要调整大小的文本
        max_width: 最大宽度（像素）
        max_height: 最大高度（像素）
        start_size: 尝试的起始字体大小

    Returns:
        最佳字体大小
    """
    font_size = start_size
    while font_size > 10:
        width, height = get_text_size(text, font_size)
        if width <= max_width and height <= max_height:
            return font_size
        font_size -= 2
    return 10  # 最小字体大小


def scale_font_for_frame(base_size: int, frame_width: int, frame_height: int) -> int:
    """
    按帧尺寸比例缩放字体大小。

    用于在不同的GIF尺寸中保持相对文本大小。

    Args:
        base_size: 480x480帧的基础字体大小
        frame_width: 实际帧宽度
        frame_height: 实际帧高度

    Returns:
        缩放后的字体大小
    """
    # 使用平均尺寸进行缩放
    avg_dimension = (frame_width + frame_height) / 2
    base_dimension = 480  # 参考尺寸
    scale_factor = avg_dimension / base_dimension
    return max(10, int(base_size * scale_factor))
