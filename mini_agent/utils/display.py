"""终端颜色和显示格式化工具。

本模块提供统一的 Colors 类，用于终端输出格式化，
可替代 agent.py 和 cli.py 中的重复实现。
"""

from typing import Final


class Colors:
    """终端颜色定义，使用 ANSI 转义码。

    本类为整个应用程序提供一致的颜色格式化。
    所有颜色使用大多数现代终端支持的 标准 ANSI 转义码。

    用法:
        from mini_agent.utils.display import Colors

        print(f"{Colors.RED}错误消息{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}成功！{Colors.RESET}")

    颜色分类:
        - Basic: 标准 8 种颜色（黑、红、绿、黄、蓝、品红、青、白）
        - Bright: 基本颜色的高强度版本
        - Background: 背景颜色变体
        - Style: 文本样式（粗体、暗色、重置）
    """

    # 样式修饰符
    RESET: Final[str] = "\033[0m"
    BOLD: Final[str] = "\033[1m"
    DIM: Final[str] = "\033[2m"

    # 基本前景色
    BLACK: Final[str] = "\033[30m"
    RED: Final[str] = "\033[31m"
    GREEN: Final[str] = "\033[32m"
    YELLOW: Final[str] = "\033[33m"
    BLUE: Final[str] = "\033[34m"
    MAGENTA: Final[str] = "\033[35m"
    CYAN: Final[str] = "\033[36m"
    WHITE: Final[str] = "\033[37m"

    # 明亮前景色
    BRIGHT_BLACK: Final[str] = "\033[90m"
    BRIGHT_RED: Final[str] = "\033[91m"
    BRIGHT_GREEN: Final[str] = "\033[92m"
    BRIGHT_YELLOW: Final[str] = "\033[93m"
    BRIGHT_BLUE: Final[str] = "\033[94m"
    BRIGHT_MAGENTA: Final[str] = "\033[95m"
    BRIGHT_CYAN: Final[str] = "\033[96m"
    BRIGHT_WHITE: Final[str] = "\033[97m"

    # 背景色
    BG_RED: Final[str] = "\033[41m"
    BG_GREEN: Final[str] = "\033[42m"
    BG_YELLOW: Final[str] = "\033[43m"
    BG_BLUE: Final[str] = "\033[44m"

    # 便捷组合样式
    # 注意: 组合样式如 BOLD + COLOR 需要一起使用
    @staticmethod
    def bold(text: str) -> str:
        """为文本应用粗体样式。"""
        return f"{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def dim(text: str) -> str:
        """为文本应用暗色样式。"""
        return f"{Colors.DIM}{text}{Colors.RESET}"

    @staticmethod
    def success(text: str) -> str:
        """为文本应用成功颜色（绿色）。"""
        return f"{Colors.BRIGHT_GREEN}{text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        """为文本应用警告颜色（黄色）。"""
        return f"{Colors.BRIGHT_YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        """为文本应用错误颜色（红色）。"""
        return f"{Colors.BRIGHT_RED}{text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        """为文本应用信息颜色（青色）。"""
        return f"{Colors.BRIGHT_CYAN}{text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        """为文本应用标题样式（粗体 + 青色）。"""
        return f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}"


class BoxDrawing:
    """终端 UI 元素的框线字符。

    提供跨平台一致工作的框线字符。
    在 Unicode 支持有限的平台上回退到 ASCII 替代方案。
    """

    # 框线字符（使用 Unicode 以获得更好的视觉效果，可用的 ASCII 替代方案）
    HORIZONTAL: Final[str] = "─"
    VERTICAL: Final[str] = "│"
    TOP_LEFT: Final[str] = "┌"
    TOP_RIGHT: Final[str] = "┐"
    BOTTOM_LEFT: Final[str] = "└"
    BOTTOM_RIGHT: Final[str] = "┘"
    CROSS: Final[str] = "┼"
    TOP_T: Final[str] = "┬"
    BOTTOM_T: Final[str] = "┴"
    LEFT_T: Final[str] = "├"
    RIGHT_T: Final[str] = "┤"

    # 粗线变体
    THICK_HORIZONTAL: Final[str] = "═"
    THICK_VERTICAL: Final[str] = "║"
    THICK_TOP_LEFT: Final[str] = "╔"
    THICK_TOP_RIGHT: Final[str] = "╗"
    THICK_BOTTOM_LEFT: Final[str] = "╚"
    THICK_BOTTOM_RIGHT: Final[str] = "╝"

    @classmethod
    def horizontal_line(cls, width: int) -> str:
        """生成指定宽度的水平线。"""
        return cls.HORIZONTAL * width

    @classmethod
    def thick_horizontal_line(cls, width: int) -> str:
        """生成指定宽度的粗水平线。"""
        return cls.THICK_HORIZONTAL * width

    @classmethod
    def box(cls, width: int, height: int = 1) -> list[str]:
        """生成指定尺寸的简单框。

        Args:
            width: 框的宽度（字符数）
            height: 框的高度（行数）

        Returns:
            表示框每一行的字符串列表
        """
        lines = []
        if height >= 1:
            lines.append(f"{cls.TOP_LEFT}{cls.HORIZONTAL * (width - 2)}{cls.TOP_RIGHT}")
        for _ in range(height - 2):
            lines.append(f"{cls.VERTICAL}{' ' * (width - 2)}{cls.VERTICAL}")
        if height >= 2:
            lines.append(f"{cls.BOTTOM_LEFT}{cls.HORIZONTAL * (width - 2)}{cls.BOTTOM_RIGHT}")
        return lines


def colorize(text: str, color: str, bold: bool = False) -> str:
    """为文本应用颜色，可选择是否添加粗体样式。

    Args:
        text: 要着色的文本
        color: 颜色名称（例如 "red"、"green"、"bright_yellow"）
        bold: 是否应用粗体样式

    Returns:
        着色后的文本，末尾附加重置代码
    """
    color_code = getattr(Colors, color.upper().replace("-", "_"), Colors.WHITE)
    prefix = Colors.BOLD if bold else ""
    return f"{prefix}{color_code}{text}{Colors.RESET}"


def create_progress_bar(
    current: float,
    total: float,
    width: int = 30,
    prefix: str = "",
    show_percentage: bool = True,
) -> str:
    """创建基于文本的进度条。

    Args:
        current: 当前进度值
        total: 总值（100%）
        width: 进度条的宽度（字符数）
        prefix: 进度条前显示的文本
        show_percentage: 是否显示百分比

    Returns:
        格式化的进度条字符串

    示例:
        >>> create_progress_bar(75, 100, width=20)
        '███████░░░░░░░░░░░ 75%'
    """
    if total <= 0:
        percentage: float = 0
        filled = 0
    else:
        percentage = min(100, max(0, (current / total) * 100))
        filled = int((current / total) * width) if total > 0 else 0

    bar = "█" * filled + "░" * (width - filled)

    if show_percentage:
        return f"{prefix}{bar} {percentage:.1f}%"
    else:
        return f"{prefix}{bar}"


def format_table_row(
    columns: list[str],
    widths: list[int],
    alignments: list[str] | None = None,
) -> str:
    """格式化表格行，具有适当的列宽和对齐方式。

    Args:
        columns: 列值列表
        widths: 列宽列表（必须与列数匹配）
        alignments: 对齐方式列表（"left"、"right"、"center"），默认为 "left"

    Returns:
        格式化的表格行字符串
    """
    from mini_agent.utils.terminal_utils import calculate_display_width

    if alignments is None:
        alignments = ["left"] * len(columns)

    formatted_cols = []
    for col, width, align in zip(columns, widths, alignments):
        col_width = calculate_display_width(col)
        if col_width > width:
            col = col[:width]
            col_width = width

        if align == "right":
            formatted_cols.append(" " * (width - col_width) + col)
        elif align == "center":
            padding = width - col_width
            left_pad = padding // 2
            right_pad = padding - left_pad
            formatted_cols.append(" " * left_pad + col + " " * right_pad)
        else:  # left
            formatted_cols.append(col + " " * (width - col_width))

    return "│ " + " │ ".join(formatted_cols) + " │"


def create_divider(char: str = "─", width: int = 60) -> str:
    """创建水平分隔线。

    Args:
        char: 用于绘制线的字符
        width: 分隔线的总宽度

    Returns:
        分隔线字符串
    """
    return char * width
