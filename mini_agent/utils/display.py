"""Terminal color and display formatting utilities.

This module provides a unified Colors class for terminal output formatting,
replacing duplicate implementations across agent.py and cli.py.
"""

from typing import Final


class Colors:
    """Terminal color definitions with ANSI escape codes.

    This class provides consistent color formatting across the entire application.
    All colors use standard ANSI escape codes supported by most modern terminals.

    Usage:
        from mini_agent.utils.display import Colors

        print(f"{Colors.RED}Error message{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}Success!{Colors.RESET}")

    Color Categories:
        - Basic: Standard 8 colors (black, red, green, yellow, blue, magenta, cyan, white)
        - Bright: High-intensity versions of basic colors
        - Background: Background color variants
        - Style: Text styling (bold, dim, reset)
    """

    # Style modifiers
    RESET: Final[str] = "\033[0m"
    BOLD: Final[str] = "\033[1m"
    DIM: Final[str] = "\033[2m"

    # Basic foreground colors
    BLACK: Final[str] = "\033[30m"
    RED: Final[str] = "\033[31m"
    GREEN: Final[str] = "\033[32m"
    YELLOW: Final[str] = "\033[33m"
    BLUE: Final[str] = "\033[34m"
    MAGENTA: Final[str] = "\033[35m"
    CYAN: Final[str] = "\033[36m"
    WHITE: Final[str] = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK: Final[str] = "\033[90m"
    BRIGHT_RED: Final[str] = "\033[91m"
    BRIGHT_GREEN: Final[str] = "\033[92m"
    BRIGHT_YELLOW: Final[str] = "\033[93m"
    BRIGHT_BLUE: Final[str] = "\033[94m"
    BRIGHT_MAGENTA: Final[str] = "\033[95m"
    BRIGHT_CYAN: Final[str] = "\033[96m"
    BRIGHT_WHITE: Final[str] = "\033[97m"

    # Background colors
    BG_RED: Final[str] = "\033[41m"
    BG_GREEN: Final[str] = "\033[42m"
    BG_YELLOW: Final[str] = "\033[43m"
    BG_BLUE: Final[str] = "\033[44m"

    # Convenience combined styles
    # Note: Combined styles like BOLD + COLOR need to be used together
    @staticmethod
    def bold(text: str) -> str:
        """Apply bold styling to text."""
        return f"{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def dim(text: str) -> str:
        """Apply dim styling to text."""
        return f"{Colors.DIM}{text}{Colors.RESET}"

    @staticmethod
    def success(text: str) -> str:
        """Apply success color (green) to text."""
        return f"{Colors.BRIGHT_GREEN}{text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        """Apply warning color (yellow) to text."""
        return f"{Colors.BRIGHT_YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        """Apply error color (red) to text."""
        return f"{Colors.BRIGHT_RED}{text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        """Apply info color (cyan) to text."""
        return f"{Colors.BRIGHT_CYAN}{text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        """Apply header styling (bold + cyan) to text."""
        return f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}"


class BoxDrawing:
    """Box drawing characters for terminal UI elements.

    Provides consistent box drawing characters that work across platforms.
    Falls back to ASCII alternatives on platforms with limited Unicode support.
    """

    # Box characters (using Unicode for better visuals, ASCII fallbacks available)
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

    # Thick variants
    THICK_HORIZONTAL: Final[str] = "═"
    THICK_VERTICAL: Final[str] = "║"
    THICK_TOP_LEFT: Final[str] = "╔"
    THICK_TOP_RIGHT: Final[str] = "╗"
    THICK_BOTTOM_LEFT: Final[str] = "╚"
    THICK_BOTTOM_RIGHT: Final[str] = "╝"

    @classmethod
    def horizontal_line(cls, width: int) -> str:
        """Generate a horizontal line of specified width."""
        return cls.HORIZONTAL * width

    @classmethod
    def thick_horizontal_line(cls, width: int) -> str:
        """Generate a thick horizontal line of specified width."""
        return cls.THICK_HORIZONTAL * width

    @classmethod
    def box(cls, width: int, height: int = 1) -> list[str]:
        """Generate a simple box of specified dimensions.

        Args:
            width: Width of the box in characters
            height: Height of the box in lines

        Returns:
            List of strings representing each line of the box
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
    """Apply color to text with optional bold styling.

    Args:
        text: Text to colorize
        color: Color name (e.g., "red", "green", "bright_yellow")
        bold: Whether to apply bold styling

    Returns:
        Colorized text with reset code appended
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
    """Create a text-based progress bar.

    Args:
        current: Current progress value
        total: Total value (100%)
        width: Width of the progress bar in characters
        prefix: Text to display before the progress bar
        show_percentage: Whether to show percentage

    Returns:
        Formatted progress bar string

    Example:
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
    """Format a table row with proper column widths and alignment.

    Args:
        columns: List of column values
        widths: List of column widths (must match columns length)
        alignments: List of alignments ("left", "right", "center"), defaults to "left"

    Returns:
        Formatted table row string
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
    """Create a horizontal divider line.

    Args:
        char: Character to use for the line
        width: Total width of the divider

    Returns:
        Divider string
    """
    return char * width
