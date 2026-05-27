from __future__ import annotations

from mini_agent.utils.display import Colors


class TestColorsConstants:
    def test_reset_starts_with_ansi_escape(self):
        assert Colors.RESET.startswith("\033[")

    def test_bold_constant_starts_with_ansi_escape(self):
        assert Colors.BOLD.startswith("\033[")

    def test_dim_starts_with_ansi_escape(self):
        assert Colors.DIM.startswith("\033[")

    def test_black_starts_with_ansi_escape(self):
        assert Colors.BLACK.startswith("\033[")

    def test_red_starts_with_ansi_escape(self):
        assert Colors.RED.startswith("\033[")

    def test_green_starts_with_ansi_escape(self):
        assert Colors.GREEN.startswith("\033[")

    def test_yellow_starts_with_ansi_escape(self):
        assert Colors.YELLOW.startswith("\033[")

    def test_blue_starts_with_ansi_escape(self):
        assert Colors.BLUE.startswith("\033[")

    def test_magenta_starts_with_ansi_escape(self):
        assert Colors.MAGENTA.startswith("\033[")

    def test_cyan_starts_with_ansi_escape(self):
        assert Colors.CYAN.startswith("\033[")

    def test_white_starts_with_ansi_escape(self):
        assert Colors.WHITE.startswith("\033[")

    def test_bright_black_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_BLACK.startswith("\033[")

    def test_bright_red_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_RED.startswith("\033[")

    def test_bright_green_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_GREEN.startswith("\033[")

    def test_bright_yellow_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_YELLOW.startswith("\033[")

    def test_bright_blue_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_BLUE.startswith("\033[")

    def test_bright_magenta_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_MAGENTA.startswith("\033[")

    def test_bright_cyan_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_CYAN.startswith("\033[")

    def test_bright_white_starts_with_ansi_escape(self):
        assert Colors.BRIGHT_WHITE.startswith("\033[")

    def test_bg_red_starts_with_ansi_escape(self):
        assert Colors.BG_RED.startswith("\033[")

    def test_bg_green_starts_with_ansi_escape(self):
        assert Colors.BG_GREEN.startswith("\033[")

    def test_bg_yellow_starts_with_ansi_escape(self):
        assert Colors.BG_YELLOW.startswith("\033[")

    def test_bg_blue_starts_with_ansi_escape(self):
        assert Colors.BG_BLUE.startswith("\033[")

    def test_all_constants_are_non_empty(self):
        constants = [
            Colors.RESET,
            Colors.BOLD,
            Colors.DIM,
            Colors.BLACK,
            Colors.RED,
            Colors.GREEN,
            Colors.YELLOW,
            Colors.BLUE,
            Colors.MAGENTA,
            Colors.CYAN,
            Colors.WHITE,
            Colors.BRIGHT_BLACK,
            Colors.BRIGHT_RED,
            Colors.BRIGHT_GREEN,
            Colors.BRIGHT_YELLOW,
            Colors.BRIGHT_BLUE,
            Colors.BRIGHT_MAGENTA,
            Colors.BRIGHT_CYAN,
            Colors.BRIGHT_WHITE,
            Colors.BG_RED,
            Colors.BG_GREEN,
            Colors.BG_YELLOW,
            Colors.BG_BLUE,
        ]
        for c in constants:
            assert c != ""


class TestColorsStaticMethods:
    def test_bold_wraps_text(self):
        result = Colors.bold("hello")
        assert result.startswith(Colors.BOLD)
        assert result.endswith(Colors.RESET)
        assert "hello" in result

    def test_dim_wraps_text(self):
        result = Colors.dim("hello")
        assert result.startswith(Colors.DIM)
        assert result.endswith(Colors.RESET)
        assert "hello" in result

    def test_success_wraps_text(self):
        result = Colors.success("hello")
        assert result.startswith(Colors.BRIGHT_GREEN)
        assert result.endswith(Colors.RESET)
        assert "hello" in result

    def test_warning_wraps_text(self):
        result = Colors.warning("hello")
        assert result.startswith(Colors.BRIGHT_YELLOW)
        assert result.endswith(Colors.RESET)
        assert "hello" in result

    def test_error_wraps_text(self):
        result = Colors.error("hello")
        assert result.startswith(Colors.BRIGHT_RED)
        assert result.endswith(Colors.RESET)
        assert "hello" in result

    def test_info_wraps_text(self):
        result = Colors.info("hello")
        assert result.startswith(Colors.BRIGHT_CYAN)
        assert result.endswith(Colors.RESET)
        assert "hello" in result

    def test_bold_includes_bold_code(self):
        result = Colors.bold("test")
        assert Colors.BOLD in result

    def test_success_includes_bright_green_code(self):
        result = Colors.success("test")
        assert Colors.BRIGHT_GREEN in result

    def test_warning_includes_bright_yellow_code(self):
        result = Colors.warning("test")
        assert Colors.BRIGHT_YELLOW in result

    def test_error_includes_bright_red_code(self):
        result = Colors.error("test")
        assert Colors.BRIGHT_RED in result

    def test_info_includes_bright_cyan_code(self):
        result = Colors.info("test")
        assert Colors.BRIGHT_CYAN in result

    def test_header_includes_bold_and_cyan(self):
        result = Colors.header("test")
        assert Colors.BOLD in result
        assert Colors.CYAN in result
        assert result.endswith(Colors.RESET)
