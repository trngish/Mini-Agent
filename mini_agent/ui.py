"""Mini Agent 终端 UI 辅助工具。"""

from __future__ import annotations

import io
import platform
import subprocess  # nosec B404
import sys
from datetime import datetime
from pathlib import Path

from .agent import Agent
from .utils import Colors

SEP = f"{Colors.DIM}┄{'─' * 56}┄{Colors.RESET}"

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


def print_image(width: int = 80) -> bool:
    """在终端中使用 ANSI 真彩色和半块字符显示 Logo 图片。

    先尝试 SVG，如失败则回退到传统的 WEBP Logo。
    """
    from PIL import Image

    config_dir = Path(__file__).parent / "config"
    svg_path = config_dir / "logo.svg"
    webp_path = config_dir / "minimax.webp"

    img: Image.Image | None = None
    try:
        if svg_path.exists():
            # 尝试通过 cairosvg 或 svglib 将 SVG 光栅化
            try:
                import cairosvg  # type: ignore[import-not-found]

                png_bytes = cairosvg.svg2png(
                    url=str(svg_path),
                    output_width=width,
                    output_height=int(width),
                )
                from io import BytesIO

                img = Image.open(BytesIO(png_bytes)).convert("RGBA")
            except ImportError:
                try:
                    from reportlab.graphics import renderPM  # type: ignore[import-untyped]
                    from svglib.svglib import svg2rlg  # type: ignore[import-not-found]

                    drawing = svg2rlg(str(svg_path))
                    img = renderPM.drawToPIL(drawing)
                except ImportError:
                    pass
        if img is None and webp_path.exists():
            img = Image.open(webp_path).convert("RGBA")
    except Exception:
        return False

    if img is None:
        return False

    try:
        bg = Image.new("RGBA", img.size, (15, 23, 42, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        aspect = img.height / img.width
        height = int(width * aspect * 0.5)
        if height < 1:
            return False
        resample = getattr(Image, "Resampling", getattr(Image, "LANCZOS", Image.Resampling.BICUBIC))
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.LANCZOS
        img = img.resize((width, height * 2), resample)
        pixels = img.load()
        for y in range(0, img.height, 2):
            parts = []
            for x in range(img.width):
                r1, g1, b1 = pixels[x, y]  # type: ignore[index,misc]
                r2, g2, b2 = pixels[x, y + 1]  # type: ignore[index,misc]
                parts.append(f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m\u2580")
            print("".join(parts) + "\x1b[0m")
        return True
    except Exception:
        return False


def get_log_directory() -> Path:
    return Path.home() / ".mini-agent" / "log"


def show_log_directory(open_file_manager: bool = True) -> None:
    log_dir = get_log_directory()
    if not log_dir.exists():
        print(f"\n{Colors.YELLOW}日志目录尚不存在。{Colors.RESET}\n")
        return
    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    print(f"\n{Colors.BRIGHT_CYAN}📁 Log Directory: {log_dir}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}日志文件总数: {len(log_files)}{Colors.RESET}\n")
    if log_files:
        print(f"{Colors.BRIGHT_CYAN}最近的日志文件:{Colors.RESET}")
        for f in log_files[:10]:
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size = f.stat().st_size
            print(f"  {Colors.DIM}{mtime}{Colors.RESET}  {Colors.GREEN}{f.name}{Colors.RESET}  ({size} bytes)")
    if len(log_files) > 10:
        print(f"  {Colors.DIM}... 还有 {len(log_files) - 10} 个文件{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    if open_file_manager:
        _open_directory_in_file_manager(log_dir)
    print()


def _open_directory_in_file_manager(directory: Path) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(directory)], check=False)
        elif system == "Windows":
            subprocess.run(["explorer", str(directory)], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(directory)], check=False)
    except FileNotFoundError:
        print(f"{Colors.YELLOW}无法打开文件管理器。{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.YELLOW}打开文件管理器时出错: {e}{Colors.RESET}")


def read_log_file(filename: str) -> None:
    log_dir = get_log_directory()
    log_file = log_dir / filename
    if not log_file.exists() or not log_file.is_file():
        print(f"\n{Colors.RED}❌ 找不到日志文件: {log_file}{Colors.RESET}\n")
        return
    print(f"\n{Colors.BRIGHT_CYAN}📄 正在读取: {log_file}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
    try:
        with open(log_file, encoding="utf-8") as f:
            content = f.read()
        print(content)
        print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
        print(f"\n{Colors.GREEN}✅ 文件结束{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}❌ 读取文件时出错: {e}{Colors.RESET}\n")


def print_banner() -> None:
    """打印一个简洁的欢迎横幅，包含精简的彩色 ASCII Logo。

    无需图片依赖 - 纯终端文本配合 ANSI 颜色。
    """
    w = 52  # banner content width

    # 紧凑的 ASCII 艺术 Logo — 3 行，紫红色 (MINI) → 青色 (AGENT)
    logo = [
        f"  {Colors.BRIGHT_MAGENTA}╔╦╗╦╔╗╔╦{Colors.RESET}  {Colors.CYAN}╔═╗╔═╗╔═╗╔╗╔╔╦╗{Colors.RESET}",
        f"  {Colors.BRIGHT_MAGENTA}║║║║║║║║{Colors.RESET}  {Colors.CYAN}╠═╣║ ╦║╣ ║║║ ║ {Colors.RESET}",
        f"  {Colors.BRIGHT_MAGENTA}╩ ╩╩╝╚╝╩{Colors.RESET}  {Colors.CYAN}╩ ╩╚═╝╚═╝╝╚╝ ╩ {Colors.RESET}",
    ]
    logo_w = 27  # 可见宽度（不含 ANSI 转义码）: 2+8+2+15

    print()
    print(f"  {Colors.DIM}╭{'─' * w}╮{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET}{' ' * w}{Colors.DIM}│{Colors.RESET}")

    for line in logo:
        pad = w - logo_w
        print(f"  {Colors.DIM}│{Colors.RESET}{line}{' ' * pad}{Colors.DIM}│{Colors.RESET}")

    print(f"  {Colors.DIM}│{Colors.RESET}{' ' * w}{Colors.DIM}│{Colors.RESET}")

    tips = (
        f"  Type a task  {Colors.DIM}·{Colors.RESET}"
        f"  /help  {Colors.DIM}·{Colors.RESET}"
        f"  Tab  {Colors.DIM}·{Colors.RESET}"
        f"  Ctrl+C exit"
    )
    tips_w = 47  # 可见宽度（不含 ANSI 转义码）
    print(f"  {Colors.DIM}│{Colors.RESET}{tips}{' ' * (w - tips_w)}{Colors.DIM}│{Colors.RESET}")

    print(f"  {Colors.DIM}│{Colors.RESET}{' ' * w}{Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}╰{'─' * w}╯{Colors.RESET}")
    print()


def print_help() -> None:
    """打印帮助信息。"""
    print(f"\n  {Colors.BOLD}{Colors.BRIGHT_WHITE}命令{Colors.RESET}")
    print(f"  {Colors.DIM}╌{'╌' * 40}╌{Colors.RESET}")
    for cmd, desc in [
        ("/help", "显示帮助信息"),
        ("/clear", "清空对话历史"),
        ("/history", "显示消息数量"),
        ("/stats", "显示会话统计"),
        ("/log", "显示日志目录"),
        ("/log <file>", "读取指定日志文件"),
        ("/mode", "切换模式: plan, agent, yolo"),
        ("/save [label]", "保存当前会话"),
        ("/load <id>", "加载已保存的会话"),
        ("/list", "列出已保存的会话"),
        ("/subagent <task>", "运行后台任务"),
        ("/skills", "列出所有可用技能"),
        ("/brainstorm", "了解头脑风暴工作流"),
        ("/plan", "了解计划工作流"),
        ("/task [start|end|cancel]", "管理任务状态"),
        ("/status", "显示当前任务状态"),
        ("/exit", "退出程序"),
    ]:
        print(f"  {Colors.GREEN}{cmd:<20}{Colors.RESET} {Colors.DIM}{desc}{Colors.RESET}")

    print(f"\n  {Colors.BOLD}{Colors.BRIGHT_WHITE}快捷键{Colors.RESET}")
    print(f"  {Colors.DIM}╌{'╌' * 40}╌{Colors.RESET}")
    for key, desc in [
        ("Tab", "循环切换模式"),
        ("Esc", "取消执行"),
        ("Ctrl+C", "退出"),
        ("↑↓", "命令历史"),
    ]:
        print(f"  {Colors.GREEN}{key:<20}{Colors.RESET} {Colors.DIM}{desc}{Colors.RESET}")
    print()


def print_session_info(agent: Agent, workspace_dir: Path, model: str) -> None:
    """以简洁的行格式打印会话信息。"""
    pm = "Windows" if platform.system() == "Windows" else "Unix"
    print(
        f"  {Colors.GREEN}{model}{Colors.RESET}  {Colors.DIM}·{Colors.RESET}"
        f"  {workspace_dir.absolute()}  {Colors.DIM}·{Colors.RESET}"
        f"  mode: {Colors.BOLD}{agent.mode.value.upper()}{Colors.RESET}"
        f"  {Colors.DIM}·{Colors.RESET}  {pm}"
    )
    print()


def print_stats(agent: Agent, session_start: datetime) -> None:
    """打印会话统计信息。"""
    duration = datetime.now() - session_start
    minutes = int(duration.total_seconds() // 60)
    seconds = int(duration.total_seconds() % 60)
    print(f"\n  {Colors.BRIGHT_WHITE}会话统计{Colors.RESET}")
    print(f"  {Colors.DIM}╌{'╌' * 36}╌{Colors.RESET}")
    for icon, label, value in [
        ("⏱", "运行时长", f"{minutes}m {seconds}s"),
        ("💬", "消息数", str(len(agent.messages))),
        ("🔢", "API 调用", f"{agent.api_call_count}"),
    ]:
        print(f"  {icon}  {Colors.DIM}{label}:{Colors.RESET} {value}")
    print()
