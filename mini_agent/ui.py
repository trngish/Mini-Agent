"""Terminal UI helpers for Mini Agent."""

import io
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .agent import Agent
from .utils import Colors
from .utils.terminal_utils import calculate_display_width

SEP = f"{Colors.DIM}┄{'─' * 56}┄{Colors.RESET}"

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


def print_image(width: int = 80) -> bool:
    """Display logo image in terminal using ANSI true color + half-blocks."""
    path = Path(__file__).parent / "config" / "minimax.webp"
    if not path.exists():
        return False
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        img = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        aspect = img.height / img.width
        height = int(width * aspect * 0.5)
        if height < 1:
            return False
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
        img = img.resize((width, height * 2), resample)
        pixels = img.load()
        for y in range(0, img.height, 2):
            parts = []
            for x in range(img.width):
                r1, g1, b1 = pixels[x, y]
                r2, g2, b2 = pixels[x, y + 1]
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
        print(f"\n{Colors.YELLOW}Log directory does not exist yet.{Colors.RESET}\n")
        return
    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    print(f"\n{Colors.BRIGHT_CYAN}📁 Log Directory: {log_dir}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}Total log files: {len(log_files)}{Colors.RESET}\n")
    if log_files:
        print(f"{Colors.BRIGHT_CYAN}Recent log files:{Colors.RESET}")
        for f in log_files[:10]:
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size = f.stat().st_size
            print(f"  {Colors.DIM}{mtime}{Colors.RESET}  {Colors.GREEN}{f.name}{Colors.RESET}  ({size} bytes)")
    if len(log_files) > 10:
        print(f"  {Colors.DIM}... and {len(log_files) - 10} more files{Colors.RESET}")
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
        print(f"{Colors.YELLOW}Could not open file manager.{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.YELLOW}Error opening file manager: {e}{Colors.RESET}")


def read_log_file(filename: str) -> None:
    log_dir = get_log_directory()
    log_file = log_dir / filename
    if not log_file.exists() or not log_file.is_file():
        print(f"\n{Colors.RED}❌ Log file not found: {log_file}{Colors.RESET}\n")
        return
    print(f"\n{Colors.BRIGHT_CYAN}📄 Reading: {log_file}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        print(content)
        print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
        print(f"\n{Colors.GREEN}✅ End of file{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error reading file: {e}{Colors.RESET}\n")


def print_banner() -> None:
    """Print welcome banner — Claude Code style with gradient M logo."""
    W = 78

    # Gradient M logo with ANSI color escape sequences
    # Each line has multiple color segments for gradient effect
    logo = [
        f"{Colors.CYAN}███╗   ███╗███╗   ███╗██╗  ██╗",
        f"{Colors.BRIGHT_CYAN}████╗ ████║████╗ ████║╚██╗██╔╝",
        f"{Colors.BRIGHT_CYAN}██╔████╔██║██╔████╔██║ ╚███╔╝",
        f"{Colors.CYAN}██║╚██╔╝██║██║╚██╔╝██║ ██╔██╗",
        f"{Colors.CYAN}██║ ╚═╝ ██║██║ ╚═╝ ██║██╔╝ ██╗",
        f"{Colors.CYAN}╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝",
    ]
    tips = [
        "Type a task to begin",
        "/help for all commands",
        "Tab to cycle modes",
        "Ctrl+C to exit",
    ]

    def pad(n: int) -> str:
        return f"{'':>{n}}"

    # Calculate actual display width of title (excluding ANSI codes)
    title = f"{Colors.BOLD}{Colors.BRIGHT_WHITE} Mini Agent {Colors.RESET}"
    tv_display_width = calculate_display_width(" Mini Agent ")
    ld = 4
    rd = W - ld - tv_display_width
    print(f"\n  {Colors.DIM}╭{'─' * ld}{Colors.RESET}{title}{Colors.DIM}{'─' * rd}╮{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET}  {pad(W-2)}{Colors.DIM}│{Colors.RESET}")

    rows = max(len(logo), len(tips) + 1)
    for i in range(rows):
        left = logo[i] if i < len(logo) else pad(60)
        if i == 0:
            right = f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Welcome!{Colors.RESET}"
        elif i - 1 < len(tips):
            right = f"{Colors.DIM}{tips[i-1]}{Colors.RESET}"
        else:
            right = ""
        
        # Calculate display widths instead of raw string lengths
        lv = calculate_display_width(left) if i < len(logo) else 60
        rv = calculate_display_width(right)
        
        # Calculate remaining space for padding
        # Layout: left content + 4 spaces + right content + padding + borders
        content_width = lv + 4 + rv
        rp = W - 2 - content_width  # -2 for left and right border characters
        
        print(f"  {Colors.DIM}│{Colors.RESET}  {left}    {right}{pad(max(0, rp))}{Colors.DIM}│{Colors.RESET}")

    print(f"  {Colors.DIM}│{Colors.RESET}  {pad(W-2)}{Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}╰{'─' * W}╯{Colors.RESET}")
    print()


def print_help() -> None:
    """Print help information."""
    print(f"\n  {Colors.BOLD}{Colors.BRIGHT_WHITE}Commands{Colors.RESET}")
    print(f"  {Colors.DIM}╌{'╌' * 40}╌{Colors.RESET}")
    for cmd, desc in [
        ("/help", "Show this help message"),
        ("/clear", "Clear conversation history"),
        ("/history", "Show message count"),
        ("/stats", "Show session statistics"),
        ("/log", "Show log directory"),
        ("/log <file>", "Read a specific log file"),
        ("/mode", "Switch mode: plan, agent, yolo"),
        ("/save [label]", "Save current session"),
        ("/load <id>", "Load a saved session"),
        ("/list", "List saved sessions"),
        ("/subagent <task>", "Run background task"),
        ("/skills", "List all available skills"),
        ("/brainstorm", "Learn about brainstorming workflow"),
        ("/plan", "Learn about planning workflow"),
        ("/exit", "Exit the program"),
    ]:
        print(f"  {Colors.GREEN}{cmd:<20}{Colors.RESET} {Colors.DIM}{desc}{Colors.RESET}")

    print(f"\n  {Colors.BOLD}{Colors.BRIGHT_WHITE}Keys{Colors.RESET}")
    print(f"  {Colors.DIM}╌{'╌' * 40}╌{Colors.RESET}")
    for key, desc in [
        ("Tab", "Cycle modes"),
        ("Esc", "Cancel execution"),
        ("Ctrl+C", "Exit"),
        ("↑↓", "Command history"),
    ]:
        print(f"  {Colors.GREEN}{key:<20}{Colors.RESET} {Colors.DIM}{desc}{Colors.RESET}")
    print()


def print_session_info(agent: Agent, workspace_dir: Path, model: str) -> None:
    """Print session information in a compact line."""
    pm = "Windows" if platform.system() == "Windows" else "Unix"
    print(f"  {Colors.GREEN}{model}{Colors.RESET}  {Colors.DIM}·{Colors.RESET}  {workspace_dir.absolute()}  {Colors.DIM}·{Colors.RESET}  mode: {Colors.BOLD}{agent.mode.value.upper()}{Colors.RESET}  {Colors.DIM}·{Colors.RESET}  {pm}")
    print()


def print_stats(agent: Agent, session_start: datetime) -> None:
    """Print session statistics."""
    duration = datetime.now() - session_start
    minutes = int(duration.total_seconds() // 60)
    seconds = int(duration.total_seconds() % 60)
    print(f"\n  {Colors.BRIGHT_WHITE}Session Stats{Colors.RESET}")
    print(f"  {Colors.DIM}╌{'╌' * 36}╌{Colors.RESET}")
    for icon, label, value in [
        ("⏱", "Duration", f"{minutes}m {seconds}s"),
        ("💬", "Messages", str(len(agent.messages))),
        ("🔢", "调用次数", f"{agent.api_call_count}"),
    ]:
        print(f"  {icon}  {Colors.DIM}{label}:{Colors.RESET} {value}")
    print()
