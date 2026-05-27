"""Shared utilities for batch tools.

Extracts duplicated logic from WorkspaceContextTool and DeepContextTool
to eliminate code duplication and ensure consistent behavior.
"""

import os
import subprocess  # nosec B404
from collections.abc import Sequence
from pathlib import Path

# Directories to skip when walking the file tree
DEFAULT_SKIP_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".git",
        "venv",
        ".venv",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "target",
        "vendor",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def get_git_status_sync(
    workspace_dir: str | Path,
    *,
    max_status_lines: int = 30,
    max_commits: int = 5,
) -> str:
    """Get git status information synchronously.

    Args:
        workspace_dir: Path to the git repository
        max_status_lines: Maximum number of status lines to show
        max_commits: Number of recent commits to show

    Returns:
        Formatted git status string
    """
    lines: list[str] = []
    try:
        # Check if in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(workspace_dir),
            timeout=5,
        )
        if result.returncode != 0:
            return "Not a git repository"

        # Get branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(workspace_dir),
            timeout=5,
        )
        branch = result.stdout.strip()
        lines.append(f"Branch: {branch}")

        # Get short status
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(workspace_dir),
            timeout=5,
        )
        status = result.stdout.strip()
        if status:
            status_lines = status.split("\n")
            lines.append("Changes:")
            for line in status_lines[:max_status_lines]:
                lines.append(f"  {line}")
            if len(status_lines) > max_status_lines:
                lines.append(f"  ... ({len(status_lines) - max_status_lines} more changes)")
        else:
            lines.append("Working tree clean")

        # Get recent commits
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{max_commits}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(workspace_dir),
            timeout=5,
        )
        if result.returncode == 0:
            lines.append("\nRecent commits:")
            for line in result.stdout.strip().split("\n"):
                lines.append(f"  {line}")

    except Exception as e:
        lines.append(f"Git info unavailable: {e}")

    return "\n".join(lines)


def get_tree_sync(
    workspace_dir: str | Path,
    max_depth: int,
    *,
    show_sizes: bool = False,
    max_files_per_dir: int = 20,
    skip_dirs: Sequence[str] = (),
) -> str:
    """Get directory tree structure synchronously.

    Args:
        workspace_dir: Root directory to walk
        max_depth: Maximum directory depth to traverse
        show_sizes: Whether to show file sizes
        max_files_per_dir: Maximum files to list per directory (0 = unlimited)
        skip_dirs: Additional directory names to skip

    Returns:
        Formatted tree string
    """
    workspace_dir = Path(workspace_dir)
    all_skip = DEFAULT_SKIP_DIRS | set(skip_dirs)
    lines: list[str] = []

    try:
        for root, dirs, files in os.walk(workspace_dir):
            # Skip hidden and common non-essential directories
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in all_skip)

            rel_root = Path(root).relative_to(workspace_dir)
            depth = len(rel_root.parts) if str(rel_root) != "." else 0

            if depth > max_depth:
                dirs.clear()
                continue

            indent = "  " * depth
            folder_name = str(rel_root) if str(rel_root) != "." else "."

            if show_sizes:
                visible_count = len([f for f in files if not f.startswith(".")])
                lines.append(f"{indent}{folder_name}/ ({visible_count} files)")
            else:
                lines.append(f"{indent}{folder_name}/")

            # List files
            file_count = 0
            for f in sorted(files):
                if f.startswith("."):
                    continue
                file_count += 1

                if max_files_per_dir > 0 and file_count > max_files_per_dir:
                    continue

                if show_sizes:
                    try:
                        size = (Path(root) / f).stat().st_size
                        size_str = f"{size}B" if size < 1024 else f"{size / 1024:.1f}K"
                        lines.append(f"{indent}  {f} ({size_str})")
                    except OSError:
                        lines.append(f"{indent}  {f}")
                else:
                    lines.append(f"{indent}  {f}")

            if max_files_per_dir > 0 and file_count > max_files_per_dir:
                lines.append(f"{indent}  ... ({file_count - max_files_per_dir} more files)")

    except Exception as e:
        lines.append(f"Error generating tree: {e}")

    return "\n".join(lines)
