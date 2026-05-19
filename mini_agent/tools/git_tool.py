"""Git helper tool for common Git operations.

This tool provides convenient wrappers for common Git operations
to help the agent work with version control more easily.
"""

import asyncio
import shlex
import platform as _platform
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from .bash_tool import BashTool


def _shell_quote(s: str, is_windows: bool) -> str:
    """Quote a string for safe shell insertion, preventing injection.

    Args:
        s: The string to quote
        is_windows: Whether the target shell is Windows

    Returns:
        Shell-safe quoted string
    """
    if is_windows:
        # PowerShell: escape single quotes and wrap in single quotes
        # Single quotes in PowerShell are literal; escape ' by doubling it
        return "'" + s.replace("'", "''") + "'"
    else:
        # Unix: use shlex.quote for POSIX shell safety
        return shlex.quote(s)


class GitTool(Tool):
    """Git helper tool for common operations."""

    def __init__(self, workspace_dir: str = "."):
        """Initialize GitTool with workspace directory.

        Args:
            workspace_dir: Base directory for Git operations
        """
        self.workspace_dir = Path(workspace_dir).absolute()
        self._bash = BashTool(workspace_dir=str(workspace_dir))

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return """Git helper tool for common version control operations.

Supports:
- git_status: Show current repository status
- git_add: Stage files for commit
- git_commit: Commit staged changes with message
- git_log: Show recent commit history
- git_diff: Show uncommitted changes
- git_branch: List all branches
- git_checkout: Switch branches
- git_pull: Pull remote changes
- git_push: Push commits to remote

Parameters:
  - operation: The Git operation to perform (status, add, commit, log, diff, branch, checkout, pull, push)
  - path: File path or directory for the operation (default: workspace root)
  - message: Commit message (required for commit operation)
  - branch: Branch name (required for checkout operation)

Examples:
  - Get status: operation="status"
  - Stage a file: operation="add", path="src/app.py"
  - Commit changes: operation="commit", message="Fix bug in login"
  - View history: operation="log", path="."
  - Switch branch: operation="checkout", branch="develop"
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Git operation to perform (status, add, commit, log, diff, branch, checkout, pull, push)",
                    "enum": ["status", "add", "commit", "log", "diff", "branch", "checkout", "pull", "push"],
                },
                "path": {
                    "type": "string",
                    "description": "File path or directory for the operation (default: workspace root)",
                    "default": ".",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (required for commit operation)",
                },
                "branch": {
                    "type": "string",
                    "description": "Branch name (required for checkout operation)",
                },
                "all": {
                    "type": "boolean",
                    "description": "Stage all modified files (for add operation)",
                    "default": False,
                },
            },
            "required": ["operation"],
        }

    async def execute(
        self,
        operation: str,
        path: str = ".",
        message: str | None = None,
        branch: str | None = None,
        all: bool = False,
    ) -> ToolResult:
        """Execute Git operation.

        Args:
            operation: The Git operation to perform
            path: File path or directory
            message: Commit message
            branch: Branch name for checkout
            all: Stage all files for add

        Returns:
            ToolResult with operation output
        """
        # Resolve path relative to workspace
        if not Path(path).is_absolute():
            target_path = self.workspace_dir / path
        else:
            target_path = Path(path)

        is_windows = _platform.system() == "Windows"
        separator = "; " if is_windows else " && "
        qd = lambda s: _shell_quote(str(s), is_windows)  # noqa: E731

        # Build git command based on operation
        if operation == "status":
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git status --porcelain", timeout=30)
            if result.success:
                output = result.content.strip()
                if not output:
                    return ToolResult(success=True, content="Working tree clean")
                lines = output.split("\n")
                staged = [l for l in lines if l.startswith("M") or l.startswith("A")]
                unstaged = [l for l in lines if l.startswith(" M") or l.startswith("??")]
                content = "Git Status:\n"
                if staged:
                    content += "\nStaged:\n  " + "\n  ".join(staged)
                if unstaged:
                    content += "\nUnstaged:\n  " + "\n  ".join(unstaged)
                return ToolResult(success=True, content=content)
            return result

        elif operation == "add":
            if all:
                result = await self._bash.execute(f"cd {qd(target_path)}{separator}git add -A", timeout=30)
            else:
                result = await self._bash.execute(f"cd {qd(target_path)}{separator}git add {qd(path)}", timeout=30)
            if result.success:
                return ToolResult(success=True, content=f"Staged: {path}")
            return result

        elif operation == "commit":
            if not message:
                return ToolResult(success=False, content="", error="Commit message required")
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git commit -m {qd(message)}", timeout=60)
            if result.success:
                return ToolResult(success=True, content=f"Committed: {message}")
            return result

        elif operation == "log":
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git log --oneline -10", timeout=30)
            if result.success:
                output = result.content.strip()
                if not output:
                    return ToolResult(success=True, content="No commits yet")
                return ToolResult(success=True, content=f"Recent Commits:\n\n{output}")
            return result

        elif operation == "diff":
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git diff --stat", timeout=30)
            diff_result = await self._bash.execute(f"cd {qd(target_path)}{separator}git diff", timeout=60)
            if result.success:
                output = result.content.strip()
                if not output:
                    return ToolResult(success=True, content="No uncommitted changes")
                diff_content = diff_result.content if diff_result.success else ""
                return ToolResult(
                    success=True,
                    content=f"Uncommitted Changes:\n\n{output}\n\n{diff_content}"
                )
            return result

        elif operation == "branch":
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git branch -a", timeout=30)
            if result.success:
                output = result.content.strip()
                return ToolResult(success=True, content=f"Branches:\n\n{output}")
            return result

        elif operation == "checkout":
            if not branch:
                return ToolResult(success=False, content="", error="Branch name required for checkout")
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git checkout {qd(branch)}", timeout=60)
            if result.success:
                return ToolResult(success=True, content=f"Switched to branch: {branch}")
            return result

        elif operation == "pull":
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git pull", timeout=120)
            return result

        elif operation == "push":
            result = await self._bash.execute(f"cd {qd(target_path)}{separator}git push", timeout=120)
            return result

        else:
            return ToolResult(
                success=False,
                content="",
                error=f"Unknown operation: {operation}. Supported: status, add, commit, log, diff, branch, checkout, pull, push"
            )


class GitStatusTool(Tool):
    """Quick git status check."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()
        self._bash = BashTool(workspace_dir=str(workspace_dir))

    @property
    def name(self) -> str:
        return "git_status"

    @property
    def description(self) -> str:
        return """Quick check of Git repository status.

Shows the current branch, staged/unstaged changes, and any untracked files.

Examples:
  - git_status() - Check status in workspace
  - git_status(path="src") - Check status in src directory
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the Git repository (default: workspace root)",
                    "default": ".",
                },
            },
        }

    async def execute(self, path: str = ".") -> ToolResult:
        """Get git status."""
        if not Path(path).is_absolute():
            target_path = self.workspace_dir / path
        else:
            target_path = Path(path)

        is_windows = _platform.system() == "Windows"
        separator = "; " if is_windows else " && "
        qd = lambda s: _shell_quote(str(s), is_windows)  # noqa: E731

        # Get branch first
        branch_result = await self._bash.execute(f"cd {qd(target_path)}{separator}git branch --show-current", timeout=10)
        branch = branch_result.content.strip() if branch_result.success else "unknown"

        # Get status
        status_result = await self._bash.execute(f"cd {qd(target_path)}{separator}git status --short", timeout=10)
        
        if status_result.success:
            lines = status_result.content.strip().split("\n") if status_result.content.strip() else []
            staged = [l for l in lines if l.startswith("M") or l.startswith("A") or l.startswith("D")]
            unstaged = [l for l in lines if l.startswith(" M") or l.startswith("??") or l.startswith(" D")]
            
            content = f"Branch: {branch}\n"
            if staged:
                content += f"\nStaged ({len(staged)}):\n  " + "\n  ".join(staged[:20])
            if unstaged:
                content += f"\nUnstaged ({len(unstaged)}):\n  " + "\n  ".join(unstaged[:20])
            if not staged and not unstaged:
                content += "\nWorking tree clean"
            
            return ToolResult(success=True, content=content)
        
        return status_result