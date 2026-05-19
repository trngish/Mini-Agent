"""File search and content search tools.

Provides grep-like functionality for searching file contents and
finding files by patterns.
"""

import fnmatch
import re
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from ..utils.platform_utils import normalize_path_separators


class GrepTool(Tool):
    """Search for patterns in file contents."""

    def __init__(self, workspace_dir: str = "."):
        """Initialize GrepTool with workspace directory.

        Args:
            workspace_dir: Base directory for resolving relative paths
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return """Search for text patterns in files (similar to grep).

Supports:
- Simple text search
- Regular expression patterns
- Case-sensitive and case-insensitive search
- Line number display for found matches

Parameters:
  - pattern: Text or regex pattern to search for
  - path: Directory or file path to search in (default: workspace root)
  - file_pattern: Glob pattern for file names to search (e.g., "*.py", "*.md")
  - case_sensitive: Whether search should be case-sensitive (default: false)
  - regex: Whether pattern is a regular expression (default: false)
  - max_results: Maximum number of results to return (default: 100)

Examples:
  - Search for "TODO" in all Python files
  - Search for "function" in src/ directory
  - Search for "^class " pattern in .py files using regex
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in (default: workspace root)",
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names to search (e.g., '*.py', '*.md')",
                    "default": "*",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search should be case-sensitive (default: false)",
                    "default": False,
                },
                "regex": {
                    "type": "boolean",
                    "description": "Whether pattern is a regular expression (default: false)",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 100)",
                    "default": 100,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 100,
    ) -> ToolResult:
        """Execute grep search."""
        try:
            # Normalize path
            search_path = normalize_path_separators(path)
            search_dir = Path(search_path)
            if not search_dir.is_absolute():
                search_dir = self.workspace_dir / search_dir

            if not search_dir.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Path not found: {path}",
                )

            # Compile regex pattern if needed
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    compiled_pattern = re.compile(pattern, flags)
                except re.error as e:
                    return ToolResult(
                        success=False,
                        content="",
                        error=f"Invalid regex pattern: {e}",
                    )
            else:
                compiled_pattern = None
                search_text = pattern if case_sensitive else pattern.lower()

            # Search files
            results = []
            files_searched = 0

            for file_path in self._iterate_files(search_dir, file_pattern):
                files_searched += 1
                if len(results) >= max_results:
                    break

                try:
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, start=1):
                            line_to_check = line if case_sensitive else line.lower()
                            matched = False

                            if compiled_pattern:
                                if compiled_pattern.search(line):
                                    matched = True
                            else:
                                if search_text in line_to_check:
                                    matched = True

                            if matched:
                                results.append({
                                    "file": str(file_path.relative_to(self.workspace_dir)),
                                    "line": line_num,
                                    "content": line.rstrip("\n"),
                                })

                                if len(results) >= max_results:
                                    break

                except Exception:
                    # Skip files that can't be read
                    continue

            # Format results
            if not results:
                return ToolResult(
                    success=True,
                    content=f"No matches found for '{pattern}' in {files_searched} files",
                )

            output = f"Found {len(results)} matches in {files_searched} files:\n\n"
            current_file = None
            for result in results:
                if result["file"] != current_file:
                    current_file = result["file"]
                    output += f"{'=' * 60}\n"
                    output += f"File: {current_file}\n"
                    output += f"{'=' * 60}\n"

                output += f"  {result['line']:6d}| {result['content']}\n"

            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    def _iterate_files(self, directory: Path, pattern: str):
        """Iterate over files matching the pattern in directory."""
        import os
        if directory.is_file():
            yield directory
            return

        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]

            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    yield Path(root) / filename


class FindTool(Tool):
    """Find files by name pattern."""

    def __init__(self, workspace_dir: str = "."):
        """Initialize FindTool with workspace directory.

        Args:
            workspace_dir: Base directory for resolving relative paths
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "find"

    @property
    def description(self) -> str:
        return """Find files by name pattern (similar to find command).

Supports glob patterns and case-sensitive/insensitive search.

Parameters:
  - pattern: Glob pattern for file names (e.g., "*.py", "test_*.txt")
  - path: Directory to search in (default: workspace root)
  - case_sensitive: Whether pattern matching is case-sensitive (default: true)
  - max_results: Maximum number of results (default: 100)

Examples:
  - Find all Python files: pattern="*.py"
  - Find all test files: pattern="test_*.py"
  - Find configuration files: pattern="*.json"
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (e.g., '*.py', 'test_*.txt')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: workspace root)",
                    "default": ".",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether pattern matching is case-sensitive (default: true)",
                    "default": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 100)",
                    "default": 100,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        case_sensitive: bool = True,
        max_results: int = 100,
    ) -> ToolResult:
        """Execute find search."""
        try:
            # Normalize path
            search_path = normalize_path_separators(path)
            search_dir = Path(search_path)
            if not search_dir.is_absolute():
                search_dir = self.workspace_dir / search_dir

            if not search_dir.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Path not found: {path}",
                )

            # Find files
            results = []
            search_pattern = pattern if case_sensitive else pattern.lower()

            import os
            for root, dirs, files in os.walk(search_dir):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                for filename in files:
                    match = False
                    if case_sensitive:
                        match = fnmatch.fnmatch(filename, search_pattern)
                    else:
                        match = fnmatch.fnmatch(filename.lower(), search_pattern)

                    if match:
                        full_path = Path(root) / filename
                        rel_path = full_path.relative_to(self.workspace_dir)
                        results.append(str(rel_path))

                        if len(results) >= max_results:
                            break

                if len(results) >= max_results:
                    break

            # Format results
            if not results:
                return ToolResult(
                    success=True,
                    content=f"No files found matching pattern: {pattern}",
                )

            output = f"Found {len(results)} files matching '{pattern}':\n\n"
            for file_path in sorted(results):
                output += f"  - {file_path}\n"

            if len(results) >= max_results:
                output += f"\n  ... (showing first {max_results} results)"

            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class TreeTool(Tool):
    """Display directory tree structure."""

    def __init__(self, workspace_dir: str = "."):
        """Initialize TreeTool with workspace directory.

        Args:
            workspace_dir: Base directory for resolving relative paths
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "tree"

    @property
    def description(self) -> str:
        return """Display directory tree structure.

Shows the file and directory hierarchy starting from a given path.

Parameters:
  - path: Directory path to display (default: workspace root)
  - max_depth: Maximum depth to display (default: 3)
  - include_hidden: Whether to include hidden files/directories (default: false)

Examples:
  - Show entire workspace tree
  - Show project structure with max_depth=2
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to display (default: workspace root)",
                    "default": ".",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth to display (default: 3)",
                    "default": 3,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files/directories (default: false)",
                    "default": False,
                },
            },
        }

    async def execute(
        self,
        path: str = ".",
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> ToolResult:
        """Execute tree display."""
        try:
            # Normalize path
            tree_path = normalize_path_separators(path)
            root_dir = Path(tree_path)
            if not root_dir.is_absolute():
                root_dir = self.workspace_dir / root_dir

            if not root_dir.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Path not found: {path}",
                )

            if not root_dir.is_dir():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Not a directory: {path}",
                )

            # Build tree
            lines = []
            self._build_tree(root_dir, "", lines, max_depth, 0, include_hidden)

            output = f"Directory tree: {root_dir.name}\n"
            output += "=" * 60 + "\n"
            output += "\n".join(lines)

            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    def _build_tree(
        self,
        directory: Path,
        prefix: str,
        lines: list,
        max_depth: int,
        current_depth: int,
        include_hidden: bool,
    ):
        """Recursively build tree lines."""
        try:
            entries = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            for i, entry in enumerate(entries):
                # Skip hidden files if not including them
                if not include_hidden and entry.name.startswith('.'):
                    continue

                is_last = (i == len(entries) - 1)
                connector = "└── " if is_last else "├── "

                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    if current_depth < max_depth - 1:
                        extension = "    " if is_last else "│   "
                        self._build_tree(
                            entry,
                            prefix + extension,
                            lines,
                            max_depth,
                            current_depth + 1,
                            include_hidden,
                        )
                else:
                    size = entry.stat().st_size
                    size_str = self._format_size(size)
                    lines.append(f"{prefix}{connector}{entry.name} ({size_str})")

        except PermissionError:
            lines.append(f"{prefix}[Permission Denied]")

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f}K"
        else:
            return f"{size / (1024 * 1024):.1f}M"