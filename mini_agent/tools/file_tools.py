"""File operation tools."""

import os
from pathlib import Path
from typing import Any

from ..utils.model_utils import is_minimax_model
from ..utils.platform_utils import normalize_path_separators as normalize_path
from ..utils.token_utils import get_encoder
from .base import Tool, ToolResult

# Centralized token limits for file content truncation
# Higher limits preserve more context and reduce redundant calls
DEFAULT_FILE_TOKEN_LIMIT = 64000
M27_FILE_TOKEN_LIMIT = 128000

_WINDOWS_BLACKLISTED_DIRS: set[str] = {
    str(Path("C:/Windows").resolve()),
    str(Path("C:/Program Files").resolve()),
    str(Path("C:/Program Files (x86)").resolve()),
    str(Path("C:/ProgramData").resolve()),
}

_UNIX_BLACKLISTED_DIRS: set[str] = {
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/root",
    "/var",
}

_HOME_BLACKLISTED_SUBDIRS: set[str] = {
    ".ssh",
    ".gnupg",
    ".config/ssh",
}


_EXTRA_BLOCKED_DIRS: set[str] = set()
_EXTRA_BLOCKED_HOME_SUBDIRS: set[str] = set()


def configure_path_blacklist(
    extra_blocked_dirs: list[str] | None = None,
    extra_blocked_home_subdirs: list[str] | None = None,
) -> None:
    """Configure additional path blacklist entries from config."""
    global _EXTRA_BLOCKED_DIRS, _EXTRA_BLOCKED_HOME_SUBDIRS
    if extra_blocked_dirs:
        _EXTRA_BLOCKED_DIRS = {str(Path(d).resolve()) for d in extra_blocked_dirs}
    if extra_blocked_home_subdirs:
        _EXTRA_BLOCKED_HOME_SUBDIRS = set(extra_blocked_home_subdirs)


def _is_path_blacklisted(resolved_path: Path) -> tuple[bool, str]:
    resolved_str = str(resolved_path)
    home_dir = Path.home().resolve()
    if os.name == "nt":
        resolved_lower = resolved_str.lower()
        for bl_dir in _WINDOWS_BLACKLISTED_DIRS:
            if resolved_lower.startswith(bl_dir.lower()):
                return True, f"Access denied: {resolved_str} is under blacklisted system directory {bl_dir}"
        for bl_dir in _EXTRA_BLOCKED_DIRS:
            if resolved_lower.startswith(bl_dir.lower()):
                return True, f"Access denied: {resolved_str} is under blacklisted directory {bl_dir}"
    else:
        for bl_dir in _UNIX_BLACKLISTED_DIRS:
            if resolved_str == bl_dir or resolved_str.startswith(bl_dir + "/"):
                return True, f"Access denied: {resolved_str} is under blacklisted system directory {bl_dir}"
        for bl_dir in _EXTRA_BLOCKED_DIRS:
            if resolved_str == bl_dir or resolved_str.startswith(bl_dir + "/"):
                return True, f"Access denied: {resolved_str} is under blacklisted directory {bl_dir}"
    home_str = str(home_dir)
    if os.name == "nt":
        if resolved_lower.startswith(home_str.lower()):
            rel = resolved_str[len(home_str) :].lstrip(os.sep).lstrip("/")
            for sub in _HOME_BLACKLISTED_SUBDIRS | _EXTRA_BLOCKED_HOME_SUBDIRS:
                sub_norm = sub.replace("/", os.sep)
                if rel.lower().startswith(sub_norm.lower()):
                    return True, f"Access denied: {resolved_str} is under blacklisted home subdirectory ~{os.sep}{sub}"
    else:
        if resolved_str == home_str or resolved_str.startswith(home_str + "/"):
            rel = resolved_str[len(home_str) :].lstrip("/")
            for sub in _HOME_BLACKLISTED_SUBDIRS | _EXTRA_BLOCKED_HOME_SUBDIRS:
                if rel == sub or rel.startswith(sub + "/"):
                    return True, f"Access denied: {resolved_str} is under blacklisted home subdirectory ~/{sub}"
    return False, ""


def _resolve_and_validate_path(path: str, workspace_dir: Path) -> Path:
    """Resolve a path and validate relative paths stay within the workspace.

    For relative paths: resolves against workspace_dir and ensures the result
    does not escape via directory traversal (e.g. ../../etc/passwd).
    For absolute paths: allows through (agent may need to read/write files
    outside the workspace, e.g. system configs), but still normalizes the path
    to eliminate traversal components.

    Args:
        path: Raw file path (absolute or relative)
        workspace_dir: The workspace root directory

    Returns:
        Resolved absolute Path

    Raises:
        ValueError: If a relative path resolves outside the workspace
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        # Relative path: resolve against workspace and validate containment
        resolved = (workspace_dir / file_path).resolve()
        workspace_resolved = workspace_dir.resolve()
        try:
            resolved.relative_to(workspace_resolved)
        except ValueError:
            raise ValueError(f"Path escapes workspace: {resolved} is outside {workspace_resolved}") from None
        return resolved
    else:
        resolved = file_path.resolve()
        is_blocked, reason = _is_path_blacklisted(resolved)
        if is_blocked:
            raise ValueError(reason)
        return resolved


def truncate_text_by_tokens(
    text: str,
    max_tokens: int,
    model_name: str | None = None,
) -> str:
    """Truncate text by token count if it exceeds the limit.

    When text exceeds the specified token limit, performs intelligent truncation
    by keeping the front and back parts while truncating the middle.

    Args:
        text: Text to be truncated
        max_tokens: Maximum token limit
        model_name: Optional model name for model-specific limits

    Returns:
        str: Truncated text if it exceeds the limit, otherwise the original text.
    """
    # Get model-specific limit if model_name provided
    if model_name and is_minimax_model(model_name):
        max_tokens = M27_FILE_TOKEN_LIMIT

    encoder = get_encoder("cl100k_base")
    token_count = len(encoder.encode(text))

    # Return original text if under limit
    if token_count <= max_tokens:
        return text

    # Calculate token/character ratio for approximation
    char_count = len(text)
    if char_count == 0:
        return ""
    ratio = token_count / char_count

    # Keep head and tail mode: allocate half space for each (with 5% safety margin)
    chars_per_half = int((max_tokens / 2) / ratio * 0.95)

    # Truncate front part: find nearest newline
    head_part = text[:chars_per_half]
    last_newline_head = head_part.rfind("\n")
    if last_newline_head > 0:
        head_part = head_part[:last_newline_head]

    # Truncate back part: find nearest newline
    tail_part = text[-chars_per_half:]
    first_newline_tail = tail_part.find("\n")
    if first_newline_tail > 0:
        tail_part = tail_part[first_newline_tail + 1 :]

    # Combine result
    truncation_note = f"\n\n... [Content truncated: {token_count} tokens -> ~{max_tokens} tokens limit] ...\n\n"
    return head_part + truncation_note + tail_part


def get_file_token_limit(model_name: str | None = None) -> int:
    """Get the appropriate token limit for file content.

    Args:
        model_name: Optional model name for model-specific limits

    Returns:
        Token limit for file content
    """
    if model_name and is_minimax_model(model_name):
        return M27_FILE_TOKEN_LIMIT
    return DEFAULT_FILE_TOKEN_LIMIT


class ReadTool(Tool):
    """Read file content."""

    def __init__(self, workspace_dir: str = "."):
        """Initialize ReadTool with workspace directory.

        Args:
            workspace_dir: Base directory for resolving relative paths
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read file contents from the filesystem. Output always includes line numbers "
            "in format 'LINE_NUMBER|LINE_CONTENT' (1-indexed). Supports reading partial content "
            "by specifying line offset and limit for large files. "
            "You can call this tool multiple times in parallel to read different files simultaneously."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed). Use for large files to read from specific line",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read. Use with offset for large files to read in chunks",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
        """Execute read file."""
        try:
            # Normalize path separators for current platform
            path = normalize_path(path)
            # Resolve and validate path stays within workspace
            try:
                file_path = _resolve_and_validate_path(path, self.workspace_dir)
            except ValueError as e:
                return ToolResult(success=False, content="", error=str(e))

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"File not found: {path}",
                )

            # Read file content with line numbers
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Apply offset and limit
            start = (offset - 1) if offset else 0
            end = (start + limit) if limit else len(lines)
            if start < 0:
                start = 0
            if end > len(lines):
                end = len(lines)

            selected_lines = lines[start:end]

            # Format with line numbers (1-indexed)
            numbered_lines = []
            for i, line in enumerate(selected_lines, start=start + 1):
                # Remove trailing newline for formatting
                line_content = line.rstrip("\n")
                numbered_lines.append(f"{i:6d}|{line_content}")

            content = "\n".join(numbered_lines)

            # Apply token truncation with centralized limit
            max_tokens = get_file_token_limit()
            content = truncate_text_by_tokens(content, max_tokens)

            return ToolResult(success=True, content=content)
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class WriteTool(Tool):
    """Write content to a file."""

    def __init__(self, workspace_dir: str = "."):
        """Initialize WriteTool with workspace directory.

        Args:
            workspace_dir: Base directory for resolving relative paths
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Will overwrite existing files completely. "
            "For existing files, you should read the file first using read_file. "
            "Prefer editing existing files over creating new ones unless explicitly needed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Complete content to write (will replace existing content)",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        """Execute write file."""
        try:
            # Normalize path separators for current platform
            path = normalize_path(path)
            # Resolve and validate path stays within workspace
            try:
                file_path = _resolve_and_validate_path(path, self.workspace_dir)
            except ValueError as e:
                return ToolResult(success=False, content="", error=str(e))

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding="utf-8")
            return ToolResult(success=True, content=f"Successfully wrote to {file_path}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class EditTool(Tool):
    """Edit file by replacing text."""

    def __init__(self, workspace_dir: str = "."):
        """Initialize EditTool with workspace directory.

        Args:
            workspace_dir: Base directory for resolving relative paths
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Perform exact string replacement in a file. The old_str must match exactly "
            "and appear uniquely in the file, otherwise the operation will fail. "
            "You must read the file first before editing. Preserve exact indentation from the source."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find and replace (must be unique in file)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string (use for refactoring, renaming, etc.)",
                },
            },
            "required": ["path", "old_str", "new_str"],
        }

    async def execute(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """Execute edit file."""
        try:
            # Normalize path separators for current platform
            path = normalize_path(path)
            # Resolve and validate path stays within workspace
            try:
                file_path = _resolve_and_validate_path(path, self.workspace_dir)
            except ValueError as e:
                return ToolResult(success=False, content="", error=str(e))

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"File not found: {path}",
                )

            content = file_path.read_text(encoding="utf-8")

            if old_str not in content:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Text not found in file: {old_str}",
                )

            count = content.count(old_str)
            if count > 1:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Text appears {count} times in file. old_str must be unique.",
                )
            new_content = content.replace(old_str, new_str)
            file_path.write_text(new_content, encoding="utf-8")

            return ToolResult(success=True, content=f"Successfully edited {file_path}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
