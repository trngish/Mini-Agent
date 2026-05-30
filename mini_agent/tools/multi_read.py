"""Multi-read tool for reading multiple files in one call."""

import json
from pathlib import Path
from typing import Any

from ..utils.context_cache import create_cache_for_workspace
from ..utils.platform_utils import normalize_path_separators as normalize_path
from ..utils.task_state import get_task_manager
from .base import Tool, ToolResult
from .file_tools import get_file_token_limit, truncate_text_by_tokens


def _ensure_list(data: list[Any] | str | None) -> list[Any]:
    """Ensure input is a list, parsing JSON string if needed."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return [data]


class MultiReadTool(Tool):
    """Read multiple files in a single tool call.

    Per-call billing optimization: merge multiple read_file calls into one, reducing API call count.

    Caching features:
    - Results are cached for reuse across steps
    - Tracks which files have been read
    - Warns when re-reading files unnecessarily
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "multi_read"

    @property
    def description(self) -> str:
        return (
            "Read multiple files at once. Returns all file contents separated by headers. "
            "Use this instead of multiple read_file calls to save API calls. "
            "Results are cached for reuse."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read (absolute or relative).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed) applied to all files.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines per file.",
                },
            },
            "required": ["paths"],
        }

    async def execute(self, paths: list[str], offset: int | None = None, limit: int | None = None) -> ToolResult:
        """Execute multi-file read with caching support."""
        paths = _ensure_list(paths)
        results = []
        total_lines = 0
        errors = []
        newly_cached = 0

        # P0 FIX: Use workspace-namespaced cache, matching Agent's cache
        cache = create_cache_for_workspace(self.workspace_dir)
        task_manager = get_task_manager()
        files_read_this_call = []

        for path in paths:
            try:
                normalized = normalize_path(path)
                file_path = Path(normalized)
                if not file_path.is_absolute():
                    file_path = self.workspace_dir / file_path

                if not file_path.exists():
                    errors.append(f"File not found: {path}")
                    continue

                # Check cache first (only for full-file reads)
                if offset is None and limit is None:
                    cached_content = cache.get_file_content(file_path)
                    if cached_content:
                        results.append(f"{'=' * 60}\n📄 {normalized}\n{'=' * 60}\n{cached_content}")
                        results.append(f"  {'':>4}(cached {len(cached_content)} chars)")
                        total_lines += cached_content.count("\n")
                        newly_cached += 1
                        continue

                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()

                start = (offset - 1) if offset else 0
                end = (start + limit) if limit else len(lines)
                start = max(0, start)
                end = min(end, len(lines))

                selected_lines = lines[start:end]
                numbered = [f"{i:6d}|{line.rstrip(chr(10))}" for i, line in enumerate(selected_lines, start=start + 1)]
                content = "\n".join(numbered)
                total_lines += len(selected_lines)

                if limit is not None and len(lines) > limit:
                    content += f"\n... ({len(lines)} total lines)"

                results.append(f"{'=' * 60}\n📄 {normalized}\n{'=' * 60}\n{content}")

                # Cache full-file reads for future use
                if offset is None and limit is None:
                    cache.set_file_content(file_path, content, ttl=600)

                # Mark file as read for task state tracking
                files_read_this_call.append(str(file_path))

            except UnicodeDecodeError:
                errors.append(f"Decode error: {path}")
            except Exception as e:
                errors.append(f"Error reading {path}: {str(e)}")

        combined = "\n".join(results)
        if errors:
            combined += "\n\n" + "\n".join(errors)

        # Add cache statistics to output
        cache_stats = cache.get_stats()
        combined += f"\n\n📦 Cache: {cache_stats['file_entries']} entries, +{newly_cached} cached this call"

        # Notify task manager about files read (after loop to avoid partial state on error)
        for file_path in files_read_this_call:
            task_manager.mark_file_read(file_path)
        if files_read_this_call:
            combined += f"\n📋 Files marked as read: {len(files_read_this_call)}"

        # Truncate if too long
        max_tokens = get_file_token_limit()
        combined = truncate_text_by_tokens(combined, max_tokens)

        return ToolResult(
            success=len(errors) == 0,
            content=combined,
            error="\n".join(errors) if errors else None,
        )
