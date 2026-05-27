"""Multi-grep tool for searching multiple patterns in one call."""

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any

from ..core.tool_execution import compress_tool_result, should_compress_result
from ..utils.context_cache import get_context_cache
from ..utils.platform_utils import normalize_path_separators as normalize_path
from .base import Tool, ToolResult


class MultiGrepTool(Tool):
    """Search multiple patterns in one tool call.

    Per-call billing optimization: merge multiple grep calls into one.
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "multi_grep"

    @property
    def description(self) -> str:
        return (
            "Search for multiple patterns in files simultaneously. "
            "Returns results grouped by pattern. "
            "Use this instead of multiple grep calls to save API calls."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "searches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Text or regex pattern"},
                            "path": {"type": "string", "description": "Directory or file path", "default": "."},
                            "file_pattern": {
                                "type": "string",
                                "description": "Glob pattern (e.g., *.py)",
                                "default": "*",
                            },
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Case-sensitive search",
                                "default": False,
                            },
                            "regex": {"type": "boolean", "description": "Treat pattern as regex", "default": False},
                            "max_results": {"type": "integer", "description": "Max results per pattern", "default": 50},
                        },
                        "required": ["pattern"],
                    },
                    "description": "List of pattern searches to execute.",
                },
            },
            "required": ["searches"],
        }

    async def execute(self, searches: list[dict[str, Any]]) -> ToolResult:
        """Execute multiple grep searches with caching."""
        searches = _ensure_list(searches)
        results = []
        total_matches = 0
        cache = get_context_cache()

        for search in searches:
            pattern = search.get("pattern", "")
            path = search.get("path", ".")
            file_pattern = search.get("file_pattern", "*")
            case_sensitive = search.get("case_sensitive", False)
            regex = search.get("regex", False)
            max_results = search.get("max_results", 50)

            if not pattern:
                continue

            search_dir = Path(normalize_path(path))
            if not search_dir.is_absolute():
                search_dir = self.workspace_dir / search_dir

            if not search_dir.exists():
                results.append(f"Path not found: {path}")
                continue

            # Check cache first
            cached_result = cache.get_grep_result(pattern, str(search_dir), file_pattern, case_sensitive)
            if cached_result is not None:
                total_matches += len(cached_result)
                header = f"Pattern: {pattern} ({len(cached_result)} matches - cached)"
                results.append(f"{header}\n" + "\n".join(cached_result[:max_results]))
                continue

            # Compile regex if needed
            compiled_pattern = None
            search_text = ""
            if regex:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    compiled_pattern = re.compile(pattern, flags)
                except re.error as e:
                    results.append(f"Pattern: {pattern} - Invalid regex: {e}")
                    continue
            else:
                search_text = pattern if case_sensitive else pattern.lower()

            # Search files
            matches: list[str] = []
            for file_path in self._iterate_files(search_dir, file_pattern):
                if len(matches) >= max_results:
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
                                matches.append(
                                    f"  {file_path.relative_to(self.workspace_dir)}:{line_num}| {line.rstrip()}"
                                )
                except Exception:
                    continue

            total_matches += len(matches)
            cache.set_grep_result(pattern, str(search_dir), file_pattern, case_sensitive, matches)

            header = f"Pattern: {pattern} ({len(matches)} matches)"
            if matches:
                results.append(f"{header}\n" + "\n".join(matches))
            else:
                results.append(f"{header}\n  No matches found")

        combined = "\n\n".join(results)
        combined += f"\n\nTotal: {len(searches)} patterns searched, {total_matches} total matches"

        if should_compress_result("multi_grep", len(combined)):
            compressed = compress_tool_result(ToolResult(success=True, content=combined))
            combined = compressed.content

        return ToolResult(success=True, content=combined)

    def _iterate_files(self, directory: Path, pattern: str) -> Any:
        """Iterate over files matching the pattern."""
        if directory.is_file():
            yield directory
            return
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    yield Path(root) / filename


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
