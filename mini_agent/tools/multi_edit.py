"""Multi-edit tool for performing multiple string replacements in one call."""

import json
from pathlib import Path
from typing import Any

from ..utils.platform_utils import normalize_path_separators as normalize_path
from .base import Tool, ToolResult


class MultiEditTool(Tool):
    """Perform multiple string replacements or create new files in one call.

    Per-call billing optimization: merge multiple edit_file/write_file calls into one.
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()
        self._default_encoding = "utf-8"

    @property
    def name(self) -> str:
        return "multi_edit"

    @property
    def description(self) -> str:
        return (
            "Perform multiple string replacements or create new files in a single call. "
            "Each edit specifies path, old_str, and new_str. "
            "If old_str is empty, creates a new file with new_str as content. "
            "IMPORTANT: Each old_str must match exactly and appear uniquely in its file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path (absolute or relative)"},
                            "old_str": {"type": "string", "description": "Exact string to find (must be unique)"},
                            "new_str": {"type": "string", "description": "Replacement string"},
                            "encoding": {"type": "string", "description": "File encoding (default: utf-8)"},
                        },
                        "required": ["path", "old_str", "new_str"],
                    },
                    "description": "List of edits to apply.",
                },
            },
            "required": ["edits"],
        }

    async def execute(self, edits: list[dict[str, str]]) -> ToolResult:
        """Execute multi-file edit."""
        edits = _ensure_list(edits)
        results = []
        success_count = 0
        error_count = 0
        files_edited = set()

        for i, edit in enumerate(edits):
            path = edit.get("path", "")
            old_str = edit.get("old_str", "")
            new_str = edit.get("new_str", "")
            encoding = edit.get("encoding", self._default_encoding)

            if not path:
                error_count += 1
                results.append(f"Edit #{i + 1}: Missing path")
                continue

            try:
                normalized = normalize_path(path)
                file_path = Path(normalized)
                if not file_path.is_absolute():
                    file_path = self.workspace_dir / file_path

                # Create new file if old_str is empty
                if not old_str:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(new_str, encoding=encoding)
                    files_edited.add(str(file_path))
                    success_count += 1
                    results.append(f"Edit #{i + 1} ({path}): Created new file")
                    continue

                if not file_path.exists():
                    error_count += 1
                    results.append(f"Edit #{i + 1} ({path}): File not found")
                    continue

                content = file_path.read_text(encoding=encoding)

                if old_str not in content:
                    error_count += 1
                    results.append(f"Edit #{i + 1} ({path}): Text not found")
                    continue

                new_content = content.replace(old_str, new_str, 1)
                file_path.write_text(new_content, encoding=encoding)
                files_edited.add(str(file_path))
                success_count += 1
                results.append(f"Edit #{i + 1} ({path}): Applied successfully")

            except UnicodeDecodeError:
                error_count += 1
                results.append(f"Edit #{i + 1} ({path}): Failed to decode with encoding {encoding}")
            except Exception as e:
                error_count += 1
                results.append(f"Edit #{i + 1} ({path}): {str(e)}")

        combined = "\n".join(results)
        summary = f"\nSummary: {success_count} succeeded, {error_count} failed, {len(files_edited)} file(s) modified"
        combined += summary

        return ToolResult(
            success=error_count == 0,
            content=combined,
            error="" if error_count == 0 else f"{error_count} edit(s) failed",
        )


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
