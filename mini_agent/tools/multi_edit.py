"""Multi-edit tool for performing multiple string replacements in one call."""

import json
from pathlib import Path
from typing import Any

from ..utils.platform_utils import normalize_path_separators as normalize_path
from .base import Tool, ToolResult
from .file_tools import _resolve_and_validate_path


class MultiEditTool(Tool):
    """Perform multiple string replacements or create new files in one call.

    Per-call billing optimization: merge multiple edit_file/write_file calls into one.

    Supports multiple parameter formats:
    - {"edits": [{"path": "...", "old_str": "...", "new_str": "..."}]}
    - {"path": "...", "old_str": "...", "new_str": "..."}  (single edit shorthand)
    - {"edits": "[{\"path\": \"...\", ...}]"}  (JSON string format)
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
            "IMPORTANT: Each old_str must match exactly and appear uniquely in its file. "
            "Also supports single edit shorthand: path + old_str + new_str (without edits array)."
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
                "path": {
                    "type": "string",
                    "description": "Single edit: File path (alternative to edits array)",
                },
                "old_str": {
                    "type": "string",
                    "description": "Single edit: Exact string to find (alternative to edits array)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Single edit: Replacement string (alternative to edits array)",
                },
            },
        }

    async def execute(
        self,
        edits: list[dict[str, str]] | None = None,
        path: str | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
    ) -> ToolResult:
        """Execute multi-file edit with flexible parameter handling."""
        # Normalize edits parameter
        edits = _normalize_edits_param(edits, path, old_str, new_str)

        if not edits:
            return ToolResult(
                success=False,
                content="",
                error="No edits provided. Use either 'edits' array or 'path'/'old_str'/'new_str' for single edit.",
            )

        results = []
        success_count = 0
        error_count = 0
        files_edited = set()

        for i, edit in enumerate(edits):
            try:
                # Handle both dict and str inputs
                if isinstance(edit, str):
                    # If edit is a string, try to parse as JSON
                    try:
                        edit = json.loads(edit)
                    except json.JSONDecodeError:
                        results.append(f"Edit #{i + 1}: Invalid edit format")
                        error_count += 1
                        continue

                # Extract edit parameters with fallback
                edit_path = edit.get("path") if isinstance(edit, dict) else None
                edit_old = edit.get("old_str") if isinstance(edit, dict) else None
                edit_new = edit.get("new_str") if isinstance(edit, dict) else None
                encoding = (
                    edit.get("encoding", self._default_encoding) if isinstance(edit, dict) else self._default_encoding
                )

                # Support alternative parameter names
                if not edit_path:
                    edit_path = edit.get("file_path") if isinstance(edit, dict) else None
                if not edit_old:
                    edit_old = edit.get("oldString") if isinstance(edit, dict) else None
                if not edit_new:
                    edit_new = edit.get("newString") if isinstance(edit, dict) else None

                if not edit_path:
                    error_count += 1
                    results.append(f"Edit #{i + 1}: Missing path")
                    continue

                # Validate path stays within workspace and is not blacklisted
                try:
                    file_path = _resolve_and_validate_path(edit_path, self.workspace_dir)
                except ValueError as e:
                    error_count += 1
                    results.append(f"Edit #{i + 1}: {str(e)}")
                    continue

                # Create new file if old_str is empty
                if not edit_old:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(edit_new or "", encoding=encoding)
                    files_edited.add(str(file_path))
                    success_count += 1
                    results.append(f"Edit #{i + 1} ({edit_path}): Created new file")
                    continue

                if not file_path.exists():
                    error_count += 1
                    results.append(f"Edit #{i + 1} ({edit_path}): File not found")
                    continue

                content = file_path.read_text(encoding=encoding)

                if edit_old not in content:
                    error_count += 1
                    results.append(f"Edit #{i + 1} ({edit_path}): Text not found")
                    continue

                new_content = content.replace(edit_old, edit_new or "", 1)
                file_path.write_text(new_content, encoding=encoding)
                files_edited.add(str(file_path))
                success_count += 1
                results.append(f"Edit #{i + 1} ({edit_path}): Applied successfully")

            except UnicodeDecodeError:
                error_count += 1
                results.append(f"Edit #{i + 1}: Failed to decode")
            except Exception as e:
                error_count += 1
                results.append(f"Edit #{i + 1}: {str(e)}")

        combined = "\n".join(results)
        summary = f"\nSummary: {success_count} succeeded, {error_count} failed, {len(files_edited)} file(s) modified"
        combined += summary

        return ToolResult(
            success=error_count == 0,
            content=combined,
            error="" if error_count == 0 else f"{error_count} edit(s) failed",
        )


def _normalize_edits_param(
    edits: list[dict[str, str]] | None,
    path: str | None,
    old_str: str | None,
    new_str: str | None,
) -> list[dict[str, str]]:
    """Normalize edits parameter from various input formats.

    Handles:
    - edits: [{"path": ..., "old_str": ..., "new_str": ...}]
    - edits: "[JSON string]"
    - path + old_str + new_str (single edit shorthand)
    """
    # Case 1: edits is None - check for single edit shorthand
    if not edits:
        if path and old_str is not None:
            return [{"path": path, "old_str": old_str, "new_str": new_str or ""}]
        return []

    # Case 2: edits is already a list
    if isinstance(edits, list):
        return _parse_edits_list(edits)

    # Case 3: edits is a JSON string
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
            if isinstance(parsed, list):
                return _parse_edits_list(parsed)
            elif isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    return []


def _parse_edits_list(edits: list[Any]) -> list[dict[str, str]]:
    """Parse edits list, handling mixed formats."""
    result = []
    for edit in edits:
        if isinstance(edit, str):
            try:
                parsed = json.loads(edit)
                if isinstance(parsed, dict):
                    result.append(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(edit, dict):
            result.append(edit)
    return result
