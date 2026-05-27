"""Batch operation helpers.

This file now only contains shared helper functions.
Actual tools have been split into independent modules:
- multi_read.py: MultiReadTool
- multi_edit.py: MultiEditTool
- multi_grep.py: MultiGrepTool
- workspace_context.py: WorkspaceContextTool
- multi_bash.py: MultiBashTool
- deep_context.py: DeepContextTool
"""

from __future__ import annotations

import json
from typing import Any


def _ensure_list(data: list[Any] | str | None) -> list[Any]:
    """Ensure input is a list, parsing JSON string if needed.

    LLMs sometimes pass JSON-encoded strings instead of proper lists.
    """
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
