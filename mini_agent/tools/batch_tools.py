"""批量操作辅助工具。

本文件现仅包含共享的辅助函数。
实际工具已拆分到独立模块：
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
    """确保输入是列表，必要时解析JSON字符串。

    大语言模型有时会传递JSON编码的字符串而不是正确的列表。
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
