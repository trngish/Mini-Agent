"""文件搜索和内容搜索工具。

提供类似grep的功能用于搜索文件内容和按模式查找文件。
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from ..utils.platform_utils import normalize_path_separators
from .base import Tool, ToolResult


class GrepTool(Tool):
    """在文件内容中搜索模式。"""

    def __init__(self, workspace_dir: str = "."):
        """使用工作区目录初始化GrepTool。

        参数:
            workspace_dir: 用于解析相对路径的基础目录
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return """在文件中搜索文本模式（类似于grep）。

支持:
- 简单文本搜索
- 正则表达式模式
- 大小写敏感和不敏感搜索
- 显示匹配行的行号

参数:
  - pattern: 要搜索的文本或正则表达式模式
  - path: 要搜索的目录或文件路径（默认: 工作区根目录）
  - file_pattern: 要搜索的文件名的Glob模式（例如："*.py", "*.md"）
  - case_sensitive: 搜索是否区分大小写（默认: false）
  - regex: 模式是否为正则表达式（默认: false）
  - max_results: 返回的最大结果数量（默认: 100）

示例:
  - 在所有Python文件中搜索"TODO"
  - 在src/目录中搜索"function"
  - 使用正则表达式在.py文件中搜索"^class "模式
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "要搜索的文本或正则表达式模式",
                },
                "path": {
                    "type": "string",
                    "description": "要搜索的目录或文件路径（默认: 工作区根目录）",
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "要搜索的文件名的Glob模式（例如：'*.py', '*.md'）",
                    "default": "*",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "搜索是否区分大小写（默认: false）",
                    "default": False,
                },
                "regex": {
                    "type": "boolean",
                    "description": "模式是否为正则表达式（默认: false）",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回的最大结果数量（默认: 100）",
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
        """执行grep搜索。"""
        try:
            # 规范化路径
            search_path = normalize_path_separators(path)
            search_dir = Path(search_path)
            if not search_dir.is_absolute():
                search_dir = self.workspace_dir / search_dir

            if not search_dir.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"路径未找到: {path}",
                )

            # 如果需要则编译正则表达式
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    compiled_pattern = re.compile(pattern, flags)
                except re.error as e:
                    return ToolResult(
                        success=False,
                        content="",
                        error=f"无效的正则表达式模式: {e}",
                    )
            else:
                compiled_pattern = None
                search_text = pattern if case_sensitive else pattern.lower()

            # 搜索文件
            results: list[dict[str, Any]] = []
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
                                results.append(
                                    {
                                        "file": str(file_path.relative_to(self.workspace_dir)),
                                        "line": line_num,
                                        "content": line.rstrip("\n"),
                                    }
                                )

                                if len(results) >= max_results:
                                    break

                except Exception:
                    # 跳过无法读取的文件
                    continue

            # 格式化结果
            if not results:
                return ToolResult(
                    success=True,
                    content=f"在 {files_searched} 个文件中未找到 '{pattern}' 的匹配",
                )

            output = f"在 {files_searched} 个文件中找到 {len(results)} 个匹配:\n\n"
            current_file = None
            for result in results:
                if result["file"] != current_file:
                    current_file = result["file"]
                    output += f"{'=' * 60}\n"
                    output += f"文件: {current_file}\n"
                    output += f"{'=' * 60}\n"

                output += f"  {result['line']:6d}| {result['content']}\n"

            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    def _iterate_files(self, directory: Path, pattern: str) -> Any:
        """遍历目录中匹配模式的文件。"""
        import os

        if directory.is_file():
            yield directory
            return

        for root, dirs, files in os.walk(directory):
            # 跳过隐藏目录和常见的非源代码目录
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]

            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    yield Path(root) / filename


class FindTool(Tool):
    """按名称模式查找文件。"""

    def __init__(self, workspace_dir: str = "."):
        """使用工作区目录初始化FindTool。

        参数:
            workspace_dir: 用于解析相对路径的基础目录
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "find"

    @property
    def description(self) -> str:
        return """按名称模式查找文件（类似于find命令）。

支持Glob模式和大小写敏感/不敏感搜索。

参数:
  - pattern: 文件名的Glob模式（例如："*.py", "test_*.txt"）
  - path: 要搜索的目录（默认: 工作区根目录）
  - case_sensitive: 模式匹配是否区分大小写（默认: true）
  - max_results: 最大结果数量（默认: 100）

示例:
  - 查找所有Python文件: pattern="*.py"
  - 查找所有测试文件: pattern="test_*.py"
  - 查找配置文件: pattern="*.json"
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "文件名的Glob模式（例如：'*.py', 'test_*.txt'）",
                },
                "path": {
                    "type": "string",
                    "description": "要搜索的目录（默认: 工作区根目录）",
                    "default": ".",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "模式匹配是否区分大小写（默认: true）",
                    "default": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大结果数量（默认: 100）",
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
        """执行查找搜索。"""
        try:
            # 规范化路径
            search_path = normalize_path_separators(path)
            search_dir = Path(search_path)
            if not search_dir.is_absolute():
                search_dir = self.workspace_dir / search_dir

            if not search_dir.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"路径未找到: {path}",
                )

            # 查找文件
            results = []
            search_pattern = pattern if case_sensitive else pattern.lower()

            import os

            for root, dirs, files in os.walk(search_dir):
                # 跳过隐藏目录
                dirs[:] = [d for d in dirs if not d.startswith(".")]

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

            # 格式化结果
            if not results:
                return ToolResult(
                    success=True,
                    content=f"未找到匹配模式: {pattern} 的文件",
                )

            output = f"找到 {len(results)} 个匹配 '{pattern}' 的文件:\n\n"
            for file_path in sorted(results):
                output += f"  - {file_path}\n"

            if len(results) >= max_results:
                output += f"\n  ... (显示前 {max_results} 个结果)"

            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class TreeTool(Tool):
    """显示目录树结构。"""

    def __init__(self, workspace_dir: str = "."):
        """使用工作区目录初始化TreeTool。

        参数:
            workspace_dir: 用于解析相对路径的基础目录
        """
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "tree"

    @property
    def description(self) -> str:
        return """显示目录树结构。

显示从给定路径开始的文件和目录层次结构。

参数:
  - path: 要显示的目录路径（默认: 工作区根目录）
  - max_depth: 显示的最大深度（默认: 3）
  - include_hidden: 是否包含隐藏文件/目录（默认: false）

示例:
  - 显示整个工作区树
  - 显示深度为2的项目结构
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要显示的目录路径（默认: 工作区根目录）",
                    "default": ".",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "显示的最大深度（默认: 3）",
                    "default": 3,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "是否包含隐藏文件/目录（默认: false）",
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
        """执行树状显示。"""
        try:
            # 规范化路径
            tree_path = normalize_path_separators(path)
            root_dir = Path(tree_path)
            if not root_dir.is_absolute():
                root_dir = self.workspace_dir / root_dir

            if not root_dir.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"路径未找到: {path}",
                )

            if not root_dir.is_dir():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"不是目录: {path}",
                )

            # 构建树
            lines: list[str] = []
            self._build_tree(root_dir, "", lines, max_depth, 0, include_hidden)

            output = f"目录树: {root_dir.name}\n"
            output += "=" * 60 + "\n"
            output += "\n".join(lines)

            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    def _build_tree(
        self,
        directory: Path,
        prefix: str,
        lines: list[str],
        max_depth: int,
        current_depth: int,
        include_hidden: bool,
    ) -> None:
        """递归构建树状线条。"""
        try:
            entries = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            for i, entry in enumerate(entries):
                # 如果不包含隐藏文件则跳过隐藏文件
                if not include_hidden and entry.name.startswith("."):
                    continue

                is_last = i == len(entries) - 1
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
            lines.append(f"{prefix}[权限被拒绝]")

    def _format_size(self, size: int) -> str:
        """格式化文件大小为人类可读格式。"""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f}K"
        else:
            return f"{size / (1024 * 1024):.1f}M"
