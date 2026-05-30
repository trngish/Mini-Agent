"""工作空间上下文工具，用于一次调用获取全面的项目概览。"""

from pathlib import Path
from typing import Any

from ..core.tool_execution import compress_tool_result, should_compress_result
from .base import Tool, ToolResult
from .batch_shared import get_git_status_sync, get_tree_sync
from .file_tools import get_file_token_limit, truncate_text_by_tokens


class WorkspaceContextTool(Tool):
    """一次调用获取全面的工作空间上下文。

    每次调用计费优化：一次调用获取项目结构、git 状态和关键文件列表。
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "workspace_context"

    @property
    def description(self) -> str:
        return (
            "Get comprehensive workspace context in one call: directory tree, git status, "
            "key config files content. Use this at the START of tasks to get full project context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth for tree (default: 3)",
                    "default": 3,
                },
                "include_git": {
                    "type": "boolean",
                    "description": "Include git status info (default: true)",
                    "default": True,
                },
                "include_config_files": {
                    "type": "boolean",
                    "description": "Include config files content (default: true)",
                    "default": True,
                },
            },
        }

    async def execute(
        self,
        max_depth: int = 3,
        include_git: bool = True,
        include_config_files: bool = True,
    ) -> ToolResult:
        """获取全面的工作空间上下文。"""
        sections = []

        # 1. 目录树
        sections.append(f"目录树 (depth={max_depth})")
        sections.append("=" * 50)
        sections.append(get_tree_sync(self.workspace_dir, max_depth, show_sizes=False, max_files_per_dir=20))

        # 2. Git 状态
        if include_git:
            sections.append("\nGit 状态")
            sections.append("=" * 50)
            sections.append(get_git_status_sync(self.workspace_dir, max_status_lines=30, max_commits=5))

        # 3. 关键配置文件内容
        if include_config_files:
            sections.append("\n找到的关键配置文件")
            sections.append("=" * 50)
            sections.append(self._find_config_files())

            config_content = self._read_config_files_content()
            if config_content:
                sections.append("\n配置文件内容")
                sections.append("=" * 50)
                sections.append(config_content)

        combined = "\n".join(sections)
        max_tokens = get_file_token_limit()
        combined = truncate_text_by_tokens(combined, max_tokens)

        if should_compress_result("workspace_context", len(combined)):
            compressed = compress_tool_result(ToolResult(success=True, content=combined))
            combined = compressed.content

        return ToolResult(success=True, content=combined)

    def _find_config_files(self) -> str:
        """查找关键配置文件。"""
        config_patterns = [
            "package.json",
            "tsconfig.json",
            "pyproject.toml",
            "setup.py",
            "Cargo.toml",
            "go.mod",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            ".env.example",
            "Makefile",
            "justfile",
            "build.gradle",
            "pom.xml",
            "README.md",
            "CONTRIBUTING.md",
            "LICENSE",
            ".eslintrc.js",
            ".eslintrc.json",
            "prettier.config.js",
            "vite.config.*",
            "webpack.config.js",
            "babel.config.js",
            "mypy.ini",
            ".ruff.toml",
            "ruff.toml",
            ".pre-commit-config.yaml",
            "SECURITY.md",
        ]
        found = []
        for pattern in config_patterns:
            if "*" in pattern:
                import fnmatch

                for f in self.workspace_dir.iterdir():
                    if f.is_file() and fnmatch.fnmatch(f.name, pattern):
                        found.append(str(f.name))
            else:
                if (self.workspace_dir / pattern).exists():
                    found.append(pattern)
        return "\n".join(found) if found else "No config files found"

    def _read_config_files_content(self) -> str:
        """读取关键配置文件的内容。"""
        config_patterns = [
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "setup.py",
            "tsconfig.json",
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
        ]
        sections = []
        for pattern in config_patterns:
            path = self.workspace_dir / pattern
            if path.exists() and path.stat().st_size < 30000:
                try:
                    content = path.read_text(encoding="utf-8")
                    sections.append(f"File: {pattern}:\n```\n{content}\n```")
                except Exception:
                    pass
        return "\n\n".join(sections)
