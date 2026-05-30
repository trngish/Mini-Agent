"""深度上下文工具，用于一次调用进行全面的项目分析。"""

from pathlib import Path
from typing import Any

from ..core.tool_execution import compress_tool_result, should_compress_result
from .base import Tool, ToolResult
from .batch_shared import get_git_status_sync, get_tree_sync


class DeepContextTool(Tool):
    """一次调用获取深度项目上下文。

    每次调用计费优化：一次调用获取项目的完整深度上下文。
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "deep_context"

    @property
    def description(self) -> str:
        return (
            "Get deep project context in a single call. Returns directory tree, git status, "
            "key config file contents, entry point files, and project structure analysis."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth (default: 4)",
                    "default": 4,
                },
                "read_entry_files": {
                    "type": "boolean",
                    "description": "Read main entry point files (default: true)",
                    "default": True,
                },
                "read_config_files": {
                    "type": "boolean",
                    "description": "Read key config files (default: true)",
                    "default": True,
                },
            },
        }

    async def execute(
        self,
        max_depth: int = 4,
        read_entry_files: bool = True,
        read_config_files: bool = True,
    ) -> ToolResult:
        """获取深度项目上下文。"""
        sections = []

        # 1. 目录树
        sections.append(f"目录树 (depth={max_depth})")
        sections.append("=" * 60)
        sections.append(get_tree_sync(self.workspace_dir, max_depth))

        # 2. Git 状态
        sections.append("\nGit 状态")
        sections.append("=" * 60)
        sections.append(get_git_status_sync(self.workspace_dir, max_status_lines=50, max_commits=10))

        # 3. 关键配置文件
        if read_config_files:
            config_content = self._read_config_files()
            if config_content:
                sections.append("\n关键配置文件（完整内容）")
                sections.append("=" * 60)
                sections.append(config_content)

        # 4. 入口点文件
        if read_entry_files:
            entry_content = self._read_entry_files()
            if entry_content:
                sections.append("\n入口点文件")
                sections.append("=" * 60)
                sections.append(entry_content)

        # 5. 项目结构分析
        sections.append("\n项目结构分析")
        sections.append("=" * 60)
        sections.append(self._analyze_structure())

        combined = "\n".join(sections)

        if should_compress_result("deep_context", len(combined)):
            compressed = compress_tool_result(ToolResult(success=True, content=combined))
            combined = compressed.content

        return ToolResult(success=True, content=combined)

    def _read_config_files(self) -> str:
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
                    sections.append(f"文件: {pattern}:\n```\n{content}\n```")
                except Exception:
                    pass
        return "\n\n".join(sections)

    def _read_entry_files(self) -> str:
        """读取常见的入口点文件。"""
        entry_patterns = [
            "main.py",
            "app.py",
            "index.py",
            "manage.py",
            "run.py",
            "src/main.py",
            "src/app.py",
            "src/index.ts",
            "src/index.js",
            "index.ts",
            "index.js",
            "server.py",
            "cli.py",
        ]
        sections = []
        for pattern in entry_patterns:
            path = self.workspace_dir / pattern
            if path.exists() and path.stat().st_size < 20000:
                try:
                    lines = path.read_text(encoding="utf-8").split("\n")
                    content = "\n".join(f"{i + 1:6d}|{line}" for i, line in enumerate(lines[:80]))
                    if len(lines) > 80:
                        content += f"\n... ({len(lines)} total lines)"
                    sections.append(f"File: {pattern}:\n{content}")
                except Exception:
                    pass
        return "\n\n".join(sections)

    def _analyze_structure(self) -> str:
        """分析项目结构并提供提示。"""
        hints = []

        # 检测项目类型
        if (self.workspace_dir / "pyproject.toml").exists():
            hints.append("Python 项目 (pyproject.toml)")
        if (self.workspace_dir / "package.json").exists():
            hints.append("Node.js 项目 (package.json)")
        if (self.workspace_dir / "Cargo.toml").exists():
            hints.append("Rust 项目 (Cargo.toml)")
        if (self.workspace_dir / "go.mod").exists():
            hints.append("Go 项目 (go.mod)")

        # 检测框架
        try:
            for pkg_file in ["package.json", "pyproject.toml", "requirements.txt"]:
                path = self.workspace_dir / pkg_file
                if path.exists():
                    content = path.read_text(encoding="utf-8").lower()
                    if "django" in content:
                        hints.append("检测到 Django 框架")
                    if "flask" in content:
                        hints.append("检测到 Flask 框架")
                    if "fastapi" in content:
                        hints.append("检测到 FastAPI 框架")
                    if "react" in content:
                        hints.append("检测到 React")
                    if "vue" in content:
                        hints.append("检测到 Vue.js")
                    if "next" in content:
                        hints.append("检测到 Next.js")
        except Exception:
            pass

        # 检测测试框架
        test_dirs = ["tests", "test", "__tests__", "spec"]
        for d in test_dirs:
            if (self.workspace_dir / d).exists():
                hints.append(f"找到测试目录: {d}/")
                break

        # 检测 CI/CD
        ci_files = [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile", ".circleci"]
        for ci in ci_files:
            if (self.workspace_dir / ci).exists():
                hints.append(f"CI/CD: {ci}")
                break

        return "\n".join(hints) if hints else "标准项目结构"
