"""Batch operation tools for reducing API call count.

按次数计费优化：合并多个操作为一个工具调用，减少总API调用次数。
- multi_read: 一次读取多个文件
- multi_edit: 一次编辑多个位置（含创建新文件）
- workspace_context: 一次获取完整工作区上下文（含关键文件内容）
- multi_grep: 一次搜索多个模式
- multi_bash: 一次执行多个独立命令
- deep_context: 深度项目上下文分析
"""

import asyncio
import json
import os
import re
import subprocess

from .batch_shared import get_git_status_sync, get_tree_sync
from pathlib import Path
from typing import Any


def _ensure_list(data: list | str | None) -> list:
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

from .base import Tool, ToolResult
from .file_tools import truncate_text_by_tokens, get_file_token_limit
from ..utils.platform_utils import normalize_path_separators as normalize_path


class MultiReadTool(Tool):
    """Read multiple files in a single tool call.

    按次数计费优化：合并多次 read_file 为一次调用，减少API调用次数。
    返回所有文件内容的合并结果，每个文件之间用分隔线隔开。
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
            "Each file's content includes line numbers. Supports offset and limit per file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read (absolute or relative). Can read 2-20 files at once.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed) applied to all files. Use for large files.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines per file. Use for large files to read in chunks.",
                },
            },
            "required": ["paths"],
        }

    async def execute(self, paths: list[str], offset: int | None = None, limit: int | None = None) -> ToolResult:
        """Execute multi-file read."""
        paths = _ensure_list(paths)
        results = []
        total_lines = 0
        errors = []

        for path in paths:
            try:
                normalized = normalize_path(path)
                file_path = Path(normalized)
                if not file_path.is_absolute():
                    file_path = self.workspace_dir / file_path

                if not file_path.exists():
                    errors.append(f"❌ {path}: File not found")
                    continue

                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()

                start = (offset - 1) if offset else 0
                end = (start + limit) if limit else len(lines)
                if start < 0:
                    start = 0
                if end > len(lines):
                    end = len(lines)

                selected_lines = lines[start:end]
                numbered = []
                for i, line in enumerate(selected_lines, start=start + 1):
                    numbered.append(f"{i:6d}|{line.rstrip(chr(10))}")

                content = "\n".join(numbered)
                total_lines += len(numbered)

                # Add file header
                header = f"{'='*60}\n📄 {path} ({len(lines)} lines, showing {start+1}-{end})\n{'='*60}"
                results.append(f"{header}\n{content}")

            except Exception as e:
                errors.append(f"❌ {path}: {str(e)}")

        # Combine all results
        combined = "\n\n".join(results)

        if errors:
            combined = "\n".join(errors) + "\n\n" + combined

        # Add summary
        summary = f"\n{'='*60}\n📊 Summary: {len(results)} file(s) read, {total_lines} total lines"
        if errors:
            summary += f", {len(errors)} error(s)"
        combined += "\n" + summary

        # Apply token truncation with generous limit (tokens are free)
        max_tokens = get_file_token_limit() * 2  # Double the limit for multi-read
        combined = truncate_text_by_tokens(combined, max_tokens)

        return ToolResult(success=True, content=combined)


class MultiEditTool(Tool):
    """Edit multiple locations in one tool call.

    按次数计费优化：合并多次 edit_file 为一次调用，减少API调用次数。
    支持同一文件或不同文件的多个替换操作。
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "multi_edit"

    @property
    def description(self) -> str:
        return (
            "Perform multiple string replacements or create new files in a single call. "
            "Each edit specifies path, old_str, and new_str. If old_str is empty, creates a new file with new_str as content. "
            "All edits are applied independently. "
            "Use this instead of multiple edit_file/write_file calls to save API calls. "
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
                            "path": {
                                "type": "string",
                                "description": "File path (absolute or relative)",
                            },
                            "old_str": {
                                "type": "string",
                                "description": "Exact string to find (must be unique in file)",
                            },
                            "new_str": {
                                "type": "string",
                                "description": "Replacement string",
                            },
                        },
                        "required": ["path", "old_str", "new_str"],
                    },
                    "description": "List of edits to apply. Each edit has path, old_str, new_str.",
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

        # Group edits by file to avoid conflicts within same file
        # Process edits per file in order
        files_edited = set()

        for i, edit in enumerate(edits):
            path = edit.get("path", "")
            old_str = edit.get("old_str", "")
            new_str = edit.get("new_str", "")

            if not path:
                error_count += 1
                results.append(f"❌ Edit #{i+1}: Missing path")
                continue

            try:
                normalized = normalize_path(path)
                file_path = Path(normalized)
                if not file_path.is_absolute():
                    file_path = self.workspace_dir / file_path

                # Support creating new files: if old_str is empty, create the file
                if not old_str:
                    # Create new file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(new_str, encoding="utf-8")
                    files_edited.add(str(file_path))
                    success_count += 1
                    results.append(f"✅ Edit #{i+1} ({path}): Created new file")
                    continue

                if not file_path.exists():
                    error_count += 1
                    results.append(f"❌ Edit #{i+1} ({path}): File not found")
                    continue

                content = file_path.read_text(encoding="utf-8")

                if old_str not in content:
                    error_count += 1
                    results.append(f"❌ Edit #{i+1} ({path}): Text not found: {old_str[:80]}...")
                    continue

                new_content = content.replace(old_str, new_str, 1)  # Replace first occurrence only
                file_path.write_text(new_content, encoding="utf-8")
                files_edited.add(str(file_path))
                success_count += 1
                results.append(f"✅ Edit #{i+1} ({path}): Applied successfully")

            except Exception as e:
                error_count += 1
                results.append(f"❌ Edit #{i+1} ({path}): {str(e)}")

        # Build result message
        combined = "\n".join(results)
        summary = f"\n📊 Summary: {success_count} succeeded, {error_count} failed, {len(files_edited)} file(s) modified"
        combined += "\n" + summary

        return ToolResult(
            success=error_count == 0,
            content=combined,
            error="" if error_count == 0 else f"{error_count} edit(s) failed",
        )


class WorkspaceContextTool(Tool):
    """Get comprehensive workspace context in a single call.

    按次数计费优化：一次调用获取项目结构、git状态、关键文件列表，
    避免分别调用 tree + git_status + find 等多个工具。
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
            "key config files list, and recent file changes. "
            "Use this at the START of tasks to get full project context without multiple tool calls. "
            "This replaces calling tree + git_status + find separately."
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
                    "description": "Include list of key config files found (default: true)",
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
        """Get comprehensive workspace context."""
        sections = []

        # 1. Directory tree
        sections.append(f"📂 Directory Tree (depth={max_depth})")
        sections.append("=" * 50)
        tree_output = self._get_tree(max_depth)
        sections.append(tree_output)

        # 2. Git status
        if include_git:
            sections.append(f"\n🔀 Git Status")
            sections.append("=" * 50)
            git_output = self._get_git_status()
            sections.append(git_output)

        # 3. Key config files - include CONTENT (tokens are free!)
        if include_config_files:
            # First list what's found
            config_list = self._find_config_files()
            sections.append(f"\n📋 Key Config Files Found")
            sections.append("=" * 50)
            sections.append(config_list)

            # Then read their content (tokens are free, info > savings)
            config_content = self._read_config_files_content()
            if config_content:
                sections.append(f"\n📋 Config Files Content")
                sections.append("=" * 50)
                sections.append(config_content)

        combined = "\n".join(sections)
        # Generous token limit since tokens are free
        combined = truncate_text_by_tokens(combined, get_file_token_limit() * 2)
        return ToolResult(success=True, content=combined)

    def _get_tree(self, max_depth: int) -> str:
        """Get directory tree structure."""
        return get_tree_sync(self.workspace_dir, max_depth, show_sizes=False, max_files_per_dir=20)

    def _get_git_status(self) -> str:
        """Get git status information."""
        return get_git_status_sync(self.workspace_dir, max_status_lines=30, max_commits=5)

    def _find_config_files(self) -> str:
        """Find key configuration files in the workspace."""
        config_patterns = [
            "package.json", "tsconfig.json", "pyproject.toml", "setup.py",
            "Cargo.toml", "go.mod", "requirements.txt", "Dockerfile",
            "docker-compose.yml", "docker-compose.yaml", ".env.example",
            "Makefile", "justfile", "build.gradle", "pom.xml",
            "README.md", "README.rst", "CONTRIBUTING.md", "LICENSE",
            ".eslintrc.js", ".eslintrc.json", "prettier.config.js",
            "vite.config.ts", "next.config.js", "next.config.mjs",
            "tailwind.config.js", "webpack.config.js",
        ]

        found = []
        for pattern in config_patterns:
            path = self.workspace_dir / pattern
            if path.exists():
                size = path.stat().st_size
                found.append(f"  ✅ {pattern} ({size} bytes)")

        if not found:
            return "No common config files found"

        return "\n".join(found)

    def _read_config_files_content(self) -> str:
        """Read key configuration files content (tokens are free, so include full content)."""
        # Config files worth reading in full - typically small and critical
        config_patterns = [
            "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
            "requirements.txt", "setup.py", "tsconfig.json",
        ]

        sections = []
        for pattern in config_patterns:
            path = self.workspace_dir / pattern
            if path.exists() and path.stat().st_size < 50000:  # Skip very large files
                try:
                    content = path.read_text(encoding="utf-8")
                    sections.append(f"📄 {pattern}:\n{content}")
                except Exception:
                    pass

        return "\n\n".join(sections) if sections else ""


class MultiGrepTool(Tool):
    """Search multiple patterns in one tool call.

    按次数计费优化：合并多次 grep 为一次调用，减少API调用次数。
    一次搜索多个模式，返回按模式分组的结果。
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "multi_grep"

    @property
    def description(self) -> str:
        return (
            "Search for multiple patterns in files simultaneously. Returns results grouped by pattern. "
            "Use this instead of multiple grep calls to save API calls. "
            "Each pattern search supports its own file_pattern, case_sensitive, and regex options."
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
                            "pattern": {
                                "type": "string",
                                "description": "Text or regex pattern to search for",
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory or file path to search in (default: workspace root)",
                                "default": ".",
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "Glob pattern for file names (e.g., '*.py', '*.md')",
                                "default": "*",
                            },
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Case-sensitive search (default: false)",
                                "default": False,
                            },
                            "regex": {
                                "type": "boolean",
                                "description": "Treat pattern as regex (default: false)",
                                "default": False,
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Max results per pattern (default: 50)",
                                "default": 50,
                            },
                        },
                        "required": ["pattern"],
                    },
                    "description": "List of pattern searches to execute. Each has pattern, path, file_pattern, etc.",
                },
            },
            "required": ["searches"],
        }

    async def execute(self, searches: list[dict[str, Any]]) -> ToolResult:
        """Execute multiple grep searches."""
        searches = _ensure_list(searches)
        import fnmatch

        results = []
        total_matches = 0

        for search in searches:
            pattern = search.get("pattern", "")
            path = search.get("path", ".")
            file_pattern = search.get("file_pattern", "*")
            case_sensitive = search.get("case_sensitive", False)
            regex = search.get("regex", False)
            max_results = search.get("max_results", 50)

            if not pattern:
                continue

            # Normalize path
            search_dir = Path(normalize_path(path))
            if not search_dir.is_absolute():
                search_dir = self.workspace_dir / search_dir

            if not search_dir.exists():
                results.append(f"{'='*60}\n🔍 Pattern: '{pattern}' - Path not found: {path}\n{'='*60}")
                continue

            # Compile regex if needed
            compiled_pattern = None
            search_text = ""
            if regex:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    compiled_pattern = re.compile(pattern, flags)
                except re.error as e:
                    results.append(f"{'='*60}\n🔍 Pattern: '{pattern}' - Invalid regex: {e}\n{'='*60}")
                    continue
            else:
                search_text = pattern if case_sensitive else pattern.lower()

            # Search files
            matches = []
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
                                matches.append(f"  {file_path.relative_to(self.workspace_dir)}:{line_num}| {line.rstrip()}")
                                if len(matches) >= max_results:
                                    break
                except Exception:
                    continue

            total_matches += len(matches)
            header = f"{'='*60}\n🔍 Pattern: '{pattern}' ({len(matches)} matches)\n{'='*60}"
            if matches:
                results.append(f"{header}\n" + "\n".join(matches))
            else:
                results.append(f"{header}\n  No matches found")

        combined = "\n\n".join(results)
        combined += f"\n\n📊 Total: {len(searches)} patterns searched, {total_matches} total matches"
        return ToolResult(success=True, content=combined)

    def _iterate_files(self, directory: Path, pattern: str):
        """Iterate over files matching the pattern."""
        import fnmatch
        import os
        if directory.is_file():
            yield directory
            return
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    yield Path(root) / filename


class MultiBashTool(Tool):
    """Execute multiple independent commands in one tool call.

    按次数计费优化：合并多次 bash 为一次调用，减少API调用次数。
    多个独立命令并行执行，返回每个命令的结果。
    """

    def __init__(self, workspace_dir: str = ".", platform_mode: str = "auto"):
        self.workspace_dir = workspace_dir
        from ..utils.platform_utils import PlatformUtils
        self.is_windows = PlatformUtils.is_windows(platform_mode)

    @property
    def name(self) -> str:
        return "multi_bash"

    @property
    def description(self) -> str:
        return (
            "Execute multiple independent shell commands simultaneously. Returns results for each command. "
            "Use this instead of multiple bash calls to save API calls. "
            "Commands run in parallel - do NOT use for commands that depend on each other. "
            "For dependent commands, chain them with && or ; in a single bash call instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Label for this command (for identification in results)",
                            },
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 60)",
                                "default": 60,
                            },
                        },
                        "required": ["command"],
                    },
                    "description": "List of commands to execute in parallel. Each has command, optional label and timeout.",
                },
            },
            "required": ["commands"],
        }

    async def execute(self, commands: list[dict[str, Any]]) -> ToolResult:
        """Execute multiple commands in parallel."""
        commands = _ensure_list(commands)
        from .bash_shared import get_platform_shell_args, get_subprocess_env
        from ..utils.command_validator import assess_command_danger, DangerLevel, detect_platform_mismatch

        # Validate all commands for dangerous patterns before execution
        blocked = []
        for cmd_dict in commands:
            command = cmd_dict.get("command", "")
            level, reason = assess_command_danger(command)
            if level == DangerLevel.BLOCKED:
                blocked.append(f"{cmd_dict.get('label', command[:40])}: {reason}")
        if blocked:
            return ToolResult(
                success=False,
                content="",
                error="Blocked commands: " + "; ".join(blocked),
            )

        async def run_single(cmd_dict: dict) -> str:
            command = cmd_dict.get("command", "")
            label = cmd_dict.get("label", command[:40])
            timeout = min(cmd_dict.get("timeout", 60), 120)

            # Platform compatibility check
            platform_warning = detect_platform_mismatch(command, self.is_windows)

            try:
                shell_exe, shell_args, _ = get_platform_shell_args(
                    "windows" if self.is_windows else "linux"
                )
                env = get_subprocess_env()

                if self.is_windows:
                    shell_cmd = [shell_exe] + shell_args + [command]
                    process = await asyncio.create_subprocess_exec(
                        *shell_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.workspace_dir,
                        env=env,
                    )
                else:
                    process = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.workspace_dir,
                        env=env,
                    )

                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    return f"❌ [{label}]: Timed out after {timeout}s"

                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                exit_code = process.returncode or 0

                # Prepend platform warning if detected
                if platform_warning:
                    stdout_text = f"{platform_warning}\n\n{stdout_text}" if stdout_text else platform_warning

                if exit_code == 0:
                    output = stdout_text or "(no output)"
                    # Truncate very long outputs
                    if len(output) > 3000:
                        output = output[:3000] + f"\n... (truncated, {len(output)} total chars)"
                    return f"✅ [{label}] (exit={exit_code}):\n{output}"
                else:
                    error = stderr_text or stdout_text or f"exit code {exit_code}"
                    if len(error) > 1500:
                        error = error[:1500] + "..."
                    return f"❌ [{label}] (exit={exit_code}):\n{error}"

            except Exception as e:
                return f"❌ [{label}]: {type(e).__name__}: {str(e)}"

        # Execute all commands in parallel
        tasks = [run_single(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks)

        combined = "\n\n".join(results)
        success_count = sum(1 for r in results if r.startswith("✅"))
        fail_count = len(results) - success_count
        combined += f"\n\n📊 Summary: {success_count} succeeded, {fail_count} failed"

        return ToolResult(
            success=fail_count == 0,
            content=combined,
            error="" if fail_count == 0 else f"{fail_count} command(s) failed",
        )


class DeepContextTool(Tool):
    """Get deep project context in one call - the ultimate context gathering tool.

    按次数计费优化：一次调用获取项目的完整深度上下文，包括目录结构、
    git状态、关键配置文件内容、入口文件内容、依赖关系等。
    替代分别调用 tree + git_status + multi_read + grep 等多个工具。
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
            "key config file contents (pyproject.toml, package.json, etc.), main entry point files, "
            "and project structure analysis. Use this at the START of any task to get full context "
            "without multiple tool calls. This is the most comprehensive context tool - it replaces "
            "calling tree + git_status + multi_read + find separately."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth for tree (default: 4)",
                    "default": 4,
                },
                "read_entry_files": {
                    "type": "boolean",
                    "description": "Read main entry point files (main.py, app.py, etc.) content (default: true)",
                    "default": True,
                },
                "read_config_files": {
                    "type": "boolean",
                    "description": "Read key config files content (default: true)",
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
        """Get deep project context."""
        sections = []

        # 1. Directory tree (deeper than workspace_context)
        sections.append(f"📂 Directory Tree (depth={max_depth})")
        sections.append("=" * 60)
        sections.append(self._get_tree(max_depth))

        # 2. Git status
        sections.append(f"\n🔀 Git Status")
        sections.append("=" * 60)
        sections.append(self._get_git_status())

        # 3. Key config files CONTENT (not just list - tokens are free!)
        if read_config_files:
            config_content = self._read_config_files()
            if config_content:
                sections.append(f"\n📋 Key Config Files (full content)")
                sections.append("=" * 60)
                sections.append(config_content)

        # 4. Entry point files
        if read_entry_files:
            entry_content = self._read_entry_files()
            if entry_content:
                sections.append(f"\n🚀 Entry Point Files")
                sections.append("=" * 60)
                sections.append(entry_content)

        # 5. Project structure hints
        sections.append(f"\n🏗️ Project Structure Analysis")
        sections.append("=" * 60)
        sections.append(self._analyze_structure())

        combined = "\n".join(sections)
        # Generous token limit since tokens are free
        combined = truncate_text_by_tokens(combined, get_file_token_limit() * 3)
        return ToolResult(success=True, content=combined)

    def _get_tree(self, max_depth: int) -> str:
        """Get directory tree structure."""
        return get_tree_sync(self.workspace_dir, max_depth, show_sizes=True, max_files_per_dir=0)

    def _get_git_status(self) -> str:
        """Get git status information."""
        return get_git_status_sync(self.workspace_dir, max_status_lines=50, max_commits=10)

    def _read_config_files(self) -> str:
        """Read key config files content."""
        config_patterns = [
            "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
            "requirements.txt", "setup.py", "tsconfig.json",
            "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ]
        sections = []
        for pattern in config_patterns:
            path = self.workspace_dir / pattern
            if path.exists() and path.stat().st_size < 30000:
                try:
                    content = path.read_text(encoding="utf-8")
                    sections.append(f"📄 {pattern}:\n```\n{content}\n```")
                except Exception:
                    pass
        return "\n\n".join(sections)

    def _read_entry_files(self) -> str:
        """Read common entry point files."""
        entry_patterns = [
            "main.py", "app.py", "index.py", "manage.py", "run.py",
            "src/main.py", "src/app.py", "src/index.ts", "src/index.js",
            "index.ts", "index.js", "server.py", "cli.py",
        ]
        sections = []
        for pattern in entry_patterns:
            path = self.workspace_dir / pattern
            if path.exists() and path.stat().st_size < 20000:
                try:
                    lines = path.read_text(encoding="utf-8").split("\n")
                    # Show first 80 lines
                    content = "\n".join(f"{i+1:6d}|{line}" for i, line in enumerate(lines[:80]))
                    if len(lines) > 80:
                        content += f"\n... ({len(lines)} total lines)"
                    sections.append(f"📄 {pattern}:\n{content}")
                except Exception:
                    pass
        return "\n\n".join(sections)

    def _analyze_structure(self) -> str:
        """Analyze project structure and provide hints."""
        hints = []

        # Detect project type
        if (self.workspace_dir / "pyproject.toml").exists():
            hints.append("📦 Python project (pyproject.toml)")
        if (self.workspace_dir / "package.json").exists():
            hints.append("📦 Node.js project (package.json)")
        if (self.workspace_dir / "Cargo.toml").exists():
            hints.append("📦 Rust project (Cargo.toml)")
        if (self.workspace_dir / "go.mod").exists():
            hints.append("📦 Go project (go.mod)")

        # Detect frameworks
        try:
            for pkg_file in ["package.json", "pyproject.toml", "requirements.txt"]:
                path = self.workspace_dir / pkg_file
                if path.exists():
                    content = path.read_text(encoding="utf-8").lower()
                    if "django" in content:
                        hints.append("🌐 Django framework detected")
                    if "flask" in content:
                        hints.append("🌐 Flask framework detected")
                    if "fastapi" in content:
                        hints.append("🌐 FastAPI framework detected")
                    if "react" in content:
                        hints.append("⚛️ React detected")
                    if "vue" in content:
                        hints.append("💚 Vue.js detected")
                    if "next" in content:
                        hints.append("▲ Next.js detected")
        except Exception:
            pass

        # Detect test frameworks
        test_dirs = ["tests", "test", "__tests__", "spec"]
        for d in test_dirs:
            if (self.workspace_dir / d).exists():
                hints.append(f"🧪 Test directory found: {d}/")
                break

        # Detect CI/CD
        ci_files = [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile", ".circleci"]
        for ci in ci_files:
            if (self.workspace_dir / ci).exists():
                hints.append(f"🔄 CI/CD: {ci}")
                break

        return "\n".join(hints) if hints else "Standard project structure"
