"""Comprehensive unit tests for deep_context tool module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mini_agent.tools.base import ToolResult
from mini_agent.tools.deep_context import DeepContextTool


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tool(temp_workspace):
    return DeepContextTool(workspace_dir=temp_workspace)


class TestDeepContextToolInit:
    def test_default_workspace_dir(self):
        tool = DeepContextTool()
        assert tool.workspace_dir.is_absolute()

    def test_custom_workspace_dir(self, temp_workspace):
        tool = DeepContextTool(workspace_dir=temp_workspace)
        assert str(tool.workspace_dir) == str(Path(temp_workspace).absolute())

    def test_workspace_dir_resolved_to_absolute(self):
        tool = DeepContextTool(workspace_dir=".")
        assert tool.workspace_dir.is_absolute()


class TestDeepContextToolProperties:
    def test_name(self, tool):
        assert tool.name == "deep_context"

    def test_description(self, tool):
        assert "deep project context" in tool.description.lower()
        assert "directory tree" in tool.description.lower()
        assert "git status" in tool.description.lower()

    def test_parameters_schema(self, tool):
        params = tool.parameters
        assert params["type"] == "object"
        assert "max_depth" in params["properties"]
        assert "read_entry_files" in params["properties"]
        assert "read_config_files" in params["properties"]
        assert params["properties"]["max_depth"]["type"] == "integer"
        assert params["properties"]["max_depth"]["default"] == 4
        assert params["properties"]["read_entry_files"]["type"] == "boolean"
        assert params["properties"]["read_entry_files"]["default"] is True
        assert params["properties"]["read_config_files"]["type"] == "boolean"
        assert params["properties"]["read_config_files"]["default"] is True

    def test_to_schema(self, tool):
        schema = tool.to_schema()
        assert schema["name"] == "deep_context"
        assert "description" in schema
        assert "input_schema" in schema

    def test_to_openai_schema(self, tool):
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "deep_context"
        assert "parameters" in schema["function"]


class TestDeepContextToolExecute:
    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main\nWorking tree clean")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./\n  file.py")
    async def test_execute_default(self, mock_tree, mock_git, mock_compress, tool):
        result = await tool.execute()
        assert result.success is True
        assert "目录树" in result.content or "Directory Tree" in result.content
        assert "Git 状态" in result.content or "Git Status" in result.content
        assert "项目结构分析" in result.content or "Project Structure Analysis" in result.content
        mock_tree.assert_called_once_with(tool.workspace_dir, 4)
        mock_git.assert_called_once_with(tool.workspace_dir, max_status_lines=50, max_commits=10)

    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_custom_max_depth(self, mock_tree, mock_git, mock_compress, tool):
        result = await tool.execute(max_depth=2)
        assert result.success is True
        assert "depth=2" in result.content
        mock_tree.assert_called_once_with(tool.workspace_dir, 2)

    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_with_config_files(self, mock_tree, mock_git, mock_compress, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]\nname = 'test'", encoding="utf-8")
        result = await tool.execute(read_config_files=True)
        assert result.success is True
        assert "关键配置文件" in result.content or "Key Config Files" in result.content
        assert "pyproject.toml" in result.content

    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_without_config_files(self, mock_tree, mock_git, mock_compress, tool):
        result = await tool.execute(read_config_files=False)
        assert result.success is True
        assert "关键配置文件" not in result.content or "Key Config Files" not in result.content

    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_with_entry_files(self, mock_tree, mock_git, mock_compress, tool, temp_workspace):
        (Path(temp_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")
        result = await tool.execute(read_entry_files=True)
        assert result.success is True
        assert "入口点文件" in result.content or "Entry Point Files" in result.content
        assert "main.py" in result.content

    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_without_entry_files(self, mock_tree, mock_git, mock_compress, tool):
        result = await tool.execute(read_entry_files=False)
        assert result.success is True
        assert "入口点文件" not in result.content or "Entry Point Files" not in result.content

    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_both_disabled(self, mock_tree, mock_git, mock_compress, tool):
        result = await tool.execute(read_entry_files=False, read_config_files=False)
        assert result.success is True
        assert "关键配置文件" not in result.content or "Key Config Files" not in result.content
        assert "入口点文件" not in result.content or "Entry Point Files" not in result.content
        assert "目录树" in result.content or "Directory Tree" in result.content
        assert "Git 状态" in result.content or "Git Status" in result.content
        assert "项目结构分析" in result.content or "Project Structure Analysis" in result.content

    @patch("mini_agent.tools.deep_context.compress_tool_result")
    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=True)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_with_compression(self, mock_tree, mock_git, mock_compress_check, mock_compress, tool):
        mock_compress.return_value = ToolResult(success=True, content="compressed content")
        result = await tool.execute()
        assert result.success is True
        assert result.content == "compressed content"
        mock_compress_check.assert_called_once()
        mock_compress.assert_called_once()

    @patch("mini_agent.tools.deep_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.deep_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.deep_context.get_tree_sync", return_value="./")
    async def test_execute_no_compression(self, mock_tree, mock_git, mock_compress_check, tool):
        result = await tool.execute()
        assert result.success is True
        mock_compress_check.assert_called_once_with("deep_context", len(result.content))


class TestReadConfigFiles:
    def test_no_config_files(self, tool, temp_workspace):
        result = tool._read_config_files()
        assert result == ""

    def test_single_config_file(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]\nname = 'test'", encoding="utf-8")
        result = tool._read_config_files()
        assert "pyproject.toml" in result
        assert "[project]" in result
        assert "```" in result

    def test_multiple_config_files(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (Path(temp_workspace) / "package.json").write_text('{"name": "test"}', encoding="utf-8")
        result = tool._read_config_files()
        assert "pyproject.toml" in result
        assert "package.json" in result

    def test_config_file_too_large(self, tool, temp_workspace):
        large_content = "x" * 30001
        (Path(temp_workspace) / "pyproject.toml").write_text(large_content, encoding="utf-8")
        result = tool._read_config_files()
        assert result == ""

    def test_config_file_at_size_limit(self, tool, temp_workspace):
        content = "x" * 29999
        (Path(temp_workspace) / "pyproject.toml").write_text(content, encoding="utf-8")
        result = tool._read_config_files()
        assert "pyproject.toml" in result

    def test_config_file_read_exception(self, tool, temp_workspace):
        config_path = Path(temp_workspace) / "pyproject.toml"
        config_path.write_text("content", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = tool._read_config_files()
            assert result == ""

    def test_all_config_patterns(self, tool, temp_workspace):
        patterns = [
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "setup.py",
            "tsconfig.json",
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
        ]
        for pattern in patterns:
            (Path(temp_workspace) / pattern).write_text(f"content of {pattern}", encoding="utf-8")
        result = tool._read_config_files()
        for pattern in patterns:
            assert pattern in result

    def test_nonexistent_config_patterns_skipped(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        result = tool._read_config_files()
        assert "pyproject.toml" in result
        assert "package.json" not in result

    def test_config_file_format(self, tool, temp_workspace):
        (Path(temp_workspace) / "Dockerfile").write_text("FROM python:3.10", encoding="utf-8")
        result = tool._read_config_files()
        assert result.startswith("文件: Dockerfile:\n```\n") or result.startswith("File: Dockerfile:\n```\n")
        assert result.endswith("\n```")


class TestReadEntryFiles:
    def test_no_entry_files(self, tool, temp_workspace):
        result = tool._read_entry_files()
        assert result == ""

    def test_single_entry_file(self, tool, temp_workspace):
        (Path(temp_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")
        result = tool._read_entry_files()
        assert "main.py" in result
        assert "print('hello')" in result

    def test_multiple_entry_files(self, tool, temp_workspace):
        (Path(temp_workspace) / "main.py").write_text("print('main')\n", encoding="utf-8")
        (Path(temp_workspace) / "app.py").write_text("print('app')\n", encoding="utf-8")
        result = tool._read_entry_files()
        assert "main.py" in result
        assert "app.py" in result

    def test_entry_file_with_line_numbers(self, tool, temp_workspace):
        (Path(temp_workspace) / "main.py").write_text("line1\nline2\n", encoding="utf-8")
        result = tool._read_entry_files()
        assert "1|" in result
        assert "2|" in result

    def test_entry_file_truncated_at_80_lines(self, tool, temp_workspace):
        lines = [f"line {i}" for i in range(100)]
        (Path(temp_workspace) / "main.py").write_text("\n".join(lines), encoding="utf-8")
        result = tool._read_entry_files()
        assert "80|" in result
        assert "100 total lines" in result

    def test_entry_file_exactly_80_lines(self, tool, temp_workspace):
        lines = [f"line {i}" for i in range(80)]
        (Path(temp_workspace) / "main.py").write_text("\n".join(lines), encoding="utf-8")
        result = tool._read_entry_files()
        assert "80|" in result
        assert "total lines" not in result

    def test_entry_file_under_80_lines(self, tool, temp_workspace):
        lines = [f"line {i}" for i in range(10)]
        (Path(temp_workspace) / "main.py").write_text("\n".join(lines), encoding="utf-8")
        result = tool._read_entry_files()
        assert "10|" in result
        assert "total lines" not in result

    def test_entry_file_too_large(self, tool, temp_workspace):
        large_content = "x" * 20001
        (Path(temp_workspace) / "main.py").write_text(large_content, encoding="utf-8")
        result = tool._read_entry_files()
        assert result == ""

    def test_entry_file_at_size_limit(self, tool, temp_workspace):
        content = "x" * 19999
        (Path(temp_workspace) / "main.py").write_text(content, encoding="utf-8")
        result = tool._read_entry_files()
        assert "main.py" in result

    def test_entry_file_read_exception(self, tool, temp_workspace):
        (Path(temp_workspace) / "main.py").write_text("content", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = tool._read_entry_files()
            assert result == ""

    def test_all_entry_patterns(self, tool, temp_workspace):
        patterns = [
            "main.py",
            "app.py",
            "index.py",
            "manage.py",
            "run.py",
            "server.py",
            "cli.py",
        ]
        for pattern in patterns:
            (Path(temp_workspace) / pattern).write_text(f"# {pattern}", encoding="utf-8")
        result = tool._read_entry_files()
        for pattern in patterns:
            assert pattern in result

    def test_nested_entry_file(self, tool, temp_workspace):
        src_dir = Path(temp_workspace) / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('src main')\n", encoding="utf-8")
        result = tool._read_entry_files()
        assert "src/main.py" in result

    def test_entry_file_format(self, tool, temp_workspace):
        (Path(temp_workspace) / "main.py").write_text("hello", encoding="utf-8")
        result = tool._read_entry_files()
        assert result.startswith("文件: main.py:\n") or result.startswith("File: main.py:\n")


class TestAnalyzeStructure:
    def test_python_project(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        result = tool._analyze_structure()
        assert "Python 项目" in result or "Python project" in result

    def test_nodejs_project(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text("{}", encoding="utf-8")
        result = tool._analyze_structure()
        assert "Node.js 项目" in result or "Node.js project" in result

    def test_rust_project(self, tool, temp_workspace):
        (Path(temp_workspace) / "Cargo.toml").write_text("[package]", encoding="utf-8")
        result = tool._analyze_structure()
        assert "Rust 项目" in result or "Rust project" in result

    def test_go_project(self, tool, temp_workspace):
        (Path(temp_workspace) / "go.mod").write_text("module test", encoding="utf-8")
        result = tool._analyze_structure()
        assert "Go 项目" in result or "Go project" in result

    def test_multiple_project_types(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (Path(temp_workspace) / "package.json").write_text("{}", encoding="utf-8")
        result = tool._analyze_structure()
        assert "Python 项目" in result or "Python project" in result
        assert "Node.js 项目" in result or "Node.js project" in result

    def test_no_project_type(self, tool, temp_workspace):
        result = tool._analyze_structure()
        assert result == "标准项目结构" or result == "Standard project structure"

    def test_django_detection(self, tool, temp_workspace):
        (Path(temp_workspace) / "requirements.txt").write_text("django==4.0\nflask", encoding="utf-8")
        result = tool._analyze_structure()
        assert "检测到 Django 框架" in result or "Django framework detected" in result

    def test_flask_detection(self, tool, temp_workspace):
        (Path(temp_workspace) / "requirements.txt").write_text("flask==2.0", encoding="utf-8")
        result = tool._analyze_structure()
        assert "检测到 Flask 框架" in result or "Flask framework detected" in result

    def test_fastapi_detection(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("fastapi", encoding="utf-8")
        result = tool._analyze_structure()
        assert "检测到 FastAPI 框架" in result or "FastAPI framework detected" in result

    def test_react_detection(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text('{"dependencies": {"react": "^18"}}', encoding="utf-8")
        result = tool._analyze_structure()
        assert "检测到 React" in result or "React detected" in result

    def test_vue_detection(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text('{"dependencies": {"vue": "^3"}}', encoding="utf-8")
        result = tool._analyze_structure()
        assert "检测到 Vue.js" in result or "Vue.js detected" in result

    def test_nextjs_detection(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text('{"dependencies": {"next": "^13"}}', encoding="utf-8")
        result = tool._analyze_structure()
        assert "检测到 Next.js" in result or "Next.js detected" in result

    def test_framework_detection_case_insensitive(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text('{"name": "My-Django-App"}', encoding="utf-8")
        result = tool._analyze_structure()
        assert "检测到 Django 框架" in result or "Django framework detected" in result

    def test_test_directory_tests(self, tool, temp_workspace):
        (Path(temp_workspace) / "tests").mkdir()
        result = tool._analyze_structure()
        assert "找到测试目录: tests/" in result or "Test directory found: tests/" in result

    def test_test_directory_test(self, tool, temp_workspace):
        (Path(temp_workspace) / "test").mkdir()
        result = tool._analyze_structure()
        assert "找到测试目录: test/" in result or "Test directory found: test/" in result

    def test_test_directory_dunder_tests(self, tool, temp_workspace):
        (Path(temp_workspace) / "__tests__").mkdir()
        result = tool._analyze_structure()
        assert "找到测试目录: __tests__/" in result or "Test directory found: __tests__/" in result

    def test_test_directory_spec(self, tool, temp_workspace):
        (Path(temp_workspace) / "spec").mkdir()
        result = tool._analyze_structure()
        assert "找到测试目录: spec/" in result or "Test directory found: spec/" in result

    def test_only_first_test_directory_reported(self, tool, temp_workspace):
        (Path(temp_workspace) / "tests").mkdir()
        (Path(temp_workspace) / "test").mkdir()
        result = tool._analyze_structure()
        assert result.count("找到测试目录") == 1 or result.count("Test directory found") == 1

    def test_ci_github_workflows(self, tool, temp_workspace):
        workflows = Path(temp_workspace) / ".github" / "workflows"
        workflows.mkdir(parents=True)
        result = tool._analyze_structure()
        assert "CI/CD: .github/workflows" in result

    def test_ci_gitlab(self, tool, temp_workspace):
        (Path(temp_workspace) / ".gitlab-ci.yml").write_text("test:", encoding="utf-8")
        result = tool._analyze_structure()
        assert "CI/CD: .gitlab-ci.yml" in result

    def test_ci_jenkins(self, tool, temp_workspace):
        (Path(temp_workspace) / "Jenkinsfile").write_text("pipeline {}", encoding="utf-8")
        result = tool._analyze_structure()
        assert "CI/CD: Jenkinsfile" in result

    def test_ci_circleci(self, tool, temp_workspace):
        (Path(temp_workspace) / ".circleci").mkdir()
        result = tool._analyze_structure()
        assert "CI/CD: .circleci" in result

    def test_only_first_ci_reported(self, tool, temp_workspace):
        workflows = Path(temp_workspace) / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (Path(temp_workspace) / "Jenkinsfile").write_text("pipeline {}", encoding="utf-8")
        result = tool._analyze_structure()
        assert result.count("CI/CD:") == 1

    def test_framework_detection_exception(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text("{}", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = tool._analyze_structure()
            assert isinstance(result, str)

    def test_combined_analysis(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("fastapi", encoding="utf-8")
        (Path(temp_workspace) / "tests").mkdir()
        workflows = Path(temp_workspace) / ".github" / "workflows"
        workflows.mkdir(parents=True)
        result = tool._analyze_structure()
        assert "Python 项目" in result or "Python project" in result
        assert "检测到 FastAPI 框架" in result or "FastAPI framework detected" in result
        assert "找到测试目录" in result or "Test directory found" in result
        assert "CI/CD" in result
