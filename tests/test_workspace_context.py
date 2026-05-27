"""Comprehensive unit tests for workspace_context tool module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mini_agent.tools.base import ToolResult
from mini_agent.tools.workspace_context import WorkspaceContextTool


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tool(temp_workspace):
    return WorkspaceContextTool(workspace_dir=temp_workspace)


class TestWorkspaceContextToolInit:
    def test_default_workspace_dir(self):
        tool = WorkspaceContextTool()
        assert tool.workspace_dir.is_absolute()

    def test_custom_workspace_dir(self, temp_workspace):
        tool = WorkspaceContextTool(workspace_dir=temp_workspace)
        assert str(tool.workspace_dir) == str(Path(temp_workspace).absolute())

    def test_workspace_dir_resolved_to_absolute(self):
        tool = WorkspaceContextTool(workspace_dir=".")
        assert tool.workspace_dir.is_absolute()


class TestWorkspaceContextToolProperties:
    def test_name(self, tool):
        assert tool.name == "workspace_context"

    def test_description(self, tool):
        assert "workspace context" in tool.description.lower()
        assert "directory tree" in tool.description.lower()
        assert "git status" in tool.description.lower()

    def test_parameters_schema(self, tool):
        params = tool.parameters
        assert params["type"] == "object"
        assert "max_depth" in params["properties"]
        assert "include_git" in params["properties"]
        assert "include_config_files" in params["properties"]
        assert params["properties"]["max_depth"]["type"] == "integer"
        assert params["properties"]["max_depth"]["default"] == 3
        assert params["properties"]["include_git"]["type"] == "boolean"
        assert params["properties"]["include_git"]["default"] is True
        assert params["properties"]["include_config_files"]["type"] == "boolean"
        assert params["properties"]["include_config_files"]["default"] is True

    def test_to_schema(self, tool):
        schema = tool.to_schema()
        assert schema["name"] == "workspace_context"
        assert "description" in schema
        assert "input_schema" in schema

    def test_to_openai_schema(self, tool):
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "workspace_context"
        assert "parameters" in schema["function"]


class TestWorkspaceContextToolExecute:
    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main\nWorking tree clean")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./\n  file.py")
    async def test_execute_default(self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool):
        result = await tool.execute()
        assert result.success is True
        assert "Directory Tree" in result.content
        assert "Git Status" in result.content
        assert "Key Config Files Found" in result.content
        mock_tree.assert_called_once_with(tool.workspace_dir, 3, show_sizes=False, max_files_per_dir=20)
        mock_git.assert_called_once_with(tool.workspace_dir, max_status_lines=30, max_commits=5)
        mock_token_limit.assert_called_once()

    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_custom_max_depth(
        self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool
    ):
        result = await tool.execute(max_depth=5)
        assert result.success is True
        assert "depth=5" in result.content
        mock_tree.assert_called_once_with(tool.workspace_dir, 5, show_sizes=False, max_files_per_dir=20)

    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_without_git(self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool):
        result = await tool.execute(include_git=False)
        assert result.success is True
        assert "Git Status" not in result.content
        mock_git.assert_not_called()

    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_without_config_files(
        self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool
    ):
        result = await tool.execute(include_config_files=False)
        assert result.success is True
        assert "Key Config Files Found" not in result.content
        assert "Config Files Content" not in result.content

    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_both_disabled(
        self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool
    ):
        result = await tool.execute(include_git=False, include_config_files=False)
        assert result.success is True
        assert "Git Status" not in result.content
        assert "Key Config Files Found" not in result.content
        assert "Directory Tree" in result.content

    @patch("mini_agent.tools.workspace_context.compress_tool_result")
    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=True)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_with_compression(
        self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress_check, mock_compress, tool
    ):
        mock_compress.return_value = ToolResult(success=True, content="compressed content")
        result = await tool.execute()
        assert result.success is True
        assert result.content == "compressed content"
        mock_compress_check.assert_called_once()
        mock_compress.assert_called_once()

    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", return_value="truncated")
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_token_truncation(
        self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool
    ):
        result = await tool.execute()
        assert result.success is True
        mock_truncate.assert_called_once()
        assert result.content == "truncated"

    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_with_config_files_content(
        self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool, temp_workspace
    ):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]\nname = 'test'", encoding="utf-8")
        result = await tool.execute(include_config_files=True)
        assert result.success is True
        assert "Key Config Files Found" in result.content
        assert "Config Files Content" in result.content
        assert "pyproject.toml" in result.content

    @patch("mini_agent.tools.workspace_context.should_compress_result", return_value=False)
    @patch("mini_agent.tools.workspace_context.truncate_text_by_tokens", side_effect=lambda text, limit: text)
    @patch("mini_agent.tools.workspace_context.get_file_token_limit", return_value=64000)
    @patch("mini_agent.tools.workspace_context.get_git_status_sync", return_value="Branch: main")
    @patch("mini_agent.tools.workspace_context.get_tree_sync", return_value="./")
    async def test_execute_no_config_content_section_when_empty(
        self, mock_tree, mock_git, mock_token_limit, mock_truncate, mock_compress, tool
    ):
        result = await tool.execute(include_config_files=True)
        assert result.success is True
        assert "Key Config Files Found" in result.content
        assert "Config Files Content" not in result.content


class TestFindConfigFiles:
    def test_no_config_files(self, tool, temp_workspace):
        result = tool._find_config_files()
        assert result == "No config files found"

    def test_single_config_file(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text("{}", encoding="utf-8")
        result = tool._find_config_files()
        assert "package.json" in result

    def test_multiple_config_files(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text("{}", encoding="utf-8")
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        result = tool._find_config_files()
        assert "package.json" in result
        assert "pyproject.toml" in result

    def test_glob_pattern_vite_config(self, tool, temp_workspace):
        (Path(temp_workspace) / "vite.config.ts").write_text("export default {}", encoding="utf-8")
        result = tool._find_config_files()
        assert "vite.config.ts" in result

    def test_glob_pattern_vite_config_js(self, tool, temp_workspace):
        (Path(temp_workspace) / "vite.config.js").write_text("export default {}", encoding="utf-8")
        result = tool._find_config_files()
        assert "vite.config.js" in result

    def test_glob_pattern_no_match(self, tool, temp_workspace):
        (Path(temp_workspace) / "other.config.ts").write_text("export default {}", encoding="utf-8")
        result = tool._find_config_files()
        assert "other.config.ts" not in result

    def test_glob_pattern_ignores_directories(self, tool, temp_workspace):
        vite_dir = Path(temp_workspace) / "vite.config.dir"
        vite_dir.mkdir()
        result = tool._find_config_files()
        assert "vite.config.dir" not in result

    def test_all_non_glob_patterns(self, tool, temp_workspace):
        patterns = [
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
            "webpack.config.js",
            "babel.config.js",
            "mypy.ini",
            ".ruff.toml",
            "ruff.toml",
            ".pre-commit-config.yaml",
            "SECURITY.md",
        ]
        for pattern in patterns:
            (Path(temp_workspace) / pattern).write_text(f"content of {pattern}", encoding="utf-8")
        result = tool._find_config_files()
        for pattern in patterns:
            assert pattern in result

    def test_found_files_joined_by_newline(self, tool, temp_workspace):
        (Path(temp_workspace) / "package.json").write_text("{}", encoding="utf-8")
        (Path(temp_workspace) / "README.md").write_text("# Test", encoding="utf-8")
        result = tool._find_config_files()
        lines = result.split("\n")
        assert len(lines) == 2


class TestReadConfigFilesContent:
    def test_no_config_files(self, tool, temp_workspace):
        result = tool._read_config_files_content()
        assert result == ""

    def test_single_config_file(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]\nname = 'test'", encoding="utf-8")
        result = tool._read_config_files_content()
        assert "pyproject.toml" in result
        assert "[project]" in result
        assert "```" in result

    def test_multiple_config_files(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (Path(temp_workspace) / "package.json").write_text('{"name": "test"}', encoding="utf-8")
        result = tool._read_config_files_content()
        assert "pyproject.toml" in result
        assert "package.json" in result

    def test_config_file_too_large(self, tool, temp_workspace):
        large_content = "x" * 30001
        (Path(temp_workspace) / "pyproject.toml").write_text(large_content, encoding="utf-8")
        result = tool._read_config_files_content()
        assert result == ""

    def test_config_file_at_size_limit(self, tool, temp_workspace):
        content = "x" * 29999
        (Path(temp_workspace) / "pyproject.toml").write_text(content, encoding="utf-8")
        result = tool._read_config_files_content()
        assert "pyproject.toml" in result

    def test_config_file_read_exception(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("content", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = tool._read_config_files_content()
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
        result = tool._read_config_files_content()
        for pattern in patterns:
            assert pattern in result

    def test_nonexistent_config_patterns_skipped(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        result = tool._read_config_files_content()
        assert "pyproject.toml" in result
        assert "package.json" not in result

    def test_config_file_format(self, tool, temp_workspace):
        (Path(temp_workspace) / "Dockerfile").write_text("FROM python:3.10", encoding="utf-8")
        result = tool._read_config_files_content()
        assert result.startswith("File: Dockerfile:\n```\n")
        assert result.endswith("\n```")

    def test_mixed_size_files(self, tool, temp_workspace):
        (Path(temp_workspace) / "pyproject.toml").write_text("[project]", encoding="utf-8")
        large_content = "x" * 30001
        (Path(temp_workspace) / "package.json").write_text(large_content, encoding="utf-8")
        result = tool._read_config_files_content()
        assert "pyproject.toml" in result
        assert "package.json" not in result

    def test_docker_compose_yaml_variant(self, tool, temp_workspace):
        (Path(temp_workspace) / "docker-compose.yaml").write_text("services:", encoding="utf-8")
        result = tool._read_config_files_content()
        assert "docker-compose.yaml" in result

    def test_docker_compose_yml_variant(self, tool, temp_workspace):
        (Path(temp_workspace) / "docker-compose.yml").write_text("services:", encoding="utf-8")
        result = tool._read_config_files_content()
        assert "docker-compose.yml" in result
