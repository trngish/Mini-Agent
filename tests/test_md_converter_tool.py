"""Tests for md_converter_tool module - MDToPDFTool, MDToDOCXTool, and _load_converter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mini_agent.tools.md_converter_tool import MDToDOCXTool, MDToPDFTool, _load_converter


class TestLoadConverter:
    def test_load_from_package_relative_import(self):
        mock_func = MagicMock()
        mock_module = MagicMock()
        mock_module.convert_md_to_pdf = mock_func

        with patch("importlib.import_module", return_value=mock_module) as mock_import:
            result = _load_converter("md_to_pdf", "convert_md_to_pdf")

        mock_import.assert_called_once_with(".md_to_pdf", "mini_agent.tools")
        assert result is mock_func

    def test_load_fallback_to_scripts_dir(self):
        mock_func = MagicMock()
        mock_module = MagicMock()
        mock_module.convert_md_to_pdf = mock_func

        call_count = 0

        def fake_import_module(name, package=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and package is not None:
                raise ImportError("not found")
            return mock_module

        with patch("importlib.import_module", side_effect=fake_import_module):
            result = _load_converter("md_to_pdf", "convert_md_to_pdf")

        assert result is mock_func

    def test_fallback_adds_scripts_dir_to_sys_path(self):
        mock_func = MagicMock()
        mock_module = MagicMock()
        mock_module.convert_md_to_pdf = mock_func

        scripts_dir = str(Path(__file__).parent.parent / "mini_agent" / "skills" / "md-converter" / "scripts")

        call_count = 0

        def fake_import_module(name, package=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and package is not None:
                raise ImportError("not found")
            return mock_module

        original_path = sys.path.copy()
        try:
            with patch("importlib.import_module", side_effect=fake_import_module):
                _load_converter("md_to_pdf", "convert_md_to_pdf")

            if scripts_dir not in original_path:
                assert scripts_dir in sys.path
        finally:
            sys.path = original_path

    def test_fallback_does_not_duplicate_sys_path(self):
        mock_func = MagicMock()
        mock_module = MagicMock()
        mock_module.convert_md_to_pdf = mock_func

        scripts_dir = str(Path(__file__).parent.parent / "mini_agent" / "skills" / "md-converter" / "scripts")

        call_count = 0

        def fake_import_module(name, package=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and package is not None:
                raise ImportError("not found")
            return mock_module

        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        count_before = sys.path.count(scripts_dir)

        try:
            with patch("importlib.import_module", side_effect=fake_import_module):
                _load_converter("md_to_pdf", "convert_md_to_pdf")

            assert sys.path.count(scripts_dir) == count_before
        finally:
            while sys.path.count(scripts_dir) > count_before:
                sys.path.remove(scripts_dir)


class TestMDToPDFTool:
    def test_name_and_description(self):
        tool = MDToPDFTool()
        assert tool.name == "md_to_pdf"
        assert "PDF" in tool.description

    def test_parameters_schema(self):
        tool = MDToPDFTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "input_path" in params["properties"]
        assert "output_path" in params["properties"]
        assert "title" in params["properties"]
        assert "author" in params["properties"]
        assert "page_size" in params["properties"]
        assert params["required"] == ["input_path", "output_path"]
        assert params["properties"]["page_size"]["default"] == "A4"

    def test_init_default_workspace(self):
        tool = MDToPDFTool()
        assert tool.workspace_dir == Path().absolute()

    def test_init_custom_workspace(self):
        tool = MDToPDFTool(workspace_dir="/tmp/test")
        assert tool.workspace_dir == Path("/tmp/test").absolute()

    @pytest.mark.asyncio
    async def test_execute_input_file_not_found(self):
        tool = MDToPDFTool(workspace_dir="/tmp")
        result = await tool.execute(
            input_path="/nonexistent/file.md",
            output_path="/tmp/out.pdf",
        )
        assert result.success is False
        assert "Input file not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_input_file_not_found_relative_path(self):
        tool = MDToPDFTool(workspace_dir="/tmp")
        result = await tool.execute(
            input_path="nonexistent.md",
            output_path="out.pdf",
        )
        assert result.success is False
        assert "Input file not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = MDToPDFTool(workspace_dir="/tmp")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.pdf",
                title="My Doc",
                author="Author",
                page_size="Letter",
            )

        assert result.success is True
        assert "PDF created successfully" in result.content
        mock_converter.assert_called_once()
        call_kwargs = mock_converter.call_args
        assert call_kwargs[1]["title"] == "My Doc"
        assert call_kwargs[1]["author"] == "Author"
        assert call_kwargs[1]["page_size"] == "Letter"

    @pytest.mark.asyncio
    async def test_execute_success_default_params(self):
        tool = MDToPDFTool(workspace_dir="/tmp")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.pdf",
            )

        assert result.success is True
        call_kwargs = mock_converter.call_args
        assert call_kwargs[1]["title"] is None
        assert call_kwargs[1]["author"] is None
        assert call_kwargs[1]["page_size"] == "A4"

    @pytest.mark.asyncio
    async def test_execute_conversion_failure(self):
        tool = MDToPDFTool(workspace_dir="/tmp")
        mock_converter = MagicMock(side_effect=RuntimeError("PDF engine error"))

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.pdf",
            )

        assert result.success is False
        assert "Conversion failed" in result.error
        assert "PDF engine error" in result.error

    @pytest.mark.asyncio
    async def test_execute_absolute_input_path(self):
        tool = MDToPDFTool(workspace_dir="/tmp")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="/absolute/path/test.md",
                output_path="/absolute/path/test.pdf",
            )

        assert result.success is True
        call_args = mock_converter.call_args[0]
        assert call_args[0].endswith("test.md")
        assert "absolute" in call_args[0].replace("\\", "/")
        assert call_args[1].endswith("test.pdf")
        assert "absolute" in call_args[1].replace("\\", "/")

    @pytest.mark.asyncio
    async def test_execute_relative_path_resolved(self):
        tool = MDToPDFTool(workspace_dir="/workspace")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="docs/test.md",
                output_path="output/test.pdf",
            )

        assert result.success is True
        call_args = mock_converter.call_args[0]
        assert call_args[0].endswith("test.md")
        assert "workspace" in call_args[0].replace("\\", "/")
        assert "docs" in call_args[0].replace("\\", "/")
        assert call_args[1].endswith("test.pdf")
        assert "workspace" in call_args[1].replace("\\", "/")
        assert "output" in call_args[1].replace("\\", "/")

    @pytest.mark.asyncio
    async def test_execute_converter_load_failure(self):
        tool = MDToPDFTool(workspace_dir="/tmp")

        with (
            patch(
                "mini_agent.tools.md_converter_tool._load_converter",
                side_effect=ImportError("no module"),
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.pdf",
            )

        assert result.success is False
        assert "Conversion failed" in result.error


class TestMDToDOCXTool:
    def test_name_and_description(self):
        tool = MDToDOCXTool()
        assert tool.name == "md_to_docx"
        assert "DOCX" in tool.description or "Word" in tool.description

    def test_parameters_schema(self):
        tool = MDToDOCXTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "input_path" in params["properties"]
        assert "output_path" in params["properties"]
        assert "title" in params["properties"]
        assert "author" in params["properties"]
        assert "page_size" not in params["properties"]
        assert params["required"] == ["input_path", "output_path"]

    def test_init_default_workspace(self):
        tool = MDToDOCXTool()
        assert tool.workspace_dir == Path().absolute()

    def test_init_custom_workspace(self):
        tool = MDToDOCXTool(workspace_dir="/tmp/test")
        assert tool.workspace_dir == Path("/tmp/test").absolute()

    @pytest.mark.asyncio
    async def test_execute_input_file_not_found(self):
        tool = MDToDOCXTool(workspace_dir="/tmp")
        result = await tool.execute(
            input_path="/nonexistent/file.md",
            output_path="/tmp/out.docx",
        )
        assert result.success is False
        assert "Input file not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_input_file_not_found_relative_path(self):
        tool = MDToDOCXTool(workspace_dir="/tmp")
        result = await tool.execute(
            input_path="nonexistent.md",
            output_path="out.docx",
        )
        assert result.success is False
        assert "Input file not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = MDToDOCXTool(workspace_dir="/tmp")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.docx",
                title="My Doc",
                author="Author",
            )

        assert result.success is True
        assert "DOCX created successfully" in result.content
        mock_converter.assert_called_once()
        call_kwargs = mock_converter.call_args
        assert call_kwargs[1]["title"] == "My Doc"
        assert call_kwargs[1]["author"] == "Author"

    @pytest.mark.asyncio
    async def test_execute_success_default_params(self):
        tool = MDToDOCXTool(workspace_dir="/tmp")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.docx",
            )

        assert result.success is True
        call_kwargs = mock_converter.call_args
        assert call_kwargs[1]["title"] is None
        assert call_kwargs[1]["author"] is None

    @pytest.mark.asyncio
    async def test_execute_conversion_failure(self):
        tool = MDToDOCXTool(workspace_dir="/tmp")
        mock_converter = MagicMock(side_effect=RuntimeError("DOCX engine error"))

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.docx",
            )

        assert result.success is False
        assert "Conversion failed" in result.error
        assert "DOCX engine error" in result.error

    @pytest.mark.asyncio
    async def test_execute_absolute_input_path(self):
        tool = MDToDOCXTool(workspace_dir="/tmp")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="/absolute/path/test.md",
                output_path="/absolute/path/test.docx",
            )

        assert result.success is True
        call_args = mock_converter.call_args[0]
        assert call_args[0].endswith("test.md")
        assert "absolute" in call_args[0].replace("\\", "/")
        assert call_args[1].endswith("test.docx")
        assert "absolute" in call_args[1].replace("\\", "/")

    @pytest.mark.asyncio
    async def test_execute_relative_path_resolved(self):
        tool = MDToDOCXTool(workspace_dir="/workspace")
        mock_converter = MagicMock()

        with (
            patch("mini_agent.tools.md_converter_tool._load_converter", return_value=mock_converter),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="docs/test.md",
                output_path="output/test.docx",
            )

        assert result.success is True
        call_args = mock_converter.call_args[0]
        assert call_args[0].endswith("test.md")
        assert "workspace" in call_args[0].replace("\\", "/")
        assert "docs" in call_args[0].replace("\\", "/")
        assert call_args[1].endswith("test.docx")
        assert "workspace" in call_args[1].replace("\\", "/")
        assert "output" in call_args[1].replace("\\", "/")

    @pytest.mark.asyncio
    async def test_execute_converter_load_failure(self):
        tool = MDToDOCXTool(workspace_dir="/tmp")

        with (
            patch(
                "mini_agent.tools.md_converter_tool._load_converter",
                side_effect=ImportError("no module"),
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            result = await tool.execute(
                input_path="test.md",
                output_path="test.docx",
            )

        assert result.success is False
        assert "Conversion failed" in result.error


class TestToolSchemas:
    def test_pdf_tool_to_schema(self):
        tool = MDToPDFTool()
        schema = tool.to_schema()
        assert schema["name"] == "md_to_pdf"
        assert "input_schema" in schema
        assert schema["input_schema"]["required"] == ["input_path", "output_path"]

    def test_docx_tool_to_schema(self):
        tool = MDToDOCXTool()
        schema = tool.to_schema()
        assert schema["name"] == "md_to_docx"
        assert "input_schema" in schema

    def test_pdf_tool_to_openai_schema(self):
        tool = MDToPDFTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "md_to_pdf"

    def test_docx_tool_to_openai_schema(self):
        tool = MDToDOCXTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "md_to_docx"
