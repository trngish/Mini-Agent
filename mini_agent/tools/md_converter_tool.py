"""
MD Converter Tool - Convert Markdown files to PDF or DOCX

This tool provides direct conversion functionality for Mini Agent.
"""

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


def _load_converter(module_name: str, func_name: str) -> Callable[..., Any]:
    """Dynamically load a converter function from the skills directory.

    The converter modules live under mini_agent/skills/md-converter/scripts/
    and are not part of the tools package, so they must be loaded at runtime.

    Args:
        module_name: Module name to import (e.g. "md_to_pdf")
        func_name: Function name to retrieve (e.g. "convert_md_to_pdf")

    Returns:
        The converter function

    Raises:
        ImportError: If the module cannot be found
    """
    try:
        func = getattr(importlib.import_module(f".{module_name}", __package__), func_name)
        return func  # type: ignore[no-any-return]
    except ImportError:
        scripts_dir = str(Path(__file__).parent.parent / "skills" / "md-converter" / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func  # type: ignore[no-any-return]


class MDToPDFTool(Tool):
    """Convert Markdown to PDF."""

    name = "md_to_pdf"
    description = (
        "Convert a Markdown (.md) file to PDF format."
        " Supports headings, tables, code blocks, images, and professional formatting."
    )

    parameters = {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "Path to input Markdown file (.md)",
            },
            "output_path": {
                "type": "string",
                "description": "Path to output PDF file (.pdf)",
            },
            "title": {
                "type": "string",
                "description": "PDF document title (optional, defaults to filename)",
            },
            "author": {
                "type": "string",
                "description": "PDF author name (optional)",
            },
            "page_size": {
                "type": "string",
                "description": "Page size: A4, Letter, Legal, A3 (default: A4)",
                "default": "A4",
            },
        },
        "required": ["input_path", "output_path"],
    }

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    async def execute(
        self,
        input_path: str,
        output_path: str,
        title: str | None = None,
        author: str | None = None,
        page_size: str = "A4",
    ) -> ToolResult:
        try:
            # Resolve paths
            input_file = Path(input_path)
            if not input_file.is_absolute():
                input_file = self.workspace_dir / input_file

            output_file = Path(output_path)
            if not output_file.is_absolute():
                output_file = self.workspace_dir / output_file

            if not input_file.exists():
                return ToolResult(success=False, content="", error=f"Input file not found: {input_file}")

            convert_md_to_pdf = _load_converter("md_to_pdf", "convert_md_to_pdf")

            convert_md_to_pdf(str(input_file), str(output_file), title=title, author=author, page_size=page_size)

            return ToolResult(success=True, content=f"PDF created successfully: {output_file}")

        except Exception as e:
            return ToolResult(success=False, content="", error=f"Conversion failed: {str(e)}")


class MDToDOCXTool(Tool):
    """Convert Markdown to DOCX."""

    name = "md_to_docx"
    description = (
        "Convert a Markdown (.md) file to Word (.docx) format."
        " Supports headings, tables, code blocks, lists, and inline formatting."
    )

    parameters = {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "Path to input Markdown file (.md)",
            },
            "output_path": {
                "type": "string",
                "description": "Path to output DOCX file (.docx)",
            },
            "title": {
                "type": "string",
                "description": "Document title (optional, defaults to filename)",
            },
            "author": {
                "type": "string",
                "description": "Document author (optional)",
            },
        },
        "required": ["input_path", "output_path"],
    }

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    async def execute(
        self,
        input_path: str,
        output_path: str,
        title: str | None = None,
        author: str | None = None,
    ) -> ToolResult:
        try:
            # Resolve paths
            input_file = Path(input_path)
            if not input_file.is_absolute():
                input_file = self.workspace_dir / input_file

            output_file = Path(output_path)
            if not output_file.is_absolute():
                output_file = self.workspace_dir / output_file

            if not input_file.exists():
                return ToolResult(success=False, content="", error=f"Input file not found: {input_file}")

            convert_md_to_docx = _load_converter("md_to_docx", "convert_md_to_docx")

            convert_md_to_docx(str(input_file), str(output_file), title=title, author=author)

            return ToolResult(success=True, content=f"DOCX created successfully: {output_file}")

        except Exception as e:
            return ToolResult(success=False, content="", error=f"Conversion failed: {str(e)}")
