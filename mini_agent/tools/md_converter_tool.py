"""
MD Converter Tool - Convert Markdown files to PDF or DOCX

This tool provides direct conversion functionality for Mini Agent.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from .base import Tool, ToolResult


class MDToPDFTool(Tool):
    """Convert Markdown to PDF."""
    
    name = "md_to_pdf"
    description = "Convert a Markdown (.md) file to PDF format. Supports headings, tables, code blocks, images, and professional formatting."
    
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
        title: Optional[str] = None,
        author: Optional[str] = None,
        page_size: str = "A4"
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
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Input file not found: {input_file}"
                )
            
            # Import converter
            try:
                from .md_to_pdf import convert_md_to_pdf
            except ImportError:
                # Try from skills directory
                sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "md-converter" / "scripts"))
                from md_to_pdf import convert_md_to_pdf
            
            # Convert
            convert_md_to_pdf(
                str(input_file),
                str(output_file),
                title=title,
                author=author,
                page_size=page_size
            )
            
            return ToolResult(
                success=True,
                content=f"PDF created successfully: {output_file}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Conversion failed: {str(e)}"
            )


class MDToDOCXTool(Tool):
    """Convert Markdown to DOCX."""
    
    name = "md_to_docx"
    description = "Convert a Markdown (.md) file to Word (.docx) format. Supports headings, tables, code blocks, lists, and inline formatting."
    
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
        title: Optional[str] = None,
        author: Optional[str] = None,
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
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Input file not found: {input_file}"
                )
            
            # Import converter
            try:
                from .md_to_docx import convert_md_to_docx
            except ImportError:
                # Try from skills directory
                sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "md-converter" / "scripts"))
                from md_to_docx import convert_md_to_docx
            
            # Convert
            convert_md_to_docx(
                str(input_file),
                str(output_file),
                title=title,
                author=author
            )
            
            return ToolResult(
                success=True,
                content=f"DOCX created successfully: {output_file}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Conversion failed: {str(e)}"
            )