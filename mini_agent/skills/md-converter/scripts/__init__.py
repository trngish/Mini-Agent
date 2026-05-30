#!/usr/bin/env python3
"""
MD Converter - Markdown 转 PDF/DOCX 转换工具

用法:
    from md_converter import convert_to_pdf, convert_to_docx

    convert_to_pdf("input.md", "output.pdf")
    convert_to_docx("input.md", "output.docx")
"""

from .md_to_docx import convert_md_to_docx as convert_to_docx
from .md_to_pdf import convert_md_to_pdf as convert_to_pdf

__all__ = ["convert_to_pdf", "convert_to_docx"]
__version__ = "1.0.0"
