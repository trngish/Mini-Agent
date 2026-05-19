#!/usr/bin/env python3
"""
MD Converter - Markdown to PDF/DOCX conversion tools

Usage:
    from md_converter import convert_to_pdf, convert_to_docx
    
    convert_to_pdf("input.md", "output.pdf")
    convert_to_docx("input.md", "output.docx")
"""

from .md_to_pdf import convert_md_to_pdf as convert_to_pdf
from .md_to_docx import convert_md_to_docx as convert_to_docx

__all__ = ["convert_to_pdf", "convert_to_docx"]
__version__ = "1.0.0"