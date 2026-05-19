---
name: md-converter
description: Convert Markdown (.md) files to PDF or Word (.docx) format. Use when user wants to convert a markdown document to PDF or DOCX. Supports tables, code blocks, images, and professional formatting.
---

# MD Converter

Convert Markdown files to PDF or Word documents with professional formatting.

## Quick Start

```python
from md_to_pdf import convert_md_to_pdf
from md_to_docx import convert_md_to_docx

# Convert to PDF
convert_md_to_pdf("input.md", "output.pdf")

# Convert to DOCX
convert_md_to_docx("input.md", "output.docx")
```

## Supported Features

| Feature | PDF | DOCX |
|---------|-----|------|
| Headings (H1-H6) | ✅ | ✅ |
| Bold/Italic | ✅ | ✅ |
| Tables | ✅ | ✅ |
| Code blocks | ✅ | ✅ |
| Images | ✅ | ✅ |
| Lists | ✅ | ✅ |
| Links | ✅ | ✅ |
| Blockquotes | ✅ | ✅ |
| Horizontal rules | ✅ | ✅ |

## Usage

### Convert to PDF

```python
from md_to_pdf import convert_md_to_pdf

# Basic conversion
convert_md_to_pdf("document.md", "document.pdf")

# With options
convert_md_to_pdf(
    "document.md", 
    "document.pdf",
    title="My Document",      # PDF title
    author="Author Name",     # PDF author
    page_size="A4"           # Page size (A4, Letter, etc.)
)
```

### Convert to DOCX

```python
from md_to_docx import convert_md_to_docx

# Basic conversion
convert_md_to_docx("document.md", "document.docx")

# With options
convert_md_to_docx(
    "document.md",
    "document.docx",
    title="My Document",      # Document title
    author="Author Name",     # Document author
    styles=True              # Apply professional styles
)
```

## Installation

Required packages:
```bash
uv pip install reportlab python-docx markdown Pillow
```

## File Paths

- Input: Absolute or relative to workspace
- Output: Will be created in the same directory or specified path