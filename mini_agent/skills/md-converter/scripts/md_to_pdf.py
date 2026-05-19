#!/usr/bin/env python3
"""
Markdown to PDF Converter - Via DOCX

Two-step conversion:
1. Markdown → DOCX (using python-docx, good Chinese support)
2. DOCX → PDF (using LibreOffice headless)

Dependencies:
    pip install python-docx markdown
    # Also need LibreOffice installed for PDF conversion

Usage:
    from md_to_pdf import convert_md_to_pdf
    convert_md_to_pdf("input.md", "output.pdf")
"""

import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

try:
    import markdown
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Install with: uv pip install python-docx markdown")
    sys.exit(1)


def find_libreoffice():
    """Find LibreOffice executable path."""
    # Common Windows paths
    windows_paths = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        r"C:\Program Files\LibreOffice 7\program\soffice.exe",
        r"C:\Program Files\LibreOffice 8\program\soffice.exe",
    ]
    
    # Common Linux paths
    linux_paths = [
        "/usr/bin/soffice",
        "/usr/local/bin/soffice",
        "/opt/libreoffice7.0/program/soffice",
        "/opt/libreoffice7.1/program/soffice",
        "/opt/libreoffice7.2/program/soffice",
        "/opt/libreoffice7.3/program/soffice",
        "/opt/libreoffice7.4/program/soffice",
        "/opt/libreoffice7.5/program/soffice",
        "/opt/libreoffice7.6/program/soffice",
    ]
    
    # Check Windows paths
    if sys.platform.startswith('win'):
        for path in windows_paths:
            if os.path.exists(path):
                return path
        # Try to find in PATH
        try:
            result = subprocess.run(['where', 'soffice'], capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
    
    # Check Linux paths
    for path in linux_paths:
        if os.path.exists(path):
            return path
    
    # Try which command on Linux/Mac
    for cmd in ['soffice', 'libreoffice']:
        try:
            result = subprocess.run(['which', cmd], capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
    
    return None


def docx_to_pdf(docx_path, pdf_path, libreoffice_path=None):
    """Convert DOCX to PDF using LibreOffice headless.
    
    Args:
        docx_path: Path to input .docx file
        pdf_path: Path to output .pdf file  
        libreoffice_path: Optional path to LibreOffice executable
        
    Returns:
        True if successful, False otherwise
    """
    if libreoffice_path is None:
        libreoffice_path = find_libreoffice()
    
    if libreoffice_path is None:
        raise RuntimeError(
            "LibreOffice not found. Please install LibreOffice:\n"
            "Windows: https://www.libreoffice.org/download/download/\n"
            "Linux: sudo apt install libreoffice\n"
            "Mac: brew install --cask libreoffice"
        )
    
    docx_path = Path(docx_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    
    # Ensure output directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command - LibreOffice headless conversion
    cmd = [
        libreoffice_path,
        '--headless',
        '--convert-to', 'pdf',
        '--outdir', str(pdf_path.parent),
        str(docx_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=120
        )
        
        # LibreOffice outputs to same directory as input by default with --outdir
        # But it might also output with the same filename
        expected_pdf = pdf_path.parent / f"{docx_path.stem}.pdf"
        
        if expected_pdf.exists() and expected_pdf != pdf_path:
            # Rename to desired output path
            if pdf_path.exists():
                pdf_path.unlink()
            expected_pdf.rename(pdf_path)
        
        return pdf_path.exists()
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice conversion timed out")
    except Exception as e:
        raise RuntimeError(f"LibreOffice conversion failed: {e}")


def process_inline_formatting(text):
    """Process inline formatting and return list of (type, text) tuples.
    
    Returns list of (type, text) tuples where type is 'text', 'bold', 'italic', 'code'.
    """
    parts = []
    
    # Combined pattern for all inline formatting
    # Order matters: process code first, then bold, then italic
    pattern = r'(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_)'
    
    last_end = 0
    for match in re.finditer(pattern, text):
        # Add text before match
        if match.start() > last_end:
            parts.append(('text', text[last_end:match.start()]))
        
        matched = match.group()
        if matched.startswith('`') and matched.endswith('`'):
            parts.append(('code', matched[1:-1]))
        elif matched.startswith('**') and matched.endswith('**'):
            parts.append(('bold', matched[2:-2]))
        elif matched.startswith('__') and matched.endswith('__'):
            parts.append(('bold', matched[2:-2]))
        elif matched.startswith('*') and matched.endswith('*'):
            parts.append(('italic', matched[1:-1]))
        elif matched.startswith('_') and matched.endswith('_'):
            parts.append(('italic', matched[1:-1]))
        
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        parts.append(('text', text[last_end:]))
    
    return parts if parts else [('text', text)]


def set_cell_shading(cell, color):
    """Set cell background color."""
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color)
    cell._tc.get_or_add_tcPr().append(shading)


def parse_markdown_to_docx(doc, md_content):
    """Parse markdown content and add elements to docx document."""
    lines = md_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines
        if not line.strip():
            i += 1
            continue
        
        # Headings
        if line.startswith('#######'):
            # H7 - treat as body
            p = doc.add_paragraph(line[7:].strip())
        elif line.startswith('######'):
            p = doc.add_heading(line[6:].strip(), level=6)
        elif line.startswith('#####'):
            p = doc.add_heading(line[5:].strip(), level=5)
        elif line.startswith('####'):
            p = doc.add_heading(line[4:].strip(), level=4)
        elif line.startswith('###'):
            p = doc.add_heading(line[3:].strip(), level=3)
        elif line.startswith('##'):
            p = doc.add_heading(line[2:].strip(), level=2)
        elif line.startswith('#'):
            p = doc.add_heading(line[1:].strip(), level=1)
        
        # Horizontal rule
        elif line.strip() in ['---', '***', '___']:
            p = doc.add_paragraph()
            p.add_run('─' * 50)
        
        # Blockquote
        elif line.startswith('>'):
            text = line[1:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1)
            run = p.add_run(text)
            run.italic = True
        
        # Unordered list
        elif line.strip().startswith(('- ', '* ', '+ ')):
            while i < len(lines) and lines[i].strip().startswith(('- ', '* ', '+ ')):
                text = lines[i].strip()[2:].strip()
                doc.add_paragraph(text, style='List Bullet')
                i += 1
            continue
        
        # Ordered list
        elif re.match(r'^\d+\.\s', line):
            while i < len(lines) and re.match(r'^\d+\.\s', lines[i]):
                text = re.sub(r'^\d+\.\s', '', lines[i]).strip()
                doc.add_paragraph(text, style='List Number')
                i += 1
            continue
        
        # Table
        elif line.startswith('|'):
            table_data = []
            
            # Parse table rows
            while i < len(lines) and lines[i].strip().startswith('|'):
                row_text = lines[i].strip().strip('|')
                if row_text and not all(c in '-: ' for c in row_text):
                    cells = [c.strip() for c in row_text.split('|')]
                    table_data.append(cells)
                i += 1
            
            if table_data and len(table_data) >= 2:
                headers = table_data[0]
                rows = table_data[2:] if len(table_data) > 2 else []
                
                # Create table
                table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
                table.style = 'Table Grid'
                
                # Header row
                for col_idx, header_text in enumerate(headers):
                    cell = table.rows[0].cells[col_idx]
                    cell.text = header_text
                    # Set header background color
                    set_cell_shading(cell, '4A90D9')
                    # Set white text color for header
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.color.rgb = RGBColor(255, 255, 255)
                            run.bold = True
                
                # Data rows with alternating colors
                for row_idx, row_data in enumerate(rows):
                    for col_idx, cell_text in enumerate(row_data):
                        cell = table.rows[row_idx + 1].cells[col_idx]
                        cell.text = cell_text
                        # Alternate row colors
                        if row_idx % 2 == 1:
                            set_cell_shading(cell, 'F8F8F8')
                
                doc.add_paragraph()  # Space after table
            continue
        
        # Code block
        elif line.strip().startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            
            code_text = '\n'.join(code_lines)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            run = p.add_run(code_text)
            run.font.name = 'Courier New'
            run.font.size = Pt(9)
            # Set gray background
            p.paragraph_format.shading = True
            i += 1
            continue
        
        # Regular paragraph with inline formatting
        else:
            p = doc.add_paragraph()
            
            # Process inline formatting
            formatted_parts = process_inline_formatting(line)
            
            for part_type, text in formatted_parts:
                if not text:
                    continue
                
                run = p.add_run(text)
                
                if part_type == 'bold':
                    run.bold = True
                elif part_type == 'italic':
                    run.italic = True
                elif part_type == 'code':
                    run.font.name = 'Courier New'
                    run.font.color.rgb = RGBColor(0x2e, 0x7d, 0x32)
                    run.font.size = Pt(10)
                else:
                    # Normal text - set default font for Chinese
                    run.font.name = 'Microsoft YaHei'
        
        i += 1


def convert_md_to_pdf(
    input_path: str,
    output_path: str,
    title: str = None,
    author: str = None,
    page_size: str = "A4"
):
    """
    Convert a Markdown file to PDF via DOCX.
    
    Two-step conversion:
    1. Markdown → DOCX (using python-docx)
    2. DOCX → PDF (using LibreOffice)
    
    Args:
        input_path: Path to input .md file
        output_path: Path to output .pdf file
        title: Document title (default: filename)
        author: Document author (default: "MD Converter")
        page_size: Page size - A4, Letter, Legal, A3 (default: A4)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read markdown content
    md_content = input_path.read_text(encoding='utf-8')
    
    # Set defaults
    if title is None:
        title = input_path.stem
    if author is None:
        author = "MD Converter"
    
    # Create document
    doc = Document()
    
    # Set document properties
    doc.core_properties.title = title
    doc.core_properties.author = author
    doc.core_properties.created = datetime.now()
    
    # Set default font for Chinese
    style = doc.styles['Normal']
    style.font.name = 'Microsoft YaHei'
    style.font.size = Pt(11)
    
    # Add title if not already in markdown
    if not md_content.strip().startswith('#'):
        heading = doc.add_heading(title, 0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Parse markdown
    parse_markdown_to_docx(doc, md_content)
    
    # Create temp DOCX file
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_docx:
        tmp_docx_path = Path(tmp_docx.name)
    
    try:
        # Save DOCX to temp file
        doc.save(str(tmp_docx_path))
        
        # Convert DOCX to PDF
        result = docx_to_pdf(tmp_docx_path, output_path)
        
        if result:
            print(f"✅ PDF created: {output_path}")
            return str(output_path)
        else:
            raise RuntimeError("PDF conversion failed")
    
    finally:
        # Clean up temp file
        if tmp_docx_path.exists():
            tmp_docx_path.unlink()


def convert_md_to_docx(
    input_path: str,
    output_path: str,
    title: str = None,
    author: str = None
):
    """
    Convert a Markdown file to DOCX.
    
    Args:
        input_path: Path to input .md file
        output_path: Path to output .docx file
        title: Document title (default: filename)
        author: Document author (default: "MD Converter")
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read markdown content
    md_content = input_path.read_text(encoding='utf-8')
    
    # Set defaults
    if title is None:
        title = input_path.stem
    if author is None:
        author = "MD Converter"
    
    # Create document
    doc = Document()
    
    # Set document properties
    doc.core_properties.title = title
    doc.core_properties.author = author
    doc.core_properties.created = datetime.now()
    
    # Set default font for Chinese
    style = doc.styles['Normal']
    style.font.name = 'Microsoft YaHei'
    style.font.size = Pt(11)
    
    # Add title if not already in markdown
    if not md_content.strip().startswith('#'):
        heading = doc.add_heading(title, 0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Parse markdown
    parse_markdown_to_docx(doc, md_content)
    
    # Save
    doc.save(str(output_path))
    
    print(f"✅ DOCX created: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF via DOCX")
    parser.add_argument("input", help="Input markdown file")
    parser.add_argument("output", help="Output PDF file")
    parser.add_argument("--title", help="Document title")
    parser.add_argument("--author", help="Document author")
    parser.add_argument("--page-size", default="A4", choices=["A4", "Letter", "Legal", "A3"])
    
    args = parser.parse_args()
    convert_md_to_pdf(args.input, args.output, args.title, args.author, args.page_size)