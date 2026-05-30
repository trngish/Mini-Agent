#!/usr/bin/env python3
"""解包 Office 文件（.docx、.pptx、.xlsx）的 XML 内容并格式化。"""

import random
import sys
import zipfile
from pathlib import Path

import defusedxml.minidom

# 获取命令行参数
if len(sys.argv) != 3:
    print(f"用法：python unpack.py <office_file> <output_dir>", file=sys.stderr)
    sys.exit(1)
input_file, output_dir = sys.argv[1], sys.argv[2]

# 解包并格式化
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(input_file, "r") as zf:
    zf.extractall(output_path)

# 美化打印所有 XML 文件
xml_files = list(output_path.rglob("*.xml")) + list(output_path.rglob("*.rels"))
for xml_file in xml_files:
    content = xml_file.read_text(encoding="utf-8")
    dom = defusedxml.minidom.parseString(content)
    xml_file.write_bytes(dom.toprettyxml(indent="  ", encoding="utf-8"))

# 对于 .docx 文件，建议使用 RSID 用于修订跟踪
if input_file.endswith(".docx"):
    suggested_rsid = "".join(random.choices("0123456789ABCDEF", k=8))
    print(f"建议用于编辑会话的 RSID：{suggested_rsid}")