"""
Word 文档中追踪修订的验证器。
"""

import subprocess
import tempfile
import zipfile
from pathlib import Path


class RedliningValidator:
    """Word 文档中追踪修订的验证器。"""

    def __init__(self, unpacked_dir, original_docx, verbose=False):
        self.unpacked_dir = Path(unpacked_dir)
        self.original_docx = Path(original_docx)
        self.verbose = verbose
        self.namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    def validate(self):
        """主验证方法，返回 True 表示验证通过，False 表示验证失败。"""
        # 验证解压目录存在且结构正确
        modified_file = self.unpacked_dir / "word" / "document.xml"
        if not modified_file.exists():
            print(f"FAILED - Modified document.xml not found at {modified_file}")
            return False

        # 首先，检查是否存在 Claude 创作的追踪修订需要验证
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(modified_file)
            root = tree.getroot()

            # 检查 w:del 或 w:ins 标签中是否有 Claude 创作的内容
            del_elements = root.findall(".//w:del", self.namespaces)
            ins_elements = root.findall(".//w:ins", self.namespaces)

            # 筛选出仅包含 Claude 创作的变化
            claude_del_elements = [
                elem for elem in del_elements if elem.get(f"{{{self.namespaces['w']}}}author") == "Claude"
            ]
            claude_ins_elements = [
                elem for elem in ins_elements if elem.get(f"{{{self.namespaces['w']}}}author") == "Claude"
            ]

            # 只有在使用了 Claude 创作的追踪修订时才需要进行修订验证。
            if not claude_del_elements and not claude_ins_elements:
                if self.verbose:
                    print("PASSED - No tracked changes by Claude found.")
                return True

        except Exception:
            # 如果无法解析 XML，则继续进行完整验证
            pass

        # 创建临时目录用于解压原始 docx
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 解压原始 docx
            try:
                with zipfile.ZipFile(self.original_docx, "r") as zip_ref:
                    zip_ref.extractall(temp_path)
            except Exception as e:
                print(f"FAILED - Error unpacking original docx: {e}")
                return False

            original_file = temp_path / "word" / "document.xml"
            if not original_file.exists():
                print(f"FAILED - Original document.xml not found in {self.original_docx}")
                return False

            # 使用 xml.etree.ElementTree 解析两个 XML 文件进行修订验证
            try:
                import xml.etree.ElementTree as ET

                modified_tree = ET.parse(modified_file)
                modified_root = modified_tree.getroot()
                original_tree = ET.parse(original_file)
                original_root = original_tree.getroot()
            except ET.ParseError as e:
                print(f"FAILED - Error parsing XML files: {e}")
                return False

            # 从两个文档中移除 Claude 的追踪修订
            self._remove_claude_tracked_changes(original_root)
            self._remove_claude_tracked_changes(modified_root)

            # 提取并比较文本内容
            modified_text = self._extract_text_content(modified_root)
            original_text = self._extract_text_content(original_root)

            if modified_text != original_text:
                # 显示每个段落详细的字符级差异
                error_message = self._generate_detailed_diff(original_text, modified_text)
                print(error_message)
                return False

            if self.verbose:
                print("PASSED - All changes by Claude are properly tracked")
            return True

    def _generate_detailed_diff(self, original_text, modified_text):
        """使用 git word diff 生成详细的单词级差异。"""
        error_parts = [
            "FAILED - Document text doesn't match after removing Claude's tracked changes",
            "",
            "Likely causes:",
            "  1. Modified text inside another author's <w:ins> or <w:del> tags",
            "  2. Made edits without proper tracked changes",
            "  3. Didn't nest <w:del> inside <w:ins> when deleting another's insertion",
            "",
            "For pre-redlined documents, use correct patterns:",
            "  - To reject another's INSERTION: Nest <w:del> inside their <w:ins>",
            "  - To restore another's DELETION: Add new <w:ins> AFTER their <w:del>",
            "",
        ]

        # 显示 git word diff
        git_diff = self._get_git_word_diff(original_text, modified_text)
        if git_diff:
            error_parts.extend(["Differences:", "============", git_diff])
        else:
            error_parts.append("Unable to generate word diff (git not available)")

        return "\n".join(error_parts)

    def _get_git_word_diff(self, original_text, modified_text):
        """使用 git 生成具有字符级精度的单词差异。"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # 创建两个文件
                original_file = temp_path / "original.txt"
                modified_file = temp_path / "modified.txt"

                original_file.write_text(original_text, encoding="utf-8")
                modified_file.write_text(modified_text, encoding="utf-8")

                # 首先尝试字符级差异以获得精确的差异
                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--word-diff=plain",
                        "--word-diff-regex=.",  # 逐字符差异
                        "-U0",  # 零行上下文 - 仅显示更改的行
                        "--no-index",
                        str(original_file),
                        str(modified_file),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                if result.stdout.strip():
                    # 清理输出 - 移除 git diff 头部行
                    lines = result.stdout.split("\n")
                    # 跳过头部行（diff --git, index, +++, ---, @@）
                    content_lines = []
                    in_content = False
                    for line in lines:
                        if line.startswith("@@"):
                            in_content = True
                            continue
                        if in_content and line.strip():
                            content_lines.append(line)

                    if content_lines:
                        return "\n".join(content_lines)

                # 如果字符级差异过于详细，则回退到单词级差异
                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--word-diff=plain",
                        "-U0",  # 零行上下文
                        "--no-index",
                        str(original_file),
                        str(modified_file),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                if result.stdout.strip():
                    lines = result.stdout.split("\n")
                    content_lines = []
                    in_content = False
                    for line in lines:
                        if line.startswith("@@"):
                            in_content = True
                            continue
                        if in_content and line.strip():
                            content_lines.append(line)
                    return "\n".join(content_lines)

        except (subprocess.CalledProcessError, FileNotFoundError, Exception):
            # Git 不可用或其他错误，返回 None 使用备用方案
            pass

        return None

    def _remove_claude_tracked_changes(self, root):
        """从 XML 根节点中移除 Claude 创作的追踪修订。"""
        ins_tag = f"{{{self.namespaces['w']}}}ins"
        del_tag = f"{{{self.namespaces['w']}}}del"
        author_attr = f"{{{self.namespaces['w']}}}author"

        # 移除 w:ins 元素
        for parent in root.iter():
            to_remove = []
            for child in parent:
                if child.tag == ins_tag and child.get(author_attr) == "Claude":
                    to_remove.append(child)
            for elem in to_remove:
                parent.remove(elem)

        # 解包作者为 "Claude" 的 w:del 元素中的内容
        deltext_tag = f"{{{self.namespaces['w']}}}delText"
        t_tag = f"{{{self.namespaces['w']}}}t"

        for parent in root.iter():
            to_process = []
            for child in parent:
                if child.tag == del_tag and child.get(author_attr) == "Claude":
                    to_process.append((child, list(parent).index(child)))

            # 按反向顺序处理以维持索引
            for del_elem, del_index in reversed(to_process):
                # 在移动之前将 w:delText 转换为 w:t
                for elem in del_elem.iter():
                    if elem.tag == deltext_tag:
                        elem.tag = t_tag

                # 在移除 w:del 之前，将其所有子元素移动到其父元素
                for child in reversed(list(del_elem)):
                    parent.insert(del_index, child)
                parent.remove(del_elem)

    def _extract_text_content(self, root):
        """从 Word XML 中提取文本内容，保持段落结构。

        跳过空段落以避免在追踪插入仅添加结构元素而没有文本内容时产生误报。
        """
        p_tag = f"{{{self.namespaces['w']}}}p"
        t_tag = f"{{{self.namespaces['w']}}}t"

        paragraphs = []
        for p_elem in root.findall(f".//{p_tag}"):
            # 获取此段落中的所有文本元素
            text_parts = []
            for t_elem in p_elem.findall(f".//{t_tag}"):
                if t_elem.text:
                    text_parts.append(t_elem.text)
            paragraph_text = "".join(text_parts)
            # 跳过空段落 - 它们不影响内容验证
            if paragraph_text:
                paragraphs.append(paragraph_text)

        return "\n".join(paragraphs)


if __name__ == "__main__":
    raise RuntimeError("此模块不应直接运行。")
