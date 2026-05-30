"""
技能加载器 - 加载Claude技能

支持从SKILL.md文件加载技能并提供给Agent
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    """技能数据结构"""

    name: str
    description: str
    content: str
    license: str | None = None
    allowed_tools: list[str] | None = None
    metadata: dict[str, str] | None = None
    skill_path: Path | None = None

    def to_prompt(self) -> str:
        """将技能转换为提示格式"""
        # 注入技能根目录路径以提供上下文
        skill_root = str(self.skill_path.parent) if self.skill_path else "unknown"

        return f"""
# Skill: {self.name}

{self.description}

**Skill Root Directory:** `{skill_root}`

All files and references in this skill are relative to this directory.

---

{self.content}
"""


class SkillLoader:
    """技能加载器"""

    def __init__(self, skills_dir: str = "./skills"):
        """
        初始化技能加载器

        参数:
            skills_dir: 技能目录路径
        """
        self.skills_dir = Path(skills_dir)
        self.loaded_skills: dict[str, Skill] = {}
        self._discovered_paths: list[Path] | None = None  # 已发现技能路径的缓存

    def load_skill(self, skill_path: Path) -> Skill | None:
        """
        从SKILL.md文件加载单个技能

        参数:
            skill_path: SKILL.md文件路径

        返回:
            Skill对象，如果加载失败则返回None
        """
        try:
            content = skill_path.read_text(encoding="utf-8")

            # 解析YAML前置matter
            frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)

            if not frontmatter_match:
                print(f"⚠️  {skill_path} 缺少 YAML frontmatter")
                return None

            frontmatter_text = frontmatter_match.group(1)
            skill_content = frontmatter_match.group(2).strip()

            # 解析YAML
            try:
                frontmatter = yaml.safe_load(frontmatter_text)
            except yaml.YAMLError as e:
                print(f"❌ 解析 YAML frontmatter 失败: {e}")
                return None

            # 必填字段
            if "name" not in frontmatter or "description" not in frontmatter:
                print(f"⚠️  {skill_path} 缺少必需字段 (name 或 description)")
                return None

            # 获取技能目录（SKILL.md的父目录）
            skill_dir = skill_path.parent

            # 将内容中的相对路径替换为绝对路径
            # 这确保了脚本和资源可以从任何工作目录被找到
            processed_content = self._process_skill_paths(skill_content, skill_dir)

            # 创建Skill对象
            skill = Skill(
                name=frontmatter["name"],
                description=frontmatter["description"],
                content=processed_content,
                license=frontmatter.get("license"),
                allowed_tools=frontmatter.get("allowed-tools"),
                metadata=frontmatter.get("metadata"),
                skill_path=skill_path,
            )

            return skill

        except Exception as e:
            print(f"❌ 加载技能失败 ({skill_path}): {e}")
            return None

    def _process_skill_paths(self, content: str, skill_dir: Path) -> str:
        """
        处理技能内容，将相对路径替换为绝对路径。

        支持渐进式披露Level 3+：将相对文件引用转换为绝对路径，
        以便Agent能够轻松读取嵌套资源。

        参数:
            content: 原始技能内容
            skill_dir: 技能目录路径

        返回:
            带有绝对路径的处理后内容
        """

        # 模式1: 基于目录的路径（scripts/, references/, assets/）
        # 参见 https://agentskills.io/specification#optional-directories
        def replace_dir_path(match: re.Match[str]) -> str:
            prefix = match.group(1)  # 例如："python " 或 "`"
            rel_path = match.group(2)  # 例如："scripts/with_server.py"

            abs_path = skill_dir / rel_path
            if abs_path.exists():
                return f"{prefix}{abs_path}"
            return match.group(0)

        pattern_dirs = r"(python\s+|`)((?:scripts|references|assets)/[^\s`\)]+)"
        content = re.sub(pattern_dirs, replace_dir_path, content)

        # 模式2: 直接markdown/文档引用（forms.md, reference.md等）
        # 匹配类似 "see reference.md" 或 "read forms.md" 的短语
        def replace_doc_path(match: re.Match[str]) -> str:
            prefix = match.group(1)  # 例如："see ", "read "
            filename = match.group(2)  # 例如："reference.md"
            suffix = match.group(3)  # 例如：标点符号

            abs_path = skill_dir / filename
            if abs_path.exists():
                # 为Agent添加有用的说明
                return f"{prefix}`{abs_path}` (use read_file to access){suffix}"
            return match.group(0)

        # 匹配类似: "see reference.md" 或 "read forms.md" 的模式
        pattern_docs = r"(see|read|refer to|check)\s+([a-zA-Z0-9_-]+\.(?:md|txt|json|yaml))([.,;\s])"
        content = re.sub(pattern_docs, replace_doc_path, content, flags=re.IGNORECASE)

        # 模式3: Markdown链接 - 支持多种格式:
        # - [`filename.md`](filename.md) - 简单文件名
        # - [text](./reference/file.md) - 带./的相对路径
        # - [text](scripts/file.js) - 基于目录的路径
        # 匹配类似: "Read [`docx-js.md`](docx-js.md)" 或 "Load [Guide](./reference/guide.md)" 的模式
        def replace_markdown_link(match: re.Match[str]) -> str:
            prefix = match.group(1) if match.group(1) else ""  # 例如："Read ", "Load ", 或空
            link_text = match.group(2)  # 例如："`docx-js.md`" 或 "Guide"
            filepath = match.group(3)  # 例如："docx-js.md", "./reference/file.md", "scripts/file.js"

            # 如果有前导./则移除
            clean_path = filepath[2:] if filepath.startswith("./") else filepath

            abs_path = skill_dir / clean_path
            if abs_path.exists():
                # 保留链接文本样式（带或不带反引号）
                return f"{prefix}[{link_text}](`{abs_path}`) (use read_file to access)"
            return match.group(0)

        # 匹配带可选前缀词的markdown链接模式
        # 捕获: (可选前缀词) [链接文本] (完整文件路径包括./)
        pattern_markdown = (
            r"(?:(Read|See|Check|Refer to|Load|View)\s+)?\[(`?[^`\]]+`?)\]"
            r"\(((?:\./)?[^)]+\.(?:md|txt|json|yaml|js|py|html))\)"
        )
        content = re.sub(pattern_markdown, replace_markdown_link, content, flags=re.IGNORECASE)

        return content

    def discover_skills(self, additional_search_paths: list[Path] | None = None) -> list[Skill]:
        """ "
        发现并从多个目录加载所有技能。

        搜索顺序（同名时先找到的优先）：
        1. 额外的搜索路径（例如用户配置 ~/.mini-agent/skills/）
        2. 默认的skills_dir路径

        参数:
            additional_search_paths: 搜索技能的额外目录


        返回:
            技能列表
        """
        # 如果已经发现过则返回缓存的结果
        if self._discovered_paths is not None:
            return [self.loaded_skills[skill.name] for skill in self.loaded_skills.values()]

        skills = []
        skill_files_by_name: dict[str, Path] = {}  # 按技能名跟踪第一个找到的SKILL.md

        def collect_skill_files(search_dir: Path) -> None:
            """从目录中收集SKILL.md文件到skill_files_by_name。"""
            if not search_dir.exists():
                return
            for skill_file in search_dir.rglob("SKILL.md"):
                skill_name = skill_file.parent.name
                if skill_name not in skill_files_by_name:
                    skill_files_by_name[skill_name] = skill_file

        # 优先级1: 额外的搜索路径（用户配置目录）
        if additional_search_paths:
            for search_path in additional_search_paths:
                collect_skill_files(search_path)

        # 优先级2: 默认的skills_dir
        if self.skills_dir.exists():
            collect_skill_files(self.skills_dir)

        # 加载所有发现的技能
        self._discovered_paths = list(skill_files_by_name.values())

        for skill_file in self._discovered_paths:
            skill = self.load_skill(skill_file)
            if skill:
                skills.append(skill)
                self.loaded_skills[skill.name] = skill

        return skills

    def get_skill(self, name: str) -> Skill | None:
        """
        获取已加载的技能

        参数:
            name: 技能名称

        返回:
            Skill对象，如果未找到则返回None
        """
        return self.loaded_skills.get(name)

    def list_skills(self) -> list[str]:
        """
        列出所有已加载的技能名称

        返回:
            技能名称列表
        """
        return list(self.loaded_skills.keys())

    def get_skills_metadata_prompt(self) -> str:
        """
        生成仅包含所有技能元数据（名称+描述）的提示。
        这实现了渐进式披露 - Level 1。

        返回:
            仅包含元数据的提示字符串
        """
        if not self.loaded_skills:
            return ""

        prompt_parts = ["## Available Skills\n"]
        prompt_parts.append(
            "你可以访问专业技能。每个技能都为特定任务提供专家指导。\n"
        )
        prompt_parts.append("需要时使用相应的技能工具加载技能的完整内容。\n")

        # List all skills with their descriptions
        for skill in self.loaded_skills.values():
            prompt_parts.append(f"- `{skill.name}`: {skill.description}")

        return "\n".join(prompt_parts)
