"""
技能工具 - 用于按需加载技能的工具

实现渐进式披露（Level 2）：在需要时加载完整技能内容
"""

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from .skill_loader import SkillLoader


class GetSkillTool(Tool):
    """用于获取指定技能详细信息"""

    def __init__(self, skill_loader: SkillLoader):
        self.skill_loader = skill_loader

    @property
    def name(self) -> str:
        return "get_skill"

    @property
    def description(self) -> str:
        return "获取指定技能的完整内容和指导，用于执行特定类型的任务"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "要获取的技能名称（使用 list_skills 查看可用技能）",
                }
            },
            "required": ["skill_name"],
        }

    async def execute(self, skill_name: str) -> ToolResult:
        """获取指定技能的详细信息"""
        skill = self.skill_loader.get_skill(skill_name)

        if not skill:
            available = ", ".join(self.skill_loader.list_skills())
            return ToolResult(
                success=False,
                content="",
                error=f"技能 '{skill_name}' 不存在。可用技能: {available}",
            )

        # 返回完整的技能内容
        result = skill.to_prompt()
        return ToolResult(success=True, content=result)


def create_skill_tools(
    skills_dir: str = "./skills",
    additional_search_paths: list[Path] | None = None,
) -> tuple[list[GetSkillTool], SkillLoader]:
    """
    创建用于渐进式披露的技能工具

    仅提供 get_skill 工具 - 智能体使用系统提示中的元数据
    来了解有哪些技能可用，然后在需要时按需加载。

    Args:
        skills_dir: 技能目录路径
        additional_search_paths: 搜索技能的其他目录（如用户配置）

    Returns:
        元组：(工具列表, 技能加载器)
    """
    # 创建技能加载器
    loader = SkillLoader(skills_dir)

    # 从多个目录发现并加载技能
    skills = loader.discover_skills(additional_search_paths=additional_search_paths)
    print(f"✅ 已发现 {len(skills)} 个 Claude 技能")

    # 仅创建 get_skill 工具（渐进式披露 Level 2）
    tools = [
        GetSkillTool(loader),
    ]

    return tools, loader
