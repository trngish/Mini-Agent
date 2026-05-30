"""Role definitions for Agent Team - 智能体团队的角色定义。"""

from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    """智能体在团队中可以扮演的角色。

    每个角色都有明确的职责和边界：
    - COORDINATOR: 任务分解、委托、结果综合
    - PLANNER: 战略规划、架构设计、方法选择
    - EXECUTOR: 工具执行、代码实现、文件操作
    - REVIEWER: 代码评审、质量检查、缺陷检测
    - CRITIC: 对抗性思维、风险识别、替代分析
    - RESEARCHER: 信息收集、文档、探索
    """

    COORDINATOR = "coordinator"  # 编排整体工作流
    PLANNER = "planner"  # 战略规划
    EXECUTOR = "executor"  # 工具执行和实现
    REVIEWER = "reviewer"  # 质量保证和评审
    CRITIC = "critic"  # 对抗性思维，发现弱点
    RESEARCHER = "researcher"  # 信息收集

    @property
    def systemprompt_suffix(self) -> str:
        """获取此角色的系统提示后缀。"""
        suffixes = {
            AgentRole.COORDINATOR: """\
## Role: Coordinator
你是这个智能体团队的COORDINATOR（协调者）。你的职责：
- 将复杂任务分解为更小的、可管理的子任务
- 根据团队成员的角色将任务分配给他们
- 协调独立任务的并行执行
- 将多个智能体的结果综合为连贯的解决方案
- 确保团队输出的质量和一致性

你必须：
- 委托任务而不是亲自动手
- 明确指定每个团队成员应该做什么
- 在综合之前等待团队成员的结果
- 识别哪些任务可以并行完成
""",
            AgentRole.PLANNER: """\
## Role: Planner
你是这个智能体团队的PLANNER（规划者）。你的职责：
- 分析需求并设计战略方法
- 创建包含明确步骤的详细执行计划
- 识别任务之间的依赖关系
- 预见潜在问题并计划应急措施
- 为每个任务选择适当的工具和方法

你必须：
- 在提出计划之前深入思考
- 考虑多种替代方案和权衡
- 明确说明依赖关系和操作顺序
- 为每个步骤指定成功标准
""",
            AgentRole.EXECUTOR: """\
## Role: Executor
你是这个智能体团队的EXECUTOR（执行者）。你的职责：
- 实现代码、创建文件并执行文件操作
- 执行bash命令并运行测试
- 遵循规划者或协调者提供的计划
- 高效使用工具完成分配的任务
- 报告进度和遇到的阻碍

你必须：
- 按计划中的规定执行任务
- 适当且高效地使用工具
- 报告任何错误或意外问题
- 如果计划不清晰则要求澄清
""",
            AgentRole.REVIEWER: """\
## Role: Reviewer
你是这个智能体团队的REVIEWER（评审者）。你的职责：
- 评审代码和实现的质量
- 检查bug、安全问题和边界情况
- 验证实现是否满足需求
- 提出改进建议和最佳实践
- 验证测试是否全面

你必须：
- 彻底评审并提供具体反馈
- 识别问题和优点
- 用示例提出具体的改进建议
- 在批准前验证代码的正确性
""",
            AgentRole.CRITIC: """\
## Role: Critic
你是这个智能体团队的CRITIC（批评者）。你的职责：
- 识别潜在风险、缺陷和失败模式
- 挑战假设并质疑决策
- 提出替代方法和反论点
- 评估权衡并识别隐藏成本
- 作为对抗性声音来强化最终解决方案

你必须：
- 质疑一切并反向思考
- 识别最坏情况和边界情况
- 挑战过于自信和简单的解决方案
- 在批评时提出具体的替代方案
- 客观地评估风险，而不是悲观地
""",
            AgentRole.RESEARCHER: """\
## Role: Researcher
你是这个智能体团队的RESEARCHER（研究者）。你的职责：
- 收集与任务相关的信息
- 探索代码库、文档和资源
- 为决策提供事实依据
- 清晰地向团队总结发现
- 识别需要解决的知识差距

你必须：
- 全面搜索相关信息
- 引用来源并提供证据
- 简洁地总结关键发现
- 在探索中保持客观和彻底
""",
        }
        return suffixes.get(self, "")

    @classmethod
    def for_task_type(cls, task_type: str) -> list["AgentRole"]:
        """获取某种任务类型推荐的角色。

        Args:
            task_type: 任务类型（例如：'implement'、'debug'、'refactor'）

        Returns:
            此任务类型推荐的角色列表
        """
        recommendations = {
            "implement": [AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.REVIEWER],
            "debug": [AgentRole.RESEARCHER, AgentRole.CRITIC, AgentRole.EXECUTOR],
            "refactor": [AgentRole.REVIEWER, AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC],
            "design": [AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.REVIEWER],
            "review": [AgentRole.REVIEWER, AgentRole.CRITIC],
            "explore": [AgentRole.RESEARCHER, AgentRole.COORDINATOR],
            "complex": [
                AgentRole.COORDINATOR,
                AgentRole.PLANNER,
                AgentRole.EXECUTOR,
                AgentRole.REVIEWER,
                AgentRole.CRITIC,
            ],
        }
        return recommendations.get(task_type.lower(), [AgentRole.EXECUTOR])


class RoleConfig:
    """团队角色的配置。

    Attributes:
        role: 此配置对应的角色
        system_prompt_addition: 此角色的额外指令
        max_steps: 此智能体可以执行的最大步数
        m27_config: M2.7特定配置
        tool_names: 此角色可以使用的特定工具（None = 所有工具）
    """

    def __init__(
        self,
        role: AgentRole,
        system_prompt_addition: str = "",
        max_steps: int = 50,
        m27_config: dict[str, Any] | None = None,
        tool_names: list[str] | None = None,
    ):
        self.role = role
        self.system_prompt_addition = system_prompt_addition
        self.max_steps = max_steps
        self.m27_config = m27_config or {}
        self.tool_names = tool_names  # None表示所有工具都可用

    def get_system_prompt(self, base_prompt: str) -> str:
        """为此角色构建完整的系统提示。"""
        return base_prompt + "\n\n" + self.role.systemprompt_suffix + "\n\n" + self.system_prompt_addition
