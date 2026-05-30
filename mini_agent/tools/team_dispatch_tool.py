"""用于多智能体协作的团队调度工具。

提供允许Agent生成分工明确的专门智能体团队的工具，
用于复杂任务分解和对抗性审查/批评。
"""

import asyncio
from typing import Any

from ..llm import LLMClient
from ..team.agent_team import AgentTeam
from .base import Tool, ToolResult

DEFAULT_TIMEOUT = 300  # 5 minutes default timeout


class TeamDispatchTool(Tool):
    """为复杂任务调度智能体团队。

    此工具使主智能体能够生成了一个具备以下能力的团队：
    - 任务分解：将复杂任务分解为并行子任务
    - 对抗性审查：多个批评者挑战解决方案
    - 结果综合：将结果合成为最终输出

    使用场景：
    - 可分解为独立子任务的复杂任务
    - 多批评者的代码审查
    - 对抗性推理的设计评估
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[Tool],
        system_prompt: str = "You are a helpful AI assistant on a software engineering team.",
        m27_config: dict[str, Any] | None = None,
    ):
        self._llm_client = llm_client
        self._tools = tools
        self._system_prompt = system_prompt
        self._m27_config = m27_config or {}

    @property
    def name(self) -> str:
        return "team_dispatch"

    @property
    def description(self) -> str:
        return """调度专门化智能体团队来处理复杂任务。

支持两种模式：
1. **decompose**: 将复杂任务分解为由多个智能体执行的并行子任务
2. **review**: 运行对抗性审查，多个批评者挑战解决方案

参数:
  - task: 要执行的主要任务或问题
  - mode: "decompose"用于任务拆分，"review"用于对抗性批评
  - roles: 包含的角色（默认值取决于模式）

对于decompose模式，角色通常包括：planner, executor
对于review模式，角色通常包括：reviewer, critic（多个）

示例:
  - 分解: team_dispatch(task="构建一个完整的Web应用", mode="decompose")
  - 审查: team_dispatch(task="审查这个设计", mode="review")
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "要执行的任务或让团队处理的任务",
                },
                "mode": {
                    "type": "string",
                    "enum": ["decompose", "review"],
                    "description": (
                        "执行模式：'decompose'用于并行执行，'review'用于对抗性批评"
                    ),
                    "default": "decompose",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "审查模式的最多批评轮数",
                    "default": 2,
                },
                "timeout_per_agent": {
                    "type": "integer",
                    "description": "每个智能体的超时时间（秒）",
                    "default": 120,
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        mode: str = "decompose",
        max_rounds: int = 2,
        timeout_per_agent: int = 120,
    ) -> ToolResult:
        """执行团队调度。

        参数:
            task: 要执行的任务
            mode: "decompose" 或 "review"
            max_rounds: 批评轮数（用于review模式）
            timeout_per_agent: 每个智能体的超时时间（秒）

        返回:
            包含团队执行结果的ToolResult
        """
        try:
            team = AgentTeam(
                llm_client=self._llm_client,
                tools=self._tools,
                system_prompt=self._system_prompt,
                max_concurrent=3,
                enable_adversarial=True,
                critique_rounds=max_rounds,
                m27_config=self._m27_config,
            )

            # 使用合理的总体超时时间
            overall_timeout = min(timeout_per_agent * 3, DEFAULT_TIMEOUT)

            async with asyncio.timeout(overall_timeout):  # type: ignore[attr-defined]
                if mode == "decompose":
                    return await self._execute_decompose(team, task, timeout_per_agent)
                elif mode == "review":
                    return await self._execute_review(team, task, timeout_per_agent, max_rounds)
                else:
                    return ToolResult(
                        success=False,
                        error=f"Unknown mode: {mode}. Use 'decompose' or 'review'.",
                    )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content="",
                error=f"Team execution timed out after {overall_timeout}s",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Team dispatch failed: {str(e)}",
            )

    async def _execute_decompose(
        self,
        team: AgentTeam,
        task: str,
        timeout: int,
    ) -> ToolResult:
        """以decompose模式执行：planner + executor + reviewer。"""
        team.add_planner("planner", max_steps=40)
        team.add_executor("executor", max_steps=50)
        team.add_reviewer("reviewer", max_steps=30)

        result = await team.execute(task, timeout=timeout)

        if result.success:
            return ToolResult(
                success=True,
                content=f"✅ Task completed successfully\n\n{result.content}",
            )
        else:
            return ToolResult(
                success=False,
                content=result.content,
                error=result.error or "Task failed",
            )

    async def _execute_review(
        self,
        team: AgentTeam,
        task: str,
        timeout: int,
        max_rounds: int,  # noqa: ARG002
    ) -> ToolResult:
        """以review模式执行：reviewer + 多个critics。"""
        team.add_reviewer("reviewer", max_steps=30)
        team.add_critic("critic1", max_steps=30)
        team.add_critic("critic2", max_steps=30)

        result = await team.execute(task, timeout=timeout)

        critique_summary = []
        for _i, (name, content) in enumerate(result.agent_results.items(), 1):
            critique_summary.append(f"### {name}\n\n{content[:500]}")

        combined = "\n\n".join(critique_summary)
        if result.success:
            return ToolResult(
                success=True,
                content=f"📋 Review Summary\n\n{combined}\n\n---\n\n**Final Synthesis**\n\n{result.content}",
            )
        else:
            return ToolResult(
                success=False,
                content=combined,
                error=result.error or "Review failed",
            )
