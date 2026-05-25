"""Team dispatch tool for multi-agent collaboration.

Provides a tool that allows the Agent to spawn a team of specialized agents
for complex task decomposition and adversarial review/critique.
"""

import asyncio
from typing import Any, Optional

from .base import Tool, ToolResult
from ..llm import LLMClient
from ..tools.base import Tool
from ..team.agent_team import AgentTeam, AgentRole


DEFAULT_TIMEOUT = 300  # 5 minutes default timeout


class TeamDispatchTool(Tool):
    """Dispatch a team of agents for complex tasks.

    This tool enables the main agent to spawn a team with:
    - Task decomposition: Break complex tasks into parallel subtasks
    - Adversarial review: Multiple critics challenge the solution
    - Result synthesis: Combine results into final output

    Use cases:
    - Complex tasks that can be broken into independent subtasks
    - Code review with multiple critics
    - Design evaluation with adversarial reasoning
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[Tool],
        system_prompt: str = "You are a helpful AI assistant on a software engineering team.",
        m27_config: Optional[dict] = None,
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
        return """Dispatch a team of specialized agents to work on complex tasks.

Supports two modes:
1. **decompose**: Break a complex task into parallel subtasks executed by multiple agents
2. **review**: Run adversarial review with multiple critics challenging the solution

Parameters:
  - task: The main task or problem to solve
  - mode: "decompose" for task splitting, "review" for adversarial critique
  - roles: Which roles to include (defaults depend on mode)

For decompose mode, roles typically include: planner, executor
For review mode, roles typically include: reviewer, critic (multiple)

Examples:
  - Decompose: team_dispatch(task="Build a full web app", mode="decompose")
  - Review: team_dispatch(task="Review this design", mode="review")
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to execute or have the team work on",
                },
                "mode": {
                    "type": "string",
                    "enum": ["decompose", "review"],
                    "description": "Execution mode: 'decompose' for parallel execution, 'review' for adversarial critique",
                    "default": "decompose",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Maximum number of critique rounds for review mode",
                    "default": 2,
                },
                "timeout_per_agent": {
                    "type": "integer",
                    "description": "Timeout per agent in seconds",
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
        """Execute team dispatch.

        Args:
            task: The task to execute
            mode: "decompose" or "review"
            max_rounds: Number of critique rounds (for review mode)
            timeout_per_agent: Timeout per agent in seconds

        Returns:
            ToolResult with team execution results
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

            # Use a reasonable overall timeout
            overall_timeout = min(timeout_per_agent * 3, DEFAULT_TIMEOUT)

            async with asyncio.timeout(overall_timeout):
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
        """Execute in decompose mode: planner + executor + reviewer."""
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
        max_rounds: int,
    ) -> ToolResult:
        """Execute in review mode: reviewer + multiple critics."""
        team.add_reviewer("reviewer", max_steps=30)
        team.add_critic("critic1", max_steps=30)
        team.add_critic("critic2", max_steps=30)

        result = await team.execute(task, timeout=timeout)

        critique_summary = []
        for i, (name, content) in enumerate(result.agent_results.items(), 1):
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