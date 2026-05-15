"""Sub-agent for concurrent background task execution."""

import asyncio
from time import perf_counter
from typing import Optional

from .llm import LLMClient
from .schema import LLMResponse, Message
from .tools.base import Tool, ToolResult


class SubAgentResult:
    """Result from a sub-agent execution."""

    def __init__(self, task: str, content: str, success: bool, elapsed: float, error: str = ""):
        self.task = task
        self.content = content
        self.success = success
        self.elapsed = elapsed
        self.error = error


class SubAgent:
    """Standalone agent for background sub-tasks.

    Has its own LLM client and tool set, runs independently from the parent agent.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[Tool],
        system_prompt: str = "You are a helpful assistant. Complete the assigned task concisely.",
        max_steps: int = 10,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_list = list(tools)
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    async def run(self, task: str) -> SubAgentResult:
        """Execute a sub-task and return the result."""
        start = perf_counter()
        messages = [Message(role="system", content=self.system_prompt), Message(role="user", content=task)]

        for step in range(self.max_steps):
            try:
                response = await self.llm.generate(messages=messages, tools=self.tool_list)
            except Exception as e:
                return SubAgentResult(
                    task=task, content="", success=False,
                    elapsed=perf_counter() - start, error=str(e),
                )

            if response.tool_calls:
                for tc in response.tool_calls:
                    tool = self.tools.get(tc.function.name)
                    if tool:
                        try:
                            result = await tool.execute(**tc.function.arguments)
                        except Exception as e:
                            result = ToolResult(success=False, content="", error=str(e))
                        messages.append(Message(
                            role="tool", content=result.content if result.success else f"Error: {result.error}",
                            tool_call_id=tc.id, name=tc.function.name,
                        ))
                continue

            return SubAgentResult(
                task=task, content=response.content or "", success=True,
                elapsed=perf_counter() - start,
            )

        return SubAgentResult(
            task=task, content="", success=False,
            elapsed=perf_counter() - start, error="Max steps reached",
        )


async def run_sub_agents(
    llm_client: LLMClient,
    tasks: list[str],
    tools: list[Tool],
    max_concurrent: int = 3,
) -> list[SubAgentResult]:
    """Run multiple sub-agents concurrently.

    Args:
        llm_client: LLM client shared by all sub-agents
        tasks: List of task descriptions
        tools: Available tools
        max_concurrent: Maximum concurrent sub-agents

    Returns:
        List of SubAgentResult in the same order as tasks
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(task: str) -> SubAgentResult:
        async with semaphore:
            agent = SubAgent(llm_client=llm_client, tools=tools)
            return await agent.run(task)

    results = await asyncio.gather(*[run_one(t) for t in tasks])
    return results
