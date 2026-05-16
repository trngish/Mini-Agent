"""Sub-agent for concurrent background task execution."""

import asyncio
from time import perf_counter
from typing import Optional

from .llm import LLMClient
from .schema import AgentMode, Message
from .schema.schema import WRITE_TOOLS
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
    Supports M2.7 parallel tool execution when enabled.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[Tool],
        system_prompt: str = "You are a helpful assistant. Complete the assigned task concisely.",
        max_steps: int = 50,
        m27_config: Optional[dict] = None,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_list = list(tools)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.m27_config = m27_config or {}
        self.is_m27 = False  # SubAgent doesn't need M2.7 specifics
        self.mode = AgentMode.YOLO  # SubAgent always runs in YOLO mode
        self.write_tools = WRITE_TOOLS

    async def run(self, task: str) -> SubAgentResult:
        """Execute a sub-task and return the result."""
        start = perf_counter()
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task)
        ]

        for step in range(self.max_steps):
            try:
                response = await self.llm.generate(messages=messages, tools=self.tool_list)
            except Exception as e:
                return SubAgentResult(
                    task=task, content="", success=False,
                    elapsed=perf_counter() - start, error=str(e),
                )

            if response.tool_calls:
                # M2.7: execute tools in parallel if enabled and multiple tools
                parallel_enabled = self.m27_config.get("enable_parallel_tool_calls", True)
                max_concurrent = self.m27_config.get("max_concurrent_tools", 10)

                if parallel_enabled and len(response.tool_calls) > 1:
                    results = await self._execute_tools_parallel(response.tool_calls, max_concurrent)
                else:
                    results = await self._execute_tools_sequential(response.tool_calls)

                # Add tool messages in order
                for _, tool_msg in results:
                    messages.append(tool_msg)
                continue

            return SubAgentResult(
                task=task, content=response.content or "", success=True,
                elapsed=perf_counter() - start,
            )

        return SubAgentResult(
            task=task, content="", success=False,
            elapsed=perf_counter() - start, error="Max steps reached",
        )

    async def _execute_tools_sequential(self, tool_calls: list) -> list[tuple]:
        """Execute tools one at a time."""
        results = []
        for tc in tool_calls:
            result = await self._execute_single_tool(tc)
            results.append(result)
        return results

    async def _execute_tools_parallel(self, tool_calls: list, max_concurrent: int = 5) -> list[tuple]:
        """Execute tools in parallel using a semaphore to limit concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_execute(tc):
            async with semaphore:
                return await self._execute_single_tool(tc)

        task_results = await asyncio.gather(
            *[bounded_execute(tc) for tc in tool_calls],
            return_exceptions=True
        )
        
        # Handle any exceptions
        processed_results = []
        for tc, result in zip(tool_calls, task_results):
            if isinstance(result, Exception):
                tool_msg = Message(
                    role="tool",
                    content=f"Error: {type(result).__name__}: {str(result)}",
                    tool_call_id=tc.id,
                    name=tc.function.name,
                )
                processed_results.append((tc, tool_msg))
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_tool(self, tool_call) -> tuple:
        """Execute a single tool and return (tool_call, tool_msg)."""
        tool_call_id = tool_call.id
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        tool = self.tools.get(function_name)
        if not tool:
            tool_msg = Message(
                role="tool",
                content=f"Error: Unknown tool: {function_name}",
                tool_call_id=tool_call_id,
                name=function_name,
            )
            return (tool_call, tool_msg)

        try:
            result = await tool.execute(**arguments)
        except Exception as e:
            result = ToolResult(success=False, content="", error=str(e))

        tool_msg = Message(
            role="tool",
            content=result.content if result.success else f"Error: {result.error}",
            tool_call_id=tool_call_id,
            name=function_name,
        )
        return (tool_call, tool_msg)

    def cleanup(self) -> None:
        """Clean up resources held by the sub-agent.
        
        Should be called when sub-agent is no longer needed.
        """
        # Clear tool references to free memory
        self.tools.clear()
        self.tool_list.clear()


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
            try:
                return await agent.run(task)
            finally:
                agent.cleanup()

    results = await asyncio.gather(*[run_one(t) for t in tasks])
    return list(results)