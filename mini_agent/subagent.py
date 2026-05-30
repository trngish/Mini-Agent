"""用于并发后台任务执行的子代理。"""

from __future__ import annotations

import asyncio
from enum import Enum
from time import perf_counter
from typing import Any, cast

from .llm import LLMClient
from .schema import AgentMode, Message, ToolCall
from .schema.schema import WRITE_TOOLS
from .tools.base import Tool, ToolResult
from .utils import Colors
from .utils.model_utils import is_m27_model

BLOCKED_TOOLS_FOR_SUBAGENT = frozenset({"bash_kill", "team_dispatch"})


class SubAgentSecurityPolicy(str, Enum):
    YOLO = "yolo"
    APPROVE_WRITE = "approve_write"
    APPROVE_ALL = "approve_all"


class SubAgentResult:
    """子代理执行结果。"""

    def __init__(self, task: str, content: str, success: bool, elapsed: float, error: str = ""):
        self.task = task
        self.content = content
        self.success = success
        self.elapsed = elapsed
        self.error = error


class SubAgent:
    """用于后台子任务的独立代理。

    拥有自己的LLM客户端和工具集，与父代理独立运行。
    支持M2.7并行工具执行（启用时）。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[Tool],
        system_prompt: str = "You are a helpful assistant. Complete the assigned task concisely.",
        max_steps: int = 50,
        m27_config: dict[str, Any] | None = None,
        security_policy: SubAgentSecurityPolicy = SubAgentSecurityPolicy.YOLO,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_list = list(tools)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.m27_config = m27_config or {}
        model_name = getattr(llm_client, "model", "")
        self.is_m27 = is_m27_model(model_name)
        if security_policy == SubAgentSecurityPolicy.APPROVE_ALL:
            self.mode = AgentMode.AGENT
            self._approve_write_only = False
        elif security_policy == SubAgentSecurityPolicy.APPROVE_WRITE:
            self.mode = AgentMode.AGENT
            self._approve_write_only = True
        else:
            self.mode = AgentMode.YOLO
            self._approve_write_only = False
        self.write_tools = WRITE_TOOLS

    async def run(self, task: str) -> SubAgentResult:
        """执行子任务并返回结果。"""
        start = perf_counter()
        task_short = task[:80] + "..." if len(task) > 80 else task
        print(f"  {Colors.DIM}🔄 SubAgent: {task_short}{Colors.RESET}")

        # 如果M2.7可用，配置思考预算
        if self.is_m27 and self.m27_config:
            thinking_budget = self.m27_config.get("thinking_budget_tokens", 16384)
            if hasattr(self.llm, "configure_m27"):
                self.llm.configure_m27(self.m27_config)
            elif hasattr(self.llm, "configure_thinking_budget"):
                self.llm.configure_thinking_budget(thinking_budget)

        messages = [Message(role="system", content=self.system_prompt), Message(role="user", content=task)]

        for _step in range(self.max_steps):
            try:
                response = await self.llm.generate(messages=messages, tools=self.tool_list)
            except Exception as e:
                elapsed = perf_counter() - start
                print(f"  {Colors.RED}❌ SubAgent error ({elapsed:.1f}s): {e}{Colors.RESET}")
                return SubAgentResult(
                    task=task,
                    content="",
                    success=False,
                    elapsed=elapsed,
                    error=str(e),
                )

            if response.tool_calls:
                # M2.7：如果启用且有多个工具，并行执行工具
                parallel_enabled = self.m27_config.get("enable_parallel_tool_calls", True)
                max_concurrent = self.m27_config.get("max_concurrent_tools", 10)

                if parallel_enabled and len(response.tool_calls) > 1:
                    results = await self._execute_tools_parallel(response.tool_calls, max_concurrent)
                else:
                    results = await self._execute_tools_sequential(response.tool_calls)

                # 关键：在工具结果之前添加带tool_calls的助手消息
                # 这是API将工具结果与其调用匹配所必需的
                assistant_msg = Message(
                    role="assistant",
                    content=response.content or "",
                    thinking=response.thinking,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                # Add tool messages in order
                for _, tool_msg in results:
                    messages.append(tool_msg)
                continue

            elapsed = perf_counter() - start
            print(f"  {Colors.DIM}✅ SubAgent done ({elapsed:.1f}s): {task[:60]}{Colors.RESET}")
            return SubAgentResult(
                task=task,
                content=response.content or "",
                success=True,
                elapsed=elapsed,
            )

        elapsed = perf_counter() - start
        print(f"  {Colors.YELLOW}⚠️  SubAgent max steps ({elapsed:.1f}s): {task[:60]}{Colors.RESET}")
        return SubAgentResult(
            task=task,
            content="",
            success=False,
            elapsed=elapsed,
            error="Max steps reached",
        )

    async def _execute_tools_sequential(self, tool_calls: list[ToolCall]) -> list[tuple[ToolCall, Message]]:
        """一次执行一个工具。"""
        results: list[tuple[ToolCall, Message]] = []
        for tc in tool_calls:
            result = await self._execute_single_tool(tc)
            results.append(result)
        return results

    async def _execute_tools_parallel(
        self, tool_calls: list[ToolCall], max_concurrent: int = 5
    ) -> list[tuple[ToolCall, Message]]:
        """使用信号量限制并发来并行执行工具。"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_execute(tc: ToolCall) -> tuple[ToolCall, Message]:
            async with semaphore:
                return await self._execute_single_tool(tc)

        task_results = await asyncio.gather(*[bounded_execute(tc) for tc in tool_calls], return_exceptions=True)

        processed_results: list[tuple[ToolCall, Message]] = []
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
                processed_results.append(cast(tuple[ToolCall, Message], result))

        return processed_results

    async def _execute_single_tool(self, tool_call: ToolCall) -> tuple[ToolCall, Message]:
        tool_call_id = tool_call.id
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        if self._approve_write_only and function_name in self.write_tools:
            tool_msg = Message(
                role="tool",
                content=(
                    f"Write operation '{function_name}' requires approval "
                    f"in approve_write mode. Please use the parent agent for write operations."
                ),
                tool_call_id=tool_call_id,
                name=function_name,
            )
            return (tool_call, tool_msg)

        if function_name in BLOCKED_TOOLS_FOR_SUBAGENT:
            tool_msg = Message(
                role="tool",
                content=f"Error: Tool '{function_name}' is blocked for sub-agents for security reasons",
                tool_call_id=tool_call_id,
                name=function_name,
            )
            return (tool_call, tool_msg)

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
        """清理子代理持有的资源。

        当子代理不再需要时应调用此方法。
        """
        # 清除工具引用以释放内存
        self.tools.clear()
        self.tool_list.clear()

        # 清理后台shell（如果创建了的话）
        try:
            # 调度清理而不阻塞
            import asyncio

            from .tools.bash_background import BackgroundShellManager

            loop = asyncio.get_running_loop()
            loop.create_task(BackgroundShellManager.cleanup_all())
        except Exception:
            pass  # Ignore cleanup errors


async def run_sub_agents(
    llm_client: LLMClient,
    tasks: list[str],
    tools: list[Tool],
    max_concurrent: int = 3,
    security_policy: SubAgentSecurityPolicy = SubAgentSecurityPolicy.YOLO,
) -> list[SubAgentResult]:
    """并发运行多个子代理。

    Args:
        llm_client: LLM客户端（将为每个子代理克隆以避免tool_call id冲突）
        tasks: 任务描述列表
        tools: 可用工具
        max_concurrent: 最大并发子代理数

    Returns:
        与tasks相同顺序的SubAgentResult列表
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(task: str) -> SubAgentResult:
        async with semaphore:
            # 为每个子代理克隆LLMClient以避免tool_call id冲突
            # 共享客户端会导致API错误："tool result's tool id not found"
            agent_llm = llm_client.clone() if hasattr(llm_client, "clone") else llm_client
            agent = SubAgent(llm_client=agent_llm, tools=tools, security_policy=security_policy)
            try:
                return await agent.run(task)
            finally:
                agent.cleanup()

    results = await asyncio.gather(*[run_one(t) for t in tasks])
    return list(results)
