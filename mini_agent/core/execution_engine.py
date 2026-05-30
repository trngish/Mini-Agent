"""核心执行引擎模块。

本模块为 Agent 提供核心执行引擎，
将主执行循环与 Agent 编排逻辑分离。
"""

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from ..logger import AgentLogger
from ..schema import AgentMode, FunctionCall, Message, ToolCall
from ..tools.base import Tool, ToolResult
from ..utils import Colors
from ..utils.tool_error_handler import handle_tool_error
from ..utils.tool_group_optimizer import ToolGroupOptimizer
from .rate_limiter import RateLimiter


class ExecutionEngine:
    """处理工具执行和并行处理。

    为工具调用提供智能批处理、并行执行和错误处理。
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        logger: AgentLogger,
        retry_handler: Any,
        metrics: Any,
        error_recovery: Any,
        write_tools: set[str] | frozenset[str],
        rate_limiter: RateLimiter | None = None,
    ):
        self.tools = tools
        self.logger = logger
        self._retry_handler = retry_handler
        self._metrics = metrics
        self._error_recovery = error_recovery
        self.write_tools = write_tools
        self._rate_limiter = rate_limiter

    async def execute_tools(
        self,
        tool_calls: list[ToolCall],
        max_concurrent: int,
        parallel_enabled: bool,
        mode: AgentMode,
        check_approved_fn: Callable[[str], bool],
    ) -> list[tuple[ToolCall, Message]]:
        """智能批处理执行工具调用。

        参数:
            tool_calls: 要执行的工具调用列表
            max_concurrent: 最大并发工具执行数
            parallel_enabled: 是否启用并行执行
            mode: 当前 Agent 模式 (YOLO, AGENT, PLAN)
            check_approved_fn: 回调函数，用于检查工具在 AGENT 模式下是否已批准

        返回:
            (tool_call, message) 元组列表
        """
        if not tool_calls:
            return []

        if parallel_enabled and len(tool_calls) > 1:
            if ToolGroupOptimizer.can_parallelize(tool_calls):
                tool_calls = self._optimize_tool_calls(tool_calls)
                return await self._execute_parallel(tool_calls, max_concurrent, mode, check_approved_fn)
            else:
                return await self._execute_batched(tool_calls, max_concurrent, mode, check_approved_fn)
        else:
            return await self._execute_sequential(tool_calls, mode, check_approved_fn)

    def optimize_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        """通过去重 multi_read 中的路径来优化工具调用。"""
        return self._optimize_tool_calls(tool_calls)

    async def _execute_sequential(
        self,
        tool_calls: list[ToolCall],
        mode: AgentMode,
        check_approved_fn: Callable[[str], bool],
    ) -> list[tuple[ToolCall, Message]]:
        """逐个执行工具。"""
        results = []
        for tc in tool_calls:
            result = await self._execute_single_tool(tc, mode, check_approved_fn)
            results.append(result)
        return results

    async def _execute_parallel(
        self,
        tool_calls: list[ToolCall],
        max_concurrent: int,
        mode: AgentMode,
        check_approved_fn: Callable[[str], bool],
    ) -> list[tuple[ToolCall, Message]]:
        """带并发限制的并行执行工具。"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_limit(tc: ToolCall) -> tuple[ToolCall, Message]:
            async with semaphore:
                return await self._execute_single_tool(tc, mode, check_approved_fn)

        # B1 修复: _execute_single_tool 已打印每个工具调用。
        # 删除此循环可防止每个工具出现两次。
        task_results = await asyncio.gather(*[execute_with_limit(tc) for tc in tool_calls], return_exceptions=True)

        processed_results: list[tuple[ToolCall, Message]] = []
        for i, result in enumerate(task_results):
            tc = tool_calls[i]
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

    async def _execute_batched(
        self,
        tool_calls: list[ToolCall],
        max_concurrent: int,
        mode: AgentMode,
        check_approved_fn: Callable[[str], bool],
    ) -> list[tuple[ToolCall, Message]]:
        """按依赖顺序分批执行。"""
        batches = ToolGroupOptimizer.group_by_dependency(tool_calls)
        results = []
        for batch in batches:
            if len(batch) == 1:
                result = await self._execute_single_tool(batch[0], mode, check_approved_fn)
                results.append(result)
            else:
                batch_results = await self._execute_parallel(batch, max_concurrent, mode, check_approved_fn)
                results.extend(batch_results)
        return results

    def _extract_error_retry_after(self, error: str) -> int | None:
        """从错误消息中提取 retry_after 延迟时间。

        参数:
            error: 要解析的错误消息

        返回:
            如找到则返回秒为单位的延迟，否则返回 None
        """
        patterns = [
            r"retry[_\s]?after[:\s]*(\d+)",
            r"retry[_\s]?after[:\s]*(\d+)\s*seconds?",
            r"wait\s*(\d+)\s*seconds?",
            r"rate.*limit.*retry.*(\d+)",
            r"retry\s*in\s*(\d+)\s*seconds?",
        ]
        for pattern in patterns:
            match = re.search(pattern, error.lower())
            if match:
                return int(match.group(1))
        return None

    def _is_non_retryable_error(self, error: str | None) -> bool:
        """检查错误是否不应重试。

        参数:
            error: 要检查的错误消息

        返回:
            如果错误不应重试则返回 True
        """
        if not error:
            return True
        non_retryable = ["rejected", "blocked", "permission denied", "unauthorized", "invalid"]
        return any(x in error.lower() for x in non_retryable)

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        mode: AgentMode,
        check_approved_fn: Callable[[str], bool],
    ) -> tuple[ToolCall, Message]:
        """根据模式执行单个工具，包含基于模式的批准和错误处理。"""
        tool_call_id = tool_call.id
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        self._print_tool_call(function_name, arguments)

        if self._rate_limiter is not None:
            allowed, rate_msg = self._rate_limiter.check_rate(function_name)
            if not allowed:
                result = ToolResult(success=False, content="", error=rate_msg)
                self._on_tool_result(function_name, result, arguments)
                tool_msg = Message(
                    role="tool",
                    content=f"Error: {rate_msg}",
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                return (tool_call, tool_msg)
            valid, len_msg = self._rate_limiter.validate_input_length(function_name, arguments)
            if not valid:
                result = ToolResult(success=False, content="", error=len_msg)
                self._on_tool_result(function_name, result, arguments)
                tool_msg = Message(
                    role="tool",
                    content=f"Error: {len_msg}",
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                return (tool_call, tool_msg)

        if mode == AgentMode.PLAN and function_name in self.write_tools:
            result = ToolResult(
                success=False,
                content="",
                error=f"Blocked in PLAN mode (read-only). Switch to /mode agent to use {function_name}.",
            )
            self._on_tool_result(function_name, result, arguments)
            tool_msg = Message(
                role="tool",
                content=f"Error: {result.error}",
                tool_call_id=tool_call_id,
                name=function_name,
            )
            return (tool_call, tool_msg)

        if mode == AgentMode.AGENT and not check_approved_fn(function_name):
            result = ToolResult(
                success=False,
                content="",
                error="Tool call rejected by user. Type 'y' to approve, or switch to /mode yolo for auto-approve.",
            )
            self._on_tool_result(function_name, result, arguments)
            tool_msg = Message(
                role="tool",
                content=f"Error: {result.error}",
                tool_call_id=tool_call_id,
                name=function_name,
            )
            return (tool_call, tool_msg)

        if function_name not in self.tools:
            result = ToolResult(success=False, content="", error=f"Unknown tool: {function_name}")
        else:
            tool_start = perf_counter()

            max_retries = self._retry_handler.get_max_retries()
            last_retry_after: int | None = None
            # 工具执行超时时间（秒），避免挂起的工具阻塞整个agent
            tool_timeout = 60
            for attempt in range(max_retries):
                try:
                    tool = self.tools[function_name]
                    result = await asyncio.wait_for(tool.execute(**arguments), timeout=tool_timeout)
                    if result.success or self._is_non_retryable_error(result.error):
                        break
                    if result.error is not None and not self._retry_handler.is_transient_error(result.error):
                        break
                    # 尝试从错误消息中提取 retry_after
                    if result.error and last_retry_after is None:
                        last_retry_after = self._extract_error_retry_after(result.error)
                except asyncio.TimeoutError:
                    result = ToolResult(
                        success=False,
                        content="",
                        error=f"Tool execution timed out after {tool_timeout}s: {function_name}",
                    )
                    break
                except Exception as e:
                    tool_error = handle_tool_error(function_name, arguments, e)
                    result = ToolResult(
                        success=False,
                        content="",
                        error=tool_error.message,
                    )
                    if self._is_non_retryable_error(tool_error.message):
                        break
                    if not self._retry_handler.is_transient_error(str(e)):
                        break
                    # 尝试从异常中提取 retry_after
                    if last_retry_after is None:
                        last_retry_after = self._extract_error_retry_after(str(e))

                if attempt < max_retries - 1:
                    # 如有 retry_after 值则优先使用，否则使用指数退避
                    delay = last_retry_after if last_retry_after else self._retry_handler.get_delay(attempt)
                    await asyncio.sleep(delay)
                    last_retry_after = None  # 重置以供后续使用
            else:
                pass

            tool_duration = perf_counter() - tool_start
            self._metrics.record_tool_duration(function_name, tool_duration)
            self._metrics.record_tool_result(function_name, result.success)

        self._on_tool_result(function_name, result, arguments)

        if result.success:
            self._error_recovery.record_success()
        else:
            self._error_recovery.record_failure()
            self._error_recovery.record_error(
                error=result.error or "Unknown error",
                context=f"{function_name}({json.dumps(arguments, ensure_ascii=False)[:100]})",
            )

        content = result.content if result.success else f"Error: {result.error}"

        tool_msg = Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=function_name,
        )
        return (tool_call, tool_msg)

    def _format_arguments(self, arguments: dict[str, Any]) -> str:
        truncated: dict[str, Any] = {}
        for key, value in arguments.items():
            value_str = str(value)
            truncated[key] = value_str[:200] + "..." if len(value_str) > 200 else value
        return json.dumps(truncated, indent=2, ensure_ascii=False)

    def _print_tool_call(self, function_name: str, arguments: dict[str, Any]) -> None:
        print(f"\n  {Colors.BRIGHT_YELLOW}🔧  {function_name}{Colors.RESET}")
        for line in self._format_arguments(arguments).split("\n"):
            print(f"  {Colors.DIM}{line}{Colors.RESET}")

    def _print_tool_result(self, result: ToolResult) -> None:
        if result.success:
            text = result.content
            if len(text) > 800:  # F1 修复: 从 300 增加到显示更多上下文
                text = text[:800] + f"{Colors.DIM}...{Colors.RESET}"
            print(f"{Colors.BRIGHT_GREEN}✓ Result:\n{Colors.RESET}{text}")
        else:
            print(f"{Colors.BRIGHT_RED}✗ 错误:\n{Colors.RESET}{Colors.RED}{result.error}{Colors.RESET}\n")  # F2 修复: 尾部添加换行符

    def _on_tool_result(self, function_name: str, result: ToolResult, arguments: dict[str, Any] | None = None) -> None:
        self._print_tool_result(result)
        self.logger.log_tool_result(
            tool_name=function_name,
            arguments=arguments or {},
            result_success=result.success,
            result_content=result.content if result.success else None,
            result_error=result.error if not result.success else None,
        )

    def _optimize_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        """通过去重 multi_read 中的路径来优化工具调用。"""
        multi_read_calls = []
        other_calls = []

        for tc in tool_calls:
            if tc.function.name == "multi_read":
                multi_read_calls.append(tc)
            else:
                other_calls.append(tc)

        if len(multi_read_calls) > 1:
            all_paths = []
            for tc in multi_read_calls:
                paths = tc.function.arguments.get("paths", [])
                if isinstance(paths, list):
                    all_paths.extend(paths)

            seen = set()
            unique_paths = []
            for p in all_paths:
                normalized = str(Path(p).resolve() if Path(p).is_absolute() else p)
                if normalized not in seen:
                    seen.add(normalized)
                    unique_paths.append(p)

            if unique_paths:
                first_call = multi_read_calls[0]
                deduped_call = ToolCall(
                    id=first_call.id,
                    type="function",
                    function=FunctionCall(name="multi_read", arguments={"paths": unique_paths}),
                )
                other_calls.insert(0, deduped_call)
                return other_calls

        return tool_calls
