"""Core Agent implementation."""

import asyncio
import json
import os
import re
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

from .llm import LLMClient
from .logger import AgentLogger
from .schema import AgentMode, Message, ToolCall
from .schema.schema import WRITE_TOOLS
from .session import SessionManager
from .tools.base import Tool, ToolResult
from .utils import Colors, calculate_display_width
from .utils.token_utils import get_encoder
from .utils.error_handler import LLMErrorClassifier, format_llm_error
from .utils.model_utils import is_m27_model, get_token_limit_for_model
from .utils.tool_error_handler import handle_tool_error

# Constants - avoid magic numbers
STREAM_BUFFER_SIZE = int(os.environ.get("MINI_AGENT_STREAM_BUFFER_SIZE", "10"))
DEFAULT_ENCODING_NAME = "cl100k_base"

# Adaptive thinking budget levels (按次数计费优化：token免费，思考越深命中率越高)
# 提高基础预算：更深思考 → 更高命中率 → 更少重试 → 更少总调用次数
THINKING_BUDGET_SIMPLE = 16384      # 简单任务：原8K→16K，确保一次做对
THINKING_BUDGET_MEDIUM = 24576      # 中等任务：原16K→24K，减少返工
THINKING_BUDGET_COMPLEX = 32768     # 复杂任务：原24K→32K，深度规划
THINKING_BUDGET_SUPER = 32768       # 超复杂任务：32K上限

# Complexity indicators for auto-detection
COMPLEXITY_HIGH_KEYWORDS = {
    "重构", "refactor", "架构", "architecture", "重写", "rewrite",
    "迁移", "migrate", "全面", "comprehensive", "整体", "entire",
    "所有", "all files", "批量", "batch", "多个文件", "multi-file",
    "设计", "design", "调试", "debug", "排查", "investigate",
    "优化", "optimize", "性能", "performance",
}

COMPLEXITY_MEDIUM_KEYWORDS = {
    "修改", "modify", "fix", "修复", "实现", "implement", "添加", "add",
    "更新", "update", "创建", "create", "搜索", "search", "分析", "analyze",
    "检查", "check", "比较", "compare", "转换", "convert",
}


class Agent:
    """Single agent with basic tools and MCP support."""

    # Module-level tiktoken encoder cache for reuse (shared safely via dict lookup)
    _encoder_cache: dict[str, Any] = {}

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
        token_limit: int = 80000,
        m27_config: Optional[dict] = None,
        mode: AgentMode = AgentMode.YOLO,
    ):
        self.mode = mode
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_list = list(tools)
        self.max_steps = max_steps
        self.workspace_dir = Path(workspace_dir)
        self.cancel_event: Optional[asyncio.Event] = None
        self.session_manager = SessionManager()
        # Make write tools configurable at instance level
        self.write_tools = WRITE_TOOLS

        # M2.7 specific configuration
        self.m27_config = m27_config or {}
        model_name = getattr(llm_client, 'model', '')
        self.is_m27 = is_m27_model(model_name)
        
        # Use unified token limit calculation
        self.token_limit = get_token_limit_for_model(
            model_name,
            self.m27_config.get("token_limit") if self.is_m27 else token_limit
        )
        
        # M2.7 supports up to 32K output tokens, store for reference
        self.max_output_tokens = self.m27_config.get("max_output_tokens", 16384) if self.is_m27 else 8192
        
        # Optimization: adaptive thinking budget
        # Max budget from config, actual budget is dynamically adjusted per task
        self._max_thinking_budget = self.m27_config.get("thinking_budget_tokens", 16384) if self.is_m27 else 0
        self.thinking_budget = self._max_thinking_budget  # Start with max, will be adjusted per task
        
        # Optimization: track consecutive tool failures - handle locally before reporting
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3  # 连续失败3次才回话报告
        
        # Optimization: batch size for parallel tool execution
        # More tools per call = fewer API calls
        # M2.7 supports 20+ parallel tool calls with 97% following rate
        self._max_tools_per_call = self.m27_config.get("max_concurrent_tools", 20) if self.is_m27 else 3

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Inject workspace information into system prompt if not already present
        if "Current Workspace" not in system_prompt:
            workspace_info = f"\n\n## Current Workspace\nYou are currently working in: `{self.workspace_dir.absolute()}`\nAll relative paths will be resolved relative to this directory."
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt

        # Inject mode instructions into system prompt
        mode_instructions = {
            AgentMode.PLAN: (
                "\n\n## Mode: Plan\nYou are in PLAN mode. You can ONLY read files and explore."
                " You MUST NOT modify files, execute commands, or make any changes."
                " Propose a plan to the user before any write operations."
            ),
            AgentMode.AGENT: (
                "\n\n## Mode: Agent\nYou are in AGENT mode (default)."
                " You can use all tools. Each tool call will require user approval."
            ),
            AgentMode.YOLO: (
                "\n\n## Mode: YOLO\nYou are in YOLO mode."
                " All tool calls are auto-approved. Execute efficiently."
            ),
        }
        self.system_prompt += mode_instructions.get(self.mode, "")

        # Initialize message history
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

        # Initialize logger
        self.logger = AgentLogger()

        # API call tracking
        self.api_call_count: int = 0
        # Token usage from last API response (updated after each LLM call)
        self.api_total_tokens: int = 0
        # Flag to skip token check right after summary (avoid consecutive triggers)
        self._skip_next_token_check: bool = False
        # Incremental token estimation cache
        self._cached_token_count: int = 0
        self._cached_token_index: int = 0

    @classmethod
    def _get_cached_encoder(cls, encoding_name: str = DEFAULT_ENCODING_NAME):
        """Get cached tiktoken encoder or create new one.
        
        Args:
            encoding_name: Name of the encoding (default: cl100k_base)
            
        Returns:
            Cached encoder instance
        """
        if encoding_name not in cls._encoder_cache:
            cls._encoder_cache[encoding_name] = get_encoder(encoding_name)
        return cls._encoder_cache[encoding_name]

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))
        # Adaptively adjust thinking budget based on task complexity
        self._adjust_thinking_budget(content)

    def _adjust_thinking_budget(self, user_message: str) -> None:
        """Adaptively adjust thinking budget based on task complexity.

        按次数计费优化：token免费，思考越深→命中率越高→重试越少→总调用次数越少。
        简单任务给足思考空间也能提高单次完成率。

        Args:
            user_message: The user's message to analyze for complexity
        """
        if not self.is_m27:
            return

        msg_lower = user_message.lower()

        # Detect complexity level from keywords
        high_matches = sum(1 for kw in COMPLEXITY_HIGH_KEYWORDS if kw in msg_lower)
        medium_matches = sum(1 for kw in COMPLEXITY_MEDIUM_KEYWORDS if kw in msg_lower)

        # Estimate file count mentioned
        file_mentions = len(re.findall(r'\.(py|js|ts|jsx|tsx|java|go|rs|c|cpp|h|rb|php|yaml|yml|json|toml|md|txt|csv|sql|sh|bash|ps1)', msg_lower))

        # Determine complexity
        if high_matches >= 2 or file_mentions >= 4:
            new_budget = THINKING_BUDGET_SUPER
            level = "超复杂"
        elif high_matches >= 1 or file_mentions >= 2 or medium_matches >= 3:
            new_budget = THINKING_BUDGET_COMPLEX
            level = "复杂"
        elif medium_matches >= 1 or file_mentions >= 1:
            new_budget = THINKING_BUDGET_MEDIUM
            level = "中等"
        else:
            new_budget = THINKING_BUDGET_SIMPLE
            level = "简单"

        # Also consider message length as a signal
        msg_tokens = len(user_message) // 3  # rough estimation
        if msg_tokens > 2000:
            new_budget = max(new_budget, THINKING_BUDGET_COMPLEX)
            level = "复杂(长消息)"

        # Constrain to max budget from config
        new_budget = min(new_budget, self._max_thinking_budget)

        if new_budget != self.thinking_budget:
            old_budget = self.thinking_budget
            self.thinking_budget = new_budget
            # Update the LLM client's thinking budget dynamically
            # Note: _thinking_budget_tokens is an internal interface exposed by the underlying
            # LLM client for M2.7 optimization. Must remain compatible.
            llm_client = getattr(self.llm, '_client', None)
            if llm_client is not None and hasattr(llm_client, '_thinking_budget_tokens'):
                llm_client._thinking_budget_tokens = new_budget
            print(f"{Colors.DIM}🧠 Thinking budget adjusted: {old_budget} → {new_budget} ({level}任务){Colors.RESET}")

    def _check_cancelled(self) -> bool:
        """Check if agent execution has been cancelled.

        Returns:
            True if cancelled, False otherwise.
        """
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        return False

    def _cleanup_incomplete_messages(self):
        """Remove the incomplete assistant message and its partial tool results.

        This ensures message consistency after cancellation by removing
        only the current step's incomplete messages, preserving completed steps.
        """
        # Find the index of the last assistant message
        last_assistant_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            # No assistant message found, nothing to clean
            return

        # Remove the last assistant message and all tool results after it
        removed_count = len(self.messages) - last_assistant_idx
        if removed_count > 0:
            self.messages = self.messages[:last_assistant_idx]
            print(f"{Colors.DIM}   Cleaned up {removed_count} incomplete message(s){Colors.RESET}")

    def _estimate_tokens(self) -> int:
        """Accurately calculate token count for message history using tiktoken

        Uses incremental estimation: only encodes new messages since last check.
        Full recalculation after summarization (when message list is rebuilt).
        """
        try:
            # Use cached encoder instead of creating new one
            encoding = self._get_cached_encoder("cl100k_base")
        except Exception:
            return self._estimate_tokens_fallback()

        # Incremental: only encode messages from cached index onward
        if self._cached_token_index >= len(self.messages):
            return self._cached_token_count

        new_tokens = 0
        for msg in self.messages[self._cached_token_index:]:
            if isinstance(msg.content, str):
                new_tokens += len(encoding.encode(msg.content))
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        new_tokens += len(encoding.encode(str(block)))

            if msg.thinking:
                new_tokens += len(encoding.encode(msg.thinking))

            if msg.tool_calls:
                new_tokens += len(encoding.encode(str(msg.tool_calls)))

            new_tokens += 4

        self._cached_token_count = self._cached_token_count + new_tokens
        self._cached_token_index = len(self.messages)
        return self._cached_token_count

    def _estimate_tokens_fallback(self) -> int:
        """Fallback token estimation method (when tiktoken is unavailable)"""
        total_chars = 0
        for msg in self.messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        total_chars += len(str(block))

            if msg.thinking:
                total_chars += len(msg.thinking)

            if msg.tool_calls:
                total_chars += len(str(msg.tool_calls))

        # Rough estimation: average 2.5 characters = 1 token
        return int(total_chars / 2.5)

    async def _summarize_messages(self):
        """Message history summarization: summarize conversations between user messages when tokens exceed limit

        Strategy (Agent mode):
        - Keep all user messages (these are user intents)
        - Summarize content between each user-user pair (agent execution process)
        - If last round is still executing (has agent/tool messages but no next user), also summarize
        - Structure: system -> user1 -> summary1 -> user2 -> summary2 -> user3 -> summary3 (if executing)

        Summary is triggered when EITHER:
        - Local token estimation exceeds limit
        - API reported total_tokens exceeds limit
        """
        # Skip check if we just completed a summary (wait for next LLM call to update api_total_tokens)
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return

        estimated_tokens = self._estimate_tokens()

        # Check both local estimation and API reported tokens
        should_summarize = estimated_tokens > self.token_limit or self.api_total_tokens > self.token_limit

        # If neither exceeded, no summary needed
        if not should_summarize:
            return

        print(
            f"\n{Colors.BRIGHT_YELLOW}📊 Token usage - Local estimate: {estimated_tokens}, API reported: {self.api_total_tokens}, Limit: {self.token_limit}{Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_YELLOW}🔄 Triggering message history summarization...{Colors.RESET}")

        # Find all user message indices (skip system prompt)
        user_indices = [i for i, msg in enumerate(self.messages) if msg.role == "user" and i > 0]

        # Need at least 1 user message to perform summary
        if len(user_indices) < 1:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  Insufficient messages, cannot summarize{Colors.RESET}")
            return

        # Build new message list
        new_messages = [self.messages[0]]  # Keep system prompt
        summary_count = 0

        # Iterate through each user message and summarize the execution process after it
        for i, user_idx in enumerate(user_indices):
            # Add current user message
            new_messages.append(self.messages[user_idx])

            # Truncate long user messages to avoid token limit issues
            user_content = self.messages[user_idx].content
            if len(user_content) > 5000:  # ~2000 tokens
                self.messages[user_idx].content = user_content[:5000] + "...[truncated]..."

            # Determine message range to summarize
            # If last user, go to end of message list; otherwise to before next user
            if i < len(user_indices) - 1:
                next_user_idx = user_indices[i + 1]
            else:
                next_user_idx = len(self.messages)

            # Extract execution messages for this round
            execution_messages = self.messages[user_idx + 1 : next_user_idx]

            # If there are execution messages in this round, summarize them
            if execution_messages:
                summary_text = self._create_local_summary(execution_messages, i + 1)
                if summary_text:
                    summary_message = Message(
                        role="user",
                        content=f"[Execution Summary {i + 1}]\n\n{summary_text}",
                    )
                    new_messages.append(summary_message)
                    summary_count += 1

        # Replace message list
        self.messages = new_messages

        # Reset token cache since message list was rebuilt
        self._cached_token_count = 0
        self._cached_token_index = 0

        # Skip next token check to avoid consecutive summary triggers
        # (api_total_tokens will be updated after next LLM call)
        self._skip_next_token_check = True

        new_tokens = self._estimate_tokens()
        print(f"{Colors.BRIGHT_GREEN}✓ Summary completed, local tokens: {estimated_tokens} → {new_tokens}{Colors.RESET}")
        print(f"{Colors.DIM}  Structure: system + {len(user_indices)} user messages + {summary_count} summaries{Colors.RESET}")
        print(f"{Colors.DIM}  Note: API token count will update on next LLM call{Colors.RESET}")

    def _create_local_summary(self, messages: list[Message], round_num: int) -> str:
        """Create summary locally without LLM call (saves tokens).

        按次数计费优化：token免费，保留更多细节以减少后续重复调用。
        摘要质量越高，LLM越不需要重新获取信息。
        提高截断限制：2000字符（assistant）、1500字符（tool result），
        因为信息丢失导致的重调成本远高于多传一些token。

        Args:
            messages: List of messages to summarize
            round_num: Round number

        Returns:
            Summary text
        """
        if not messages:
            return ""

        # Build structured summary
        lines = [f"Round {round_num}:"]
        tool_calls_count = 0
        tool_results_count = 0
        assistant_responses = []

        for msg in messages:
            if msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    # 按次数计费优化：保留更多内容（2000字符），大幅减少信息丢失导致的重调
                    if len(content) > 2000:
                        content = content[:2000] + "..."
                    assistant_responses.append(content)
                if msg.tool_calls:
                    tool_names = [tc.function.name for tc in msg.tool_calls]
                    # 保留工具参数概要，帮助后续理解上下文
                    tool_details = []
                    for tc in msg.tool_calls:
                        args_str = str(tc.function.arguments)
                        if len(args_str) > 150:
                            args_str = args_str[:150] + "..."
                        tool_details.append(f"{tc.function.name}({args_str})")
                    lines.append(f"  Tools called: {', '.join(tool_details)}")
                    tool_calls_count += len(msg.tool_calls)
            elif msg.role == "tool":
                tool_results_count += 1
                result = msg.content if isinstance(msg.content, str) else str(msg.content)
                # 按次数计费优化：保留更多结果（1500字符），避免因信息不足重新获取
                if len(result) > 1500:
                    result = result[:1500] + "..."
                lines.append(f"  Result: {result}")

        # Add assistant response summary if no tools were called
        if not tool_calls_count and assistant_responses:
            lines.append(f"  Response: {assistant_responses[0]}")

        # Add statistics
        if tool_calls_count or tool_results_count:
            lines.append(f"  Stats: {tool_calls_count} tool(s), {tool_results_count} result(s)")

        return "\n".join(lines)

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task is complete or max steps reached.

        Args:
            cancel_event: Optional asyncio.Event that can be set to cancel execution.
                          When set, the agent will stop at the next safe checkpoint
                          (after completing the current step to keep messages consistent).

        Returns:
            The final response content, or error message (including cancellation message).
        """
        # Set cancellation event (can also be set via self.cancel_event before calling run())
        if cancel_event is not None:
            self.cancel_event = cancel_event

        # Start new run, initialize log file
        self.logger.start_new_run()
        print(f"{Colors.DIM}📝 Log file: {self.logger.get_log_file_path()}{Colors.RESET}")

        step = 0
        run_start_time = perf_counter()

        while step < self.max_steps:
            # Check for cancellation at start of each step
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                cancel_msg = "Task cancelled by user."
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                return cancel_msg

            step_start_time = perf_counter()
            # Check and summarize message history to prevent context overflow
            await self._summarize_messages()

            # Step header
            step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}  Step {step + 1}/{self.max_steps}{Colors.RESET}"
            print(f"\n  {Colors.DIM}╭{'─' * 44}╮{Colors.RESET}")
            print(f"  {Colors.DIM}│{Colors.RESET} {step_text}{' ' * (44 - calculate_display_width(step_text) - 1)}{Colors.DIM}│{Colors.RESET}")
            print(f"  {Colors.DIM}╰{'─' * 44}╯{Colors.RESET}")

            # Get tool list for LLM call (cached during session)
            tool_list = self.tool_list

            # Log LLM request and call LLM with Tool objects directly
            self.logger.log_request(messages=self.messages, tools=tool_list)

            try:
                # Setup streaming callbacks for real-time output with buffering
                thinking_first = True
                content_first = True
                
                # Use deque for efficient buffer management with maxlen
                buffer_size = STREAM_BUFFER_SIZE  # Configurable via MINI_AGENT_STREAM_BUFFER_SIZE
                thinking_buffer: deque[str] = deque(maxlen=buffer_size)
                content_buffer: deque[str] = deque(maxlen=buffer_size)

                def on_thinking(text: str) -> None:
                    nonlocal thinking_first, thinking_buffer
                    if thinking_first:
                        print(f"\n  {Colors.BOLD}{Colors.MAGENTA}🧠 Think{Colors.RESET}")
                        thinking_first = False
                    thinking_buffer.append(text)
                    if len(thinking_buffer) >= buffer_size:
                        print(f"{Colors.DIM}{''.join(thinking_buffer)}{Colors.RESET}", end="", flush=True)
                        thinking_buffer.clear()

                def on_text(text: str) -> None:
                    nonlocal content_first, content_buffer
                    if content_first:
                        print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                        content_first = False
                    content_buffer.append(text)
                    if len(content_buffer) >= buffer_size:
                        print(''.join(content_buffer), end="", flush=True)
                        content_buffer.clear()

                response = await self.llm.generate(
                    messages=self.messages,
                    tools=tool_list,
                    on_text=on_text,
                    on_thinking=on_thinking,
                )
                
                # Flush any remaining buffer content
                if thinking_buffer:
                    print(f"{Colors.DIM}{''.join(thinking_buffer)}{Colors.RESET}", end="", flush=True)
                if content_buffer:
                    print(''.join(content_buffer), end="", flush=True)

                # Print newline after streaming output if any was streamed
                if not content_first or not thinking_first:
                    print()

            except Exception as e:
                # Use structured error handling
                llm_error = LLMErrorClassifier.classify(e)

                if llm_error.is_retryable and llm_error.retry_after:
                    print(f"\n{Colors.BRIGHT_YELLOW}Rate limited. Waiting {llm_error.retry_after}s before returning...{Colors.RESET}")

                error_msg = format_llm_error(e)
                print(f"\n{error_msg}")
                return f"LLM call failed: {llm_error.user_guidance}"

            # Accumulate API call count and token usage
            self.api_call_count += 1
            if response.usage:
                self.api_total_tokens = response.usage.total_tokens

            # 按次数计费统计：显示调用次数和工具调用数
            tool_count = len(response.tool_calls) if response.tool_calls else 0
            print(f"\n  {Colors.DIM}📊 API Call #{self.api_call_count} | Tools: {tool_count} | "
                  f"Thinking budget: {self.thinking_budget} | "
                  f"Total tokens: {self.api_total_tokens:,}{Colors.RESET}")

            # Log LLM response
            self.logger.log_response(
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
            )

            # Add assistant message
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)

            # Check if task is complete (no tool calls)
            if not response.tool_calls:
                step_elapsed = perf_counter() - step_start_time
                total_elapsed = perf_counter() - run_start_time
                print(f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")
                print(f"{Colors.BRIGHT_GREEN}💰 Total API calls: {self.api_call_count} (按次数计费统计){Colors.RESET}")
                return response.content

            # Execute tool calls (parallel if M2.7)
            parallel_enabled = self.is_m27 and self.m27_config.get("enable_parallel_tool_calls", True)
            max_concurrent = self.m27_config.get("max_concurrent_tools", 5) if self.is_m27 else 1

            if parallel_enabled and len(response.tool_calls) > 1:
                results = await self.execute_tools_parallel(response.tool_calls, max_concurrent)
            else:
                results = await self.execute_tools_sequential(response.tool_calls)

            # Append tool messages and handle cancellation
            for tool_call, tool_msg in results:
                if self._check_cancelled():
                    self._cleanup_incomplete_messages()
                    return "Task cancelled by user."
                self.messages.append(tool_msg)

            step_elapsed = perf_counter() - step_start_time
            total_elapsed = perf_counter() - run_start_time
            print(f"\n  {Colors.DIM}✔  Step {step + 1}  ({step_elapsed:.1f}s | total: {total_elapsed:.1f}s){Colors.RESET}")

            step += 1

        # Max steps reached
        error_msg = f"Task couldn't be completed after {self.max_steps} steps."
        print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {error_msg}{Colors.RESET}")
        return error_msg

    def _format_arguments(self, arguments: dict) -> str:
        """Format tool arguments for display with truncation."""
        truncated = {}
        for key, value in arguments.items():
            value_str = str(value)
            truncated[key] = value_str[:200] + "..." if len(value_str) > 200 else value
        return json.dumps(truncated, indent=2, ensure_ascii=False)

    def _print_tool_call(self, function_name: str, arguments: dict):
        """Print tool call header and arguments."""
        print(f"\n  {Colors.BRIGHT_YELLOW}🔧  {function_name}{Colors.RESET}")
        for line in self._format_arguments(arguments).split("\n"):
            print(f"  {Colors.DIM}{line}{Colors.RESET}")

    def _print_tool_result(self, result: ToolResult):
        """Print tool execution result."""
        if result.success:
            text = result.content
            if len(text) > 300:
                text = text[:300] + f"{Colors.DIM}...{Colors.RESET}"
            print(f"{Colors.BRIGHT_GREEN}✓ Result:\n{Colors.RESET}{text}")
        else:
            print(f"{Colors.BRIGHT_RED}✗ Error:\n{Colors.RESET}{Colors.RED}{result.error}{Colors.RESET}")

    def _on_tool_result(self, function_name: str, result: ToolResult) -> None:
        """Handle tool result - print and log."""
        self._print_tool_result(result)
        self.logger.log_tool_result(
            tool_name=function_name,
            arguments={},  # Already logged in execute_single_tool
            result_success=result.success,
            result_content=result.content if result.success else None,
            result_error=result.error if not result.success else None,
        )

    async def execute_single_tool(self, tool_call: ToolCall) -> tuple[ToolCall, Message]:
        """Execute a single tool with Agent-specific behavior (print, log, approve)."""
        tool_call_id = tool_call.id
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        self._print_tool_call(function_name, arguments)

        # Plan mode: block write tools
        if self.mode == AgentMode.PLAN and function_name in self.write_tools:
            result = ToolResult(
                success=False, content="",
                error=f"Blocked in PLAN mode (read-only). Switch to /mode agent to use {function_name}.",
            )
            self._on_tool_result(function_name, result)
            tool_msg = Message(
                role="tool",
                content=f"Error: {result.error}",
                tool_call_id=tool_call_id,
                name=function_name,
            )
            return (tool_call, tool_msg)

        # Agent mode: needs confirmation
        if self.mode == AgentMode.AGENT and not self._check_approved(function_name):
            result = ToolResult(
                success=False, content="",
                error=f"Tool call rejected by user. Type 'y' to approve, or switch to /mode yolo for auto-approve.",
            )
            self._on_tool_result(function_name, result)
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
            try:
                tool = self.tools[function_name]
                result = await tool.execute(**arguments)
            except Exception as e:
                tool_error = handle_tool_error(function_name, arguments, e)
                result = ToolResult(
                    success=False,
                    content="",
                    error=tool_error.message,
                )

        self._on_tool_result(function_name, result)

        # Keep tool results intact (user pays per call, not per token)
        content = result.content if result.success else f"Error: {result.error}"
        
        tool_msg = Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=function_name,
        )
        return (tool_call, tool_msg)

    async def execute_tools_sequential(self, tool_calls: list[ToolCall]) -> list[tuple[ToolCall, Message]]:
        """Execute tools one at a time."""
        results = []
        for tc in tool_calls:
            tool_call, tool_msg = await self.execute_single_tool(tc)
            results.append((tool_call, tool_msg))
        return results

    async def execute_tools_parallel(self, tool_calls: list[ToolCall], max_concurrent: int = 5) -> list[tuple[ToolCall, Message]]:
        """Execute tools in parallel using a semaphore to limit concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_execute(tc):
            async with semaphore:
                return await self.execute_single_tool(tc)

        # Print all tool headers first
        for tc in tool_calls:
            self._print_tool_call(tc.function.name, tc.function.arguments)

        # Execute all tools concurrently
        task_results = await asyncio.gather(
            *[bounded_execute(tc) for tc in tool_calls],
            return_exceptions=True
        )
        
        # Handle any exceptions that occurred
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

    def _check_approved(self, function_name: str) -> bool:
        """Prompt user to approve a tool call in Agent mode.

        Returns True if approved, False if rejected.
        """
        if self.mode != AgentMode.AGENT:
            return True
        try:
            import threading
            import os
            result = [None]
            
            # Configurable timeout via environment (default 10 seconds)
            approval_timeout = int(os.environ.get("MINI_AGENT_APPROVAL_TIMEOUT", "10"))

            def get_input() -> None:
                result[0] = input(f"  {Colors.BRIGHT_YELLOW}Approve {function_name}? [Y/n/q]{Colors.RESET} ").strip().lower()

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=approval_timeout)

            if result[0] is None:
                return False
            if result[0] in ("q", "quit"):
                return False
            if result[0] in ("n", "no"):
                return False
            return True
        except Exception:
            return True

    def set_mode(self, mode: AgentMode) -> None:
        """Switch agent mode."""
        old_mode = self.mode
        self.mode = mode
        print(f"{Colors.GREEN}✅ Mode switched: {old_mode.value} → {mode.value}{Colors.RESET}")

    def save_session(self, label: str = "") -> str:
        """Save current session. Returns session ID."""
        sid = self.session_manager.save(self.messages, label)
        print(f"{Colors.GREEN}✅ Session saved: {sid}{Colors.RESET}")
        return sid

    def load_session(self, session_id: str) -> bool:
        """Load a saved session. Returns True on success."""
        messages = self.session_manager.load(session_id)
        if messages is None:
            print(f"{Colors.RED}❌ Session not found: {session_id}{Colors.RESET}")
            return False
        self.messages = messages
        system_count = sum(1 for m in messages if m.role == "system")
        print(f"{Colors.GREEN}✅ Session restored: {session_id} ({len(messages)} messages, {system_count} system prompts){Colors.RESET}")
        return True

    def list_sessions(self) -> list[dict]:
        """List all saved sessions."""
        return self.session_manager.list_sessions()
    
    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
    
    def cleanup(self) -> None:
        """Clean up resources held by the agent.
        
        Should be called when agent is no longer needed to ensure
        proper cleanup of background processes and connections.
        """
        # Flush logger
        if hasattr(self, 'logger'):
            self.logger.flush()
        
        # Clear cancel event
        self.cancel_event = None
        
        # Reset token cache
        self._cached_token_count = 0
        self._cached_token_index = 0
        
        # Note: Do NOT clear _encoder_cache here — it is a module-level
        # shared cache. Clearing it would affect all Agent instances.
        
        # Clean up background shells
        from .tools.bash_background import BackgroundShellManager
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(BackgroundShellManager.cleanup_all())
        except RuntimeError:
            # No running event loop — try to run synchronously
            try:
                asyncio.run(BackgroundShellManager.cleanup_all())
            except Exception as e:
                import logging
                logging.warning(f"Background shell cleanup failed: {e}")
        except Exception as e:
            import logging
            logging.warning(f"Event loop cleanup failed: {e}")