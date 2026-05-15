"""Core Agent implementation."""

import asyncio
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Optional

from .llm import LLMClient
from .logger import AgentLogger
from .schema import AgentMode, Message
from .session import SessionManager
from .subagent import SubAgent, run_sub_agents
from .tools.base import Tool, ToolResult
from .utils import Colors, calculate_display_width
from .utils.token_utils import get_encoder
from .utils.logging_config import get_logger
from .utils.message_validator import MessageValidator, ValidationError
from .utils.error_handler import LLMErrorClassifier, LLMErrorType, format_llm_error
from .utils.m27_optimization import M27Config as M27UtilsConfig, M27PromptOptimizer, M27ContextManager, is_m27_enabled


class Agent:
    """Single agent with basic tools and MCP support."""

    WRITE_TOOLS = {"write_file", "edit_file", "bash", "git"}

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

        # M2.7 specific configuration
        self.m27_config = m27_config or {}
        self.is_m27 = is_m27_enabled(llm_client.model) if hasattr(llm_client, 'model') else False
        
        # Use M2.7 token limit if configured, otherwise use provided limit
        # M2.7 has 1M context window, so we use 800K as default (80% safety margin)
        if self.is_m27 and self.m27_config.get("token_limit"):
            self.token_limit = self.m27_config["token_limit"]
        else:
            self.token_limit = token_limit
        
        # M2.7 supports up to 32K output tokens, store for reference
        self.max_output_tokens = self.m27_config.get("max_output_tokens", 16384) if self.is_m27 else 8192

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

        # Token usage from last API response (updated after each LLM call)
        self.api_total_tokens: int = 0
        # Flag to skip token check right after summary (avoid consecutive triggers)
        self._skip_next_token_check: bool = False
        # Incremental token estimation cache
        self._cached_token_count: int = 0
        self._cached_token_index: int = 0

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

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
            encoding = get_encoder("cl100k_base")
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
                summary_text = await self._create_summary(execution_messages, i + 1)
                if summary_text:
                    summary_message = Message(
                        role="user",
                        content=f"[Assistant Execution Summary]\n\n{summary_text}",
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

    async def _create_summary(self, messages: list[Message], round_num: int) -> str:
        """Create summary for one execution round

        Args:
            messages: List of messages to summarize
            round_num: Round number

        Returns:
            Summary text
        """
        if not messages:
            return ""

        # Build summary content
        summary_content = f"Round {round_num} execution process:\n\n"
        for msg in messages:
            if msg.role == "assistant":
                content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"Assistant: {content_text}\n"
                if msg.tool_calls:
                    tool_names = [tc.function.name for tc in msg.tool_calls]
                    summary_content += f"  → Called tools: {', '.join(tool_names)}\n"
            elif msg.role == "tool":
                result_preview = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"  ← Tool returned: {result_preview}...\n"

        # Call LLM to generate concise summary
        try:
            summary_prompt = f"""Please provide a concise summary of the following Agent execution process:

{summary_content}

Requirements:
1. Focus on what tasks were completed and which tools were called
2. Keep key execution results and important findings
3. Be concise and clear, within 1000 words
4. Use English
5. Do not include "user" related content, only summarize the Agent's execution process"""

            summary_msg = Message(role="user", content=summary_prompt)
            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="You are an assistant skilled at summarizing Agent execution processes.",
                    ),
                    summary_msg,
                ]
            )

            summary_text = response.content
            print(f"{Colors.BRIGHT_GREEN}✓ Summary for round {round_num} generated successfully{Colors.RESET}")
            return summary_text

        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Summary generation failed for round {round_num}: {e}{Colors.RESET}")
            # Use simple text summary on failure
            return summary_content

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
                # Setup streaming callbacks for real-time output
                thinking_first = True
                content_first = True

                def on_thinking(text: str):
                    nonlocal thinking_first
                    if thinking_first:
                        print(f"\n  {Colors.BOLD}{Colors.MAGENTA}🧠 Think{Colors.RESET}")
                        thinking_first = False
                    print(f"  {Colors.DIM}{text}{Colors.RESET}", end="", flush=True)

                def on_text(text: str):
                    nonlocal content_first
                    if content_first:
                        print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                        content_first = False
                    print(text, end="", flush=True)

                response = await self.llm.generate(
                    messages=self.messages,
                    tools=tool_list,
                    on_text=on_text,
                    on_thinking=on_thinking,
                )

                # Print newline after streaming output if any was streamed
                if not content_first or not thinking_first:
                    print()
            except Exception as e:
                # Use structured error handling
                from .utils.error_handler import LLMErrorClassifier, format_llm_error

                llm_error = LLMErrorClassifier.classify(e)

                if llm_error.is_retryable and llm_error.retry_after:
                    print(f"\n{Colors.BRIGHT_YELLOW}Rate limited. Waiting {llm_error.retry_after}s before returning...{Colors.RESET}")

                error_msg = format_llm_error(e)
                print(f"\n{error_msg}")
                return f"LLM call failed: {llm_error.user_guidance}"

            # Accumulate API reported token usage
            if response.usage:
                self.api_total_tokens = response.usage.total_tokens

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
                return response.content

            # Execute tool calls (parallel if M2.7)
            parallel_enabled = self.is_m27 and self.m27_config.get("enable_parallel_tool_calls", True)
            max_concurrent = self.m27_config.get("max_concurrent_tools", 5) if self.is_m27 else 1

            if parallel_enabled and len(response.tool_calls) > 1:
                results = await self._execute_tools_parallel(response.tool_calls, max_concurrent)
            else:
                results = await self._execute_tools_sequential(response.tool_calls)

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
            print(f"{Colors.BRIGHT_GREEN}✓ Result:{Colors.RESET} {text}")
        else:
            print(f"{Colors.BRIGHT_RED}✗ Error:{Colors.RESET} {Colors.RED}{result.error}{Colors.RESET}")

    async def _execute_single_tool(self, tool_call) -> tuple:
        """Execute a single tool and return (tool_call, tool_msg)."""
        tool_call_id = tool_call.id
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        # Plan mode: block write tools
        if self.mode == AgentMode.PLAN and function_name in self.WRITE_TOOLS:
            result = ToolResult(
                success=False, content="",
                error=f"Blocked in PLAN mode (read-only). Switch to /mode agent to use {function_name}.",
            )
            self._print_tool_call(function_name, arguments)
            self._print_tool_result(result)
            tool_msg = Message(
                role="tool",
                content=f"Error: {result.error}",
                tool_call_id=tool_call_id,
                name=function_name,
            )
            return (tool_call, tool_msg)

        self._print_tool_call(function_name, arguments)

        # YOLO mode: auto-approve, Agent mode: needs confirmation
        if self.mode == AgentMode.AGENT and not self._check_approved(function_name):
            result = ToolResult(
                success=False, content="",
                error=f"Tool call rejected by user. Type 'y' to approve, or switch to /mode yolo for auto-approve.",
            )
            self._print_tool_result(result)
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
                error_detail = f"{type(e).__name__}: {str(e)}"
                error_trace = traceback.format_exc()
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"Tool execution failed: {error_detail}\n\nTraceback:\n{error_trace}",
                )

        self.logger.log_tool_result(
            tool_name=function_name,
            arguments=arguments,
            result_success=result.success,
            result_content=result.content if result.success else None,
            result_error=result.error if not result.success else None,
        )

        self._print_tool_result(result)

        tool_msg = Message(
            role="tool",
            content=result.content if result.success else f"Error: {result.error}",
            tool_call_id=tool_call_id,
            name=function_name,
        )
        return (tool_call, tool_msg)

    async def _execute_tools_sequential(self, tool_calls: list) -> list[tuple]:
        """Execute tools one at a time."""
        results = []
        for tc in tool_calls:
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                return results
            tool_call, tool_msg = await self._execute_single_tool(tc)
            self.messages.append(tool_msg)
            results.append((tool_call, tool_msg))
        return results

    async def _execute_tools_parallel(self, tool_calls: list, max_concurrent: int = 5) -> list[tuple]:
        """Execute tools in parallel using a semaphore to limit concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_execute(tc):
            async with semaphore:
                return await self._execute_single_tool(tc)

        # Print all tool headers first
        for tc in tool_calls:
            self._print_tool_call(tc.function.name, tc.function.arguments)

        # Execute all tools concurrently
        task_results = await asyncio.gather(*[bounded_execute(tc) for tc in tool_calls])

        # Append results in original order
        for tool_call, tool_msg in task_results:
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                return task_results
            self.messages.append(tool_msg)

        return task_results

    def _check_approved(self, function_name: str) -> bool:
        """Prompt user to approve a tool call in Agent mode.

        Returns True if approved, False if rejected.
        """
        if self.mode != AgentMode.AGENT:
            return True
        try:
            import threading
            result = [None]

            def get_input():
                result[0] = input(f"  {Colors.BRIGHT_YELLOW}Approve {function_name}? [Y/n/q]{Colors.RESET} ").strip().lower()

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=30)

            if result[0] is None:
                return False
            if result[0] in ("q", "quit"):
                return False
            if result[0] in ("n", "no"):
                return False
            return True
        except Exception:
            return True

    def set_mode(self, mode: AgentMode):
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
