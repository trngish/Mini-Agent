"""Core Agent implementation."""

from __future__ import annotations

# HARD RULE: All Git operations require explicit user consent
# Including but not limited to: git add, git commit, git push, git merge
# Violations will be treated as unauthorized
import asyncio
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from .core.agent_context import AgentContext
from .core.approval import ApprovalManager
from .core.error_recovery import ErrorRecoveryManager
from .core.execution_engine import ExecutionEngine
from .core.health_check import HealthChecker
from .core.self_healing import SelfHealingManager
from .core.message_manager import MessageManager
from .core.metrics import PerformanceMetrics
from .core.rate_limiter import RateLimiter
from .core.retry_handler import create_retry_handler
from .core.semantic_memory import SemanticMemory
from .core.step_runner import StepRunner
from .core.thinking_budget import ThinkingBudgetManager
from .core.token_tracker import TokenTracker
from .llm import LLMClient
from .logger import AgentLogger
from .schema import AgentMode, Message, ToolCall
from .schema.schema import WRITE_TOOLS
from .session import SessionManager
from .subagent import SubAgentResult
from .tools.base import Tool
from .utils import Colors
from .utils.context_cache import create_cache_for_workspace
from .utils.error_handler import LLMErrorClassifier, format_llm_error
from .utils.model_utils import get_token_limit_for_model, is_m27_model
from .utils.task_state import get_task_manager
from .utils.thinking_manager import ThinkingManager

logger = logging.getLogger(__name__)

# Constants - avoid magic numbers
STREAM_BUFFER_SIZE = int(os.environ.get("MINI_AGENT_STREAM_BUFFER_SIZE", "8"))
DEFAULT_ENCODING_NAME = "cl100k_base"


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
        token_limit: int = 80000,
        m27_config: dict[str, Any] | None = None,
        mode: AgentMode = AgentMode.YOLO,
    ):
        # Store mode first as it's used in context creation
        self.mode = mode
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_list = list(tools)
        self.max_steps = max_steps
        self.workspace_dir = Path(workspace_dir)
        self.cancel_event: asyncio.Event | None = None
        self._session_manager = SessionManager(workspace_dir=self.workspace_dir)
        self.write_tools = WRITE_TOOLS

        # M2.7 specific configuration
        self.m27_config = m27_config or {}
        model_name = getattr(llm_client, "model", "")
        self.is_m27 = is_m27_model(model_name)

        # Use unified token limit calculation
        self.token_limit = get_token_limit_for_model(
            model_name, self.m27_config.get("token_limit") if self.is_m27 else token_limit
        )

        # M2.7 supports up to 32K output tokens
        self.max_output_tokens = self.m27_config.get("max_output_tokens", 16384) if self.is_m27 else 8192

        # Max budget from config
        self._max_thinking_budget = self.m27_config.get("thinking_budget_tokens", 16384) if self.is_m27 else 0
        self.thinking_budget = self._max_thinking_budget

        # Optimization: batch size for parallel tool execution
        self._max_tools_per_call = self.m27_config.get("max_concurrent_tools", 20) if self.is_m27 else 3

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Inject workspace information into system prompt
        if "Current Workspace" not in system_prompt:
            workspace_info = (
                f"\n\n## Current Workspace\n"
                f"You are currently working in: `{self.workspace_dir.absolute()}`"
                f"\nAll relative paths will be resolved relative to this directory."
            )
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt

        # D11 FIX: Semantic memory for cross-session knowledge persistence
        self._semantic_memory = SemanticMemory(workspace_dir=self.workspace_dir)

        # Inject cross-session memories into system prompt
        memory_context = self._semantic_memory.get_context_for_injection(max_entries=6)
        if memory_context:
            self.system_prompt += "\n" + memory_context

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
                "\n\n## Mode: YOLO\nYou are in YOLO mode. All tool calls are auto-approved. Execute efficiently."
            ),
        }
        self.system_prompt += mode_instructions.get(self.mode, "")

        # Initialize logger
        self.logger = AgentLogger()

        # Token tracker with incremental estimation
        self._token_tracker = TokenTracker()

        # Auto-save session after each step
        self.auto_save = os.environ.get("MINI_AGENT_AUTO_SAVE", "true").lower() == "true"

        # Create AgentContext FIRST - this is the single source of truth for state
        # All core modules receive a reference to context instead of Agent (breaks circular deps)
        self._context = AgentContext(
            messages=[Message(role="system", content=self.system_prompt)],
            mode=self.mode,
            max_steps=self.max_steps,
            workspace_dir=self.workspace_dir,
            token_limit=self.token_limit,
            api_call_count=0,
            api_total_tokens=0,
            is_m27=self.is_m27,
            thinking_budget=self._max_thinking_budget,
            llm=self.llm,
            auto_save=self.auto_save,
            token_tracker=self._token_tracker,
            record_context_fn=self._record_context_internal,
        )

        # Now initialize core modules with AgentContext (no circular dependency)
        self._thinking_budget_manager = ThinkingBudgetManager(self._context)
        self._thinking_budget_manager.configure(self._max_thinking_budget, self.is_m27)

        self._error_recovery = ErrorRecoveryManager(self._context)
        self._metrics = PerformanceMetrics(self._context)
        self._retry_handler = create_retry_handler(self)
        self._approval_manager = ApprovalManager(mode=mode, write_tools=self.write_tools)
        self._health_checker = HealthChecker(self._context)
        self._rate_limiter = RateLimiter()

        # Error recovery and metrics are accessed via delegation methods

        # Stream buffering for performance optimization
        self._stream_buffer_thinking: list[str] = []
        self._stream_buffer_text: list[str] = []
        self._buffer_flush_threshold = STREAM_BUFFER_SIZE * 2

        # Execution engine
        self._execution_engine = ExecutionEngine(
            tools=self.tools,
            logger=self.logger,
            retry_handler=self._retry_handler,
            metrics=self._metrics,
            error_recovery=self._error_recovery,
            write_tools=self.write_tools,
            rate_limiter=self._rate_limiter,
        )

        # Context cache for reducing redundant file reads/searches
        # D14 FIX: Workspace-isolated cache - each workspace gets its own cache instance
        self._context_cache = create_cache_for_workspace(self.workspace_dir)

        # Warmup cache with frequently accessed files
        if os.environ.get("MINI_AGENT_CACHE_WARMUP", "true").lower() == "true":
            try:
                cached_count = self._context_cache.warmup(self.workspace_dir)
                if cached_count > 0:
                    print(f"{Colors.DIM}📦 Cache warmed with {cached_count} files{Colors.RESET}")
            except Exception as e:
                logger.debug("Cache warmup failed: %s", e)

        # MessageManager for summarization
        self._message_manager = MessageManager(self.token_limit)

        # Thinking manager to prevent context overflow from truncated thinking
        self._thinking_manager: ThinkingManager | None = None
        if self.is_m27:
            self._thinking_manager = ThinkingManager(max_thinking_tokens=80_000)

        # Performance: throttle health checks
        self._last_health_check_step = -1
        self._health_check_interval = 5
        self._loop_detection_streak = 0

        # Store last run result for session persistence
        self._last_result: str | None = None
        # Store analysis results for session persistence
        self._last_analysis: str | None = None

        # Self-healing engine: detects anomalies and auto-fixes source code.
        # Auto-detect source dir: check if mini_agent package is in this workspace.
        _heal_source = self.workspace_dir
        if (_heal_source / "mini_agent" / "agent.py").exists():
            pass  # workspace IS the project root (editable install case)
        elif (_heal_source.parent / "mini_agent" / "agent.py").exists():
            _heal_source = _heal_source.parent  # workspace is a subdir
        self._self_healing = SelfHealingManager(
            source_dir=_heal_source,
            llm_client=self.llm if os.environ.get("MINI_AGENT_AUTO_HEAL", "0") == "1" else None,
        )

    def _record_context_internal(self, content: str, category: str = "auto") -> None:
        """Internal method for AgentContext to record context."""
        self.record_context(content, category)

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self._context.add_message(Message(role="user", content=content))
        # Adaptively adjust thinking budget based on task complexity
        self._thinking_budget_manager.adjust(content)

    @property
    def messages(self) -> list[Message]:
        """Get message history via AgentContext."""
        return self._context.get_messages()

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        """Replace message history (backward-compatible assignment)."""
        self.replace_messages(value)

    def append_message(self, message: Message) -> None:
        """Append a message to history via AgentContext."""
        self._context.add_message(message)

    def replace_messages(self, messages: list[Message]) -> None:
        """Replace entire message history via AgentContext."""
        self._context.set_messages(messages)

    # Backward compatibility properties for api_call_count
    @property
    def api_call_count(self) -> int:
        """Get API call count via AgentContext."""
        return self._context.api_call_count

    @api_call_count.setter
    def api_call_count(self, value: int) -> None:
        """Set API call count via AgentContext."""
        self._context.api_call_count = value

    # Backward compatibility properties for api_total_tokens
    @property
    def api_total_tokens(self) -> int:
        """Get API total tokens via AgentContext."""
        return self._context.api_total_tokens

    @api_total_tokens.setter
    def api_total_tokens(self, value: int) -> None:
        """Set API total tokens via AgentContext."""
        self._context.api_total_tokens = value

    @property
    def consecutive_failures(self) -> int:
        """Get consecutive failure count via AgentContext."""
        return self._context.consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        """Set consecutive failure count via AgentContext."""
        self._context.consecutive_failures = value

    def record_context(self, content: str, category: str = "auto") -> None:
        """Automatically record important context without needing explicit tool call.

        This is called internally by the agent when it encounters important
        information worth remembering for future reference.

        Args:
            content: The context to record
            category: Category tag (default: "auto" for automatic recordings)
        """
        # Use session note tool if available
        note_tool = self.tools.get("record_note")
        if note_tool:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(note_tool.execute(content=content, category=category))
            except RuntimeError:
                self.logger.debug("No running event loop for background recording")
            except Exception as e:
                self.logger.debug(f"Background recording failed: {e}")

    def _check_cancelled(self) -> bool:
        """Check if agent execution has been cancelled.

        Returns:
            True if cancelled, False otherwise.
        """
        return bool(self.cancel_event is not None and self.cancel_event.is_set())

    def _cleanup_incomplete_messages(self) -> None:
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
            self.replace_messages(self.messages[:last_assistant_idx])
            print(f"{Colors.DIM}   Cleaned up {removed_count} incomplete message(s){Colors.RESET}")

    async def _summarize_messages(self) -> None:
        """Message history summarization: delegate to MessageManager."""
        old_token_count = self._context.api_total_tokens
        new_messages = await self._message_manager.summarize_messages(self.messages, self.api_total_tokens, logger)
        if new_messages is not self.messages:
            self.replace_messages(new_messages)
            self._token_tracker.invalidate_cache()
            # Self-healing: record token pressure when summarization triggers
            self._self_healing.record_anomaly(
                "token_pressure", 0.5,
                {"api_tokens": old_token_count, "limit": self.token_limit}, 0,
            )

    async def run(self, cancel_event: asyncio.Event | None = None) -> str:
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

        # Initialize metrics session tracking
        self._metrics.set_session_id(str(id(self)))

        step = 0
        run_start_time = perf_counter()
        step_runner = StepRunner(self, run_start_time)
        self._loop_detection_streak = 0

        task_mgr = get_task_manager()
        if task_mgr.current_task:
            task_mgr.end_task()
        last_user = next(
            (m.content for m in reversed(self.messages) if m.role == "user" and m.content),
            "agent-run",
        )
        task_mgr.start_task(str(id(self)), last_user[:200], max_steps=self.max_steps)

        while step < self.max_steps:
            task_mgr.increment_steps()

            # Self-healing: tick and decay anomaly scores each step
            self._self_healing.tick(step)
            should_heal, heal_reason = self._self_healing.should_heal(step)
            if should_heal:
                await self._trigger_self_healing(step, heal_reason)

            should_stop, stop_reason = task_mgr.check_should_stop()
            if should_stop:
                stop_msg = f"Task stopped: {stop_reason}"
                self._last_result = stop_msg
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {stop_msg}{Colors.RESET}")
                return stop_msg

            # Check for cancellation at start of each step
            if self._check_cancelled():
                # D1 FIX: Flush any buffered streaming content before cancelling.
                # Without this, partially-streamed assistant response content is
                # lost and the user never sees what the AI was about to say.
                if self._stream_buffer_text:
                    print("".join(self._stream_buffer_text), end="", flush=True)
                    self._stream_buffer_text = []
                if self._stream_buffer_thinking:
                    print(f"{Colors.DIM}{''.join(self._stream_buffer_thinking)}{Colors.RESET}", end="", flush=True)
                    self._stream_buffer_thinking = []
                self._cleanup_incomplete_messages()
                cancel_msg = "Task cancelled by user."
                self._last_result = cancel_msg
                # E4 FIX: Log cancellation for audit trail
                logger.info("Task cancelled by user")
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                return cancel_msg

            step_start_time = perf_counter()
            # Check and summarize message history to prevent context overflow
            await self._summarize_messages()

            # Step header - unified single print for performance
            step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}Step {step + 1}/{self.max_steps}{Colors.RESET}"
            box_width = 44
            pad = box_width - len(f"  Step {step + 1}/{self.max_steps}") - 1
            print(
                f"\n  {Colors.DIM}╭{'─' * box_width}╮{Colors.RESET}\n"
                f"  {Colors.DIM}│{Colors.RESET}  {step_text}{' ' * max(0, pad)}{Colors.DIM}│{Colors.RESET}\n"
                f"  {Colors.DIM}╰{'─' * box_width}╯{Colors.RESET}"
            )

            # Get tool list for LLM call (cached during session)
            tool_list = self.tool_list

            # Log LLM request and call LLM with Tool objects directly
            self.logger.log_request(messages=self.messages, tools=tool_list)

            try:
                # Track streaming state for correct ordering
                thinking_started = False
                text_pending: list[str] = []

                def _flush_thinking_buffer() -> None:
                    """Flush buffered thinking output."""
                    if self._stream_buffer_thinking:
                        output = "".join(self._stream_buffer_thinking)
                        print(f"{Colors.DIM}{output}{Colors.RESET}", end="", flush=True)
                        self._stream_buffer_thinking = []

                def _flush_text_buffer() -> None:
                    """Flush buffered text output."""
                    if self._stream_buffer_text:
                        output = "".join(self._stream_buffer_text)
                        print(output, end="", flush=True)
                        self._stream_buffer_text = []

                def on_thinking(text: str) -> None:
                    nonlocal thinking_started, text_pending  # noqa: B023
                    if not thinking_started:
                        print(f"\n  {Colors.BOLD}{Colors.MAGENTA}🧠 Think{Colors.RESET}")
                        thinking_started = True
                    # FIX: Move text_pending into text_buffer but DON'T flush.
                    # Text must wait until ALL thinking is complete before display.
                    while text_pending:  # noqa: B023
                        pending = text_pending.pop(0)  # noqa: B023
                        self._stream_buffer_text.append(pending)
                    # Stream thinking content progressively
                    self._stream_buffer_thinking.append(text)
                    if len(self._stream_buffer_thinking) >= self._buffer_flush_threshold:
                        _flush_thinking_buffer()

                def on_text(text: str) -> None:  # noqa: B023
                    nonlocal thinking_started, text_pending  # noqa: B023
                    if thinking_started:  # noqa: B023
                        # FIX: Accumulate text silently while thinking is in progress.
                        # Text will only be displayed after thinking is fully complete.
                        self._stream_buffer_text.append(text)
                    else:
                        text_pending.append(text)  # noqa: B023

                response = await self.llm.generate(
                    messages=self.messages,
                    tools=tool_list,
                    on_text=on_text,
                    on_thinking=on_thinking,
                )

                # Print assistant header. Flush thinking FIRST, then text.
                # This guarantees the user sees all thinking before any result text.
                if thinking_started:
                    # Drain remaining text_pending into buffer (silently)
                    while text_pending:
                        pending = text_pending.pop(0)
                        self._stream_buffer_text.append(pending)
                    # Flush all thinking content first
                    _flush_thinking_buffer()
                    # Then print header and flush all accumulated text
                    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                    _flush_text_buffer()
                    print()
                elif text_pending:
                    # No thinking at all - just print assistant header and text
                    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                    while text_pending:
                        pending = text_pending.pop(0)
                        self._stream_buffer_text.append(pending)
                    _flush_text_buffer()
                    print()

            except Exception as e:
                # Use structured error handling
                llm_error = LLMErrorClassifier.classify(e)

                if llm_error.is_retryable and llm_error.retry_after:
                    print(
                        f"\n{Colors.BRIGHT_YELLOW}Rate limited."
                        f" Waiting {llm_error.retry_after}s before returning...{Colors.RESET}"
                    )

                error_msg = format_llm_error(e)
                print(f"\n{error_msg}")
                # Self-healing: record LLM error anomaly
                self._self_healing.record_anomaly(
                    "llm_error_pattern", 0.7,
                    {"error": str(e)[:200], "step": step}, step,
                )
                self._last_result = f"LLM call failed: {llm_error.user_guidance}"
                return self._last_result

            step_runner.process_response(response, step)

            health_issues = step_runner.check_health(step)
            if health_issues:
                for issue in health_issues:
                    print(f"  {Colors.YELLOW}⚠️  {issue}{Colors.RESET}")
                # Self-healing: record health anomaly
                self._self_healing.record_anomaly(
                    "health_issues", min(1.0, 0.3 * len(health_issues)),
                    {"count": len(health_issues), "step": step}, step,
                )

            step_runner.prune_thinking()

            # Detect loop patterns early — stop after repeated repetition
            if step > 2 and step_runner.detect_loop(response):
                self._loop_detection_streak += 1
                self._context.api_total_tokens = self._token_tracker.estimate_tokens(self.messages)
                if self._loop_detection_streak >= 2:
                    loop_msg = response.content or "Task stopped: repetitive analysis pattern detected."
                    self._last_result = loop_msg
                    # Self-healing: record loop detection anomaly
                    self._self_healing.record_anomaly(
                        "loop_detection", 0.85,
                        {"msg": loop_msg[:100], "step": step}, step,
                    )
                    if task_mgr.current_task:
                        task_mgr.current_task.analysis_complete = True
                        task_mgr.end_task()
                    # E3 FIX: Log loop detection for debugging
                    logger.warning("Loop pattern detected twice, stopping agent")
                    print(f"\n{Colors.BRIGHT_YELLOW}⚠️  Stopping: loop pattern detected twice{Colors.RESET}")
                    return loop_msg
            else:
                self._loop_detection_streak = 0

            if step_runner.is_complete(response):
                if task_mgr.current_task:
                    task_mgr.end_task()
                step_runner.print_completion_summary(step, step_start_time)
                self._last_result = response.content
                # D11 FIX: Extract semantic memories on completion
                try:
                    memories = self._semantic_memory.extract_from_session(
                        self.messages, f"completed_step_{step}"
                    )
                    if memories:
                        self._semantic_memory.add_entries(memories)
                except Exception:
                    pass
                self._session_manager.save(
                    self.messages,
                    f"completed_step_{step}",
                    result=self._last_result,
                    analysis=self._last_analysis,
                    state=self._get_runtime_state(),
                )
                return self._last_result

            assert response.tool_calls is not None
            parallel_enabled = self.is_m27 and self.m27_config.get("enable_parallel_tool_calls", True)
            max_concurrent = self.m27_config.get("max_concurrent_tools", 5) if self.is_m27 else 1

            results = await self._execution_engine.execute_tools(
                response.tool_calls,
                max_concurrent,
                parallel_enabled,
                self.mode,
                self._check_approved,
            )

            if len(results) < len(response.tool_calls):
                optimized_calls = [tc for tc, _ in results]
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    thinking=response.thinking,
                    tool_calls=optimized_calls,
                )
                self._context.replace_last_message(assistant_msg)

            # Append tool messages and handle cancellation
            for _tool_call, tool_msg in results:
                if self._check_cancelled():
                    self._cleanup_incomplete_messages()
                    self._last_result = "Task cancelled by user."
                    return self._last_result
                self._context.add_message(tool_msg)

            step_elapsed = perf_counter() - step_start_time
            self._metrics.record_step_duration(step_elapsed)

            step_runner.print_step_timing(step, step_start_time)
            step_runner.auto_save(step)

            step += 1

        # Max steps reached
        error_msg = f"Task couldn't be completed after {self.max_steps} steps."
        print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {error_msg}{Colors.RESET}")
        # Auto-save on max steps (for potential resume)
        if self.auto_save:
            try:
                sid = self._session_manager.save(self.messages, f"max_steps_{self.max_steps}", state=self._get_runtime_state())
                print(f"  {Colors.DIM}💾 Session auto-saved: {sid}{Colors.RESET}")
            except Exception as e:
                print(f"  {Colors.DIM}⚠️  Auto-save failed: {e}{Colors.RESET}")
        self._last_result = error_msg
        return error_msg

    async def execute_single_tool(self, tool_call: ToolCall) -> tuple[ToolCall, Message]:
        """Execute a single tool with Agent-specific behavior (print, log, approve)."""
        return await self._execution_engine._execute_single_tool(tool_call, self.mode, self._check_approved)

    async def execute_tools_sequential(self, tool_calls: list[ToolCall]) -> list[tuple[ToolCall, Message]]:
        """Execute tools one at a time."""
        return await self._execution_engine._execute_sequential(tool_calls, self.mode, self._check_approved)

    async def execute_tools_parallel(
        self, tool_calls: list[ToolCall], max_concurrent: int = 5
    ) -> list[tuple[ToolCall, Message]]:
        """Execute tools in parallel using a semaphore to limit concurrency."""
        return await self._execution_engine._execute_parallel(
            tool_calls, max_concurrent, self.mode, self._check_approved
        )

    def _check_approved(self, function_name: str) -> bool:
        """Prompt user to approve a tool call in Agent mode.

        Returns True if approved, False if rejected.
        """
        return self._approval_manager.is_approved(function_name)

    def set_mode(self, mode: AgentMode) -> None:
        """Switch agent mode."""
        old_mode = self.mode
        self.mode = mode
        self._context.mode = mode  # Sync to context for health check visibility
        self._approval_manager.mode = mode
        print(f"{Colors.GREEN}✅ Mode switched: {old_mode.value} → {mode.value}{Colors.RESET}")

    async def _trigger_self_healing(self, step: int, reason: str) -> None:
        """Trigger self-healing diagnosis and optional auto-fix.

        Called when anomaly scores exceed thresholds. Runs diagnosis on
        source code and optionally applies fixes if auto-heal is enabled.

        Args:
            step: Current agent step.
            reason: Why healing was triggered.
        """
        status = self._self_healing.get_status()
        top_cats = self._self_healing.get_top_anomaly_categories(3)

        print(
            f"\n{Colors.BRIGHT_CYAN}🩺 Self-healing triggered (step {step}, {reason}){Colors.RESET}"
        )
        print(f"  {Colors.DIM}Top anomalies: {', '.join(f'{c}({s:.2f})' for c, s in top_cats)}{Colors.RESET}")

        if not status["auto_heal_enabled"]:
            print(
                f"  {Colors.DIM}Auto-heal disabled. Report only."
                f" Set MINI_AGENT_AUTO_HEAL=1 to enable.{Colors.RESET}"
            )
            print(self._self_healing.get_healing_report())
            return

        try:
            diagnosis = await self._self_healing.diagnose(top_cats)
            fixes = diagnosis.get("suggested_fixes", [])

            if not fixes:
                print(f"  {Colors.DIM}No actionable fixes identified.{Colors.RESET}")
                return

            print(f"  {Colors.BRIGHT_YELLOW}Found {len(fixes)} potential fix(es):{Colors.RESET}")
            for i, fix in enumerate(fixes):
                desc = fix.get("description", fix.get("file", "unknown"))[:80]
                print(f"  {Colors.DIM}  [{i + 1}] {desc}{Colors.RESET}")

            # Apply fixes (with approval check in non-YOLO modes)
            applied = 0
            anomaly_ids = [a.id for a in self._self_healing._anomalies[-5:]]

            for fix in fixes:
                file_name = fix.get("file", "")
                description = fix.get("description", "auto-fix")
                old_str = fix.get("old_str", "")
                new_str = fix.get("new_str", "")

                if not file_name or not old_str:
                    continue

                # Approval gate in Agent mode
                if self.mode == AgentMode.AGENT:
                    print(
                        f"  {Colors.BRIGHT_YELLOW}Apply fix to {file_name}?"
                        f" {Colors.DIM}({description[:60]}){Colors.RESET}"
                    )
                    if not self._check_approved(f"heal:{file_name}"):
                        print(f"  {Colors.DIM}  Skipped (not approved){Colors.RESET}")
                        continue

                result = self._self_healing.apply_fix(
                    file_name=file_name,
                    description=description,
                    old_str=old_str,
                    new_str=new_str,
                    anomaly_ids=anomaly_ids,
                )

                if result:
                    applied += 1
                    print(
                        f"  {Colors.BRIGHT_GREEN}✅ Fixed: {file_name} - {description[:60]}"
                        f"{Colors.RESET}"
                    )
                    print(f"  {Colors.DIM}  Backup: {Path(result.backup_path).name}{Colors.RESET}")

            if applied > 0:
                print(
                    f"  {Colors.BRIGHT_GREEN}🩺 Applied {applied} fix(es)."
                    f" Changes take effect on next restart.{Colors.RESET}"
                )
            else:
                print(f"  {Colors.DIM}No fixes applied.{Colors.RESET}")

        except Exception as e:
            logger.warning("Self-healing failed: %s", e)
            print(f"  {Colors.YELLOW}⚠️  Self-healing error: {e}{Colors.RESET}")

    def save_session(self, label: str = "") -> str:
        """Save current session including last result and analysis. Returns session ID."""
        # D11 FIX: Extract semantic memories before saving
        try:
            memories = self._semantic_memory.extract_from_session(self.messages, label or "manual_save")
            if memories:
                added = self._semantic_memory.add_entries(memories)
                if added > 0:
                    print(f"  {Colors.DIM}🧠 Extracted {added} semantic memories{Colors.RESET}")
        except Exception:
            pass
        return self._session_manager.save(self.messages, label=label, result=self._last_result, state=self._get_runtime_state())

    def load_session(self, session_id: str) -> bool:
        """Load a saved session including result, analysis, and runtime state. Returns True on success."""
        messages, result, state = self._session_manager.load(session_id)
        if messages is None:
            return False
        self.replace_messages(messages)
        self._last_result = result  # Restore last result
        # Restore last analysis from metadata if available
        self._last_analysis = self._session_manager.load_analysis(session_id)
        # D3 FIX: Restore full runtime state (thinking budget, loop counter, etc.)
        self._restore_runtime_state(state)
        # Sync AgentContext state after session restore
        self._sync_context_state()
        return True

    def get_last_result(self) -> str | None:
        """Get the result from the last run."""
        return self._last_result

    def set_analysis_result(self, analysis: str) -> None:
        """Set analysis result for session persistence.

        This allows the agent to remember analysis results across sessions.
        Call this after completing an analysis task.

        D5 FIX: Analysis is now persisted to the most recent session file
        immediately, preventing data loss on crash.
        """
        self._last_analysis = analysis
        # D5: Auto-persist to most recent session
        try:
            latest_sid = self._session_manager.get_latest_session_id()
            if latest_sid:
                self._session_manager.save_analysis(latest_sid, analysis)
        except Exception:
            pass  # Non-critical, don't block the agent

    def get_analysis_result(self) -> str | None:
        """Get the last analysis result."""
        return self._last_analysis

    def _get_runtime_state(self) -> dict[str, Any]:
        """D3 FIX: Serialize all critical runtime state for session persistence.

        Captures state that would otherwise be lost on session restore:
        api_total_tokens, thinking_budget, loop_detection_streak, etc.
        """
        return {
            "api_call_count": self._context.api_call_count,
            "api_total_tokens": self._context.api_total_tokens,
            "thinking_budget": self.thinking_budget,
            "loop_detection_streak": self._loop_detection_streak,
            "consecutive_failures": self._context.consecutive_failures,
            "last_auto_save_step": self._context.last_auto_save_step,
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
        }

    def _restore_runtime_state(self, state: dict[str, Any]) -> None:
        """D3 FIX: Restore runtime state from a saved session."""
        if not state:
            return
        self._context.api_call_count = state.get("api_call_count", self._context.api_call_count)
        self._context.api_total_tokens = state.get("api_total_tokens", self._context.api_total_tokens)
        self.thinking_budget = state.get("thinking_budget", self.thinking_budget)
        self._loop_detection_streak = state.get("loop_detection_streak", 0)
        self._context.consecutive_failures = state.get("consecutive_failures", 0)
        self._context.last_auto_save_step = state.get("last_auto_save_step", 0)
        print(
            f"{Colors.DIM}  State restored: {state.get('api_call_count', 0)} API calls,"
            f" {state.get('api_total_tokens', 0)} tokens,"
            f" budget={state.get('thinking_budget', 'N/A')}{Colors.RESET}"
        )

    def _sync_context_state(self) -> None:
        """Sync AgentContext state after session restore.

        Rebuilds token count and other derived state from messages
        to ensure consistency after session restore.
        """
        # Estimate total tokens from restored messages
        estimated_tokens = self._context.estimate_tokens()
        self._context.api_total_tokens = estimated_tokens

        # Reset API call count to match restored messages
        api_calls = sum(1 for m in self._context.get_messages() if m.role == "assistant")
        self._context.api_call_count = api_calls

        # Invalidate token tracker cache to force recalculation
        self._token_tracker.invalidate_cache()

        # Log restored session info
        self.logger.debug(f"Session restored: {len(self._context.get_messages())} messages, ~{estimated_tokens} tokens")

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions."""
        return self._session_manager.list_sessions()

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()

    def get_status(self) -> dict[str, Any]:
        """Get agent status report for self-diagnosis."""
        return self._health_checker.get_status()

    def get_status_report(self) -> str:
        """Generate a human-readable status report."""
        return self._health_checker.get_status_report()

    def _check_health(self) -> list[str]:
        """Self-health check after each step. Returns list of issues found."""
        return self._health_checker.check().issues

    def get_error_patterns(self) -> dict[str, Any]:
        """Get error pattern analysis for debugging and learning.

        Returns:
            Dict with error patterns by tool and recent error history
        """
        return self._error_recovery.get_patterns()

    def get_suggestions(self) -> list[str]:
        """Get suggestions based on current agent state.

        Analyzes agent status and provides actionable suggestions.
        """
        return self._error_recovery.get_suggestions()

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the current session.

        Returns:
            Dict with timing metrics for steps, tools, and API calls
        """
        return self._metrics.get_metrics()

    async def dispatch_sub_agents(
        self,
        tasks: list[str],
        max_concurrent: int = 3,
        _system_prompt: str = "You are a helpful assistant. Complete the assigned task concisely.",  # noqa: ARG002
    ) -> list[SubAgentResult]:
        """Dispatch multiple sub-agents to work in parallel on independent tasks.

        This enables the agent to "clone itself" and tackle multiple problems
        simultaneously, then synthesize the results.

        Args:
            tasks: List of task descriptions for sub-agents
            max_concurrent: Maximum number of concurrent sub-agents
            system_prompt: Custom system prompt for sub-agents

        Returns:
            List of SubAgentResult objects with task, content, success, elapsed, error
        """
        from .subagent import run_sub_agents as run_subs

        # Get a copy of tools for sub-agents
        tools = self.tool_list

        # Run sub-agents in parallel
        results = await run_subs(
            llm_client=self.llm,
            tasks=tasks,
            tools=tools,
            max_concurrent=max_concurrent,
        )

        # Log the dispatch
        print(
            f"\n{Colors.BRIGHT_CYAN}🔄 Dispatched {len(tasks)} sub-agents ({max_concurrent} concurrent){Colors.RESET}"
        )
        successful = sum(1 for r in results if r.success)
        print(f"{Colors.BRIGHT_GREEN}✅ {successful}/{len(tasks)} succeeded{Colors.RESET}")

        return results

    def cleanup(self) -> None:
        """Clean up resources held by the agent.

        Should be called when agent is no longer needed to ensure
        proper cleanup of background processes and connections.
        """
        # Flush logger
        if hasattr(self, "logger"):
            self.logger.flush()

        # Clear cancel event
        self.cancel_event = None

        # Reset token cache
        self._token_tracker.invalidate_cache()

        # Note: Do NOT clear _encoder_cache here — it is a module-level
        # shared cache. Clearing it would affect all Agent instances.

        # Clean up background shells
        from .tools.bash_background import BackgroundShellManager

        try:
            loop = asyncio.get_running_loop()
            # Create cleanup task, fire and forget within same event loop
            # Use create_task to schedule cleanup without blocking
            loop.create_task(BackgroundShellManager.cleanup_all())
            # Don't await - let it run in background, we can't block here
            # The task will complete even if we return
        except RuntimeError:
            logger.warning("No running event loop during cleanup - cleanup may be skipped")
        except Exception as e:
            logger.warning("Event loop cleanup failed: %s", e)
