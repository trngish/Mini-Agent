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
from .core.message_manager import MessageManager
from .core.metrics import PerformanceMetrics
from .core.rate_limiter import RateLimiter
from .core.retry_handler import create_retry_handler
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
from .utils.context_cache import get_context_cache
from .utils.error_handler import LLMErrorClassifier, format_llm_error
from .utils.model_utils import get_token_limit_for_model, is_m27_model
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
        self._context_cache = get_context_cache()

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
        new_messages = await self._message_manager.summarize_messages(self.messages, self.api_total_tokens, logger)
        if new_messages is not self.messages:
            self.replace_messages(new_messages)
            self._token_tracker.invalidate_cache()

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

        step = 0
        run_start_time = perf_counter()
        step_runner = StepRunner(self, run_start_time)

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

                def on_thinking(text: str) -> None:
                    nonlocal thinking_started, text_pending  # noqa: B023
                    if not thinking_started:
                        print(f"\n  {Colors.BOLD}{Colors.MAGENTA}🧠 Think{Colors.RESET}")
                        thinking_started = True
                    print(f"{Colors.DIM}{text}{Colors.RESET}", end="", flush=True)
                    while text_pending:  # noqa: B023
                        pending = text_pending.pop(0)  # noqa: B023
                        print(pending, end="", flush=True)  # noqa: B023

                def on_text(text: str) -> None:  # noqa: B023
                    nonlocal thinking_started, text_pending  # noqa: B023
                    if thinking_started:  # noqa: B023
                        print(text, end="", flush=True)
                    else:
                        text_pending.append(text)  # noqa: B023

                response = await self.llm.generate(
                    messages=self.messages,
                    tools=tool_list,
                    on_text=on_text,
                    on_thinking=on_thinking,
                )

                # Print assistant header and flush any pending text (or just text if no thinking)
                if thinking_started:
                    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                    # Flush any text that arrived before thinking
                    while text_pending:
                        pending = text_pending.pop(0)
                        print(pending, end="", flush=True)
                    print()
                elif text_pending:
                    # No thinking at all - just print assistant header and text
                    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                    while text_pending:
                        pending = text_pending.pop(0)
                        print(pending, end="", flush=True)
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
                return f"LLM call failed: {llm_error.user_guidance}"

            step_runner.process_response(response, step)

            health_issues = step_runner.check_health(step)
            if health_issues:
                for issue in health_issues:
                    print(f"  {Colors.YELLOW}⚠️  {issue}{Colors.RESET}")

            step_runner.prune_thinking()

            if step_runner.is_complete(response):
                step_runner.print_completion_summary(step, step_start_time)
                step_runner.auto_save(step, prefix="completed_step")
                return response.content

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
                    return "Task cancelled by user."
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
                sid = self._session_manager.save(self.messages, f"max_steps_{self.max_steps}")
                print(f"  {Colors.DIM}💾 Session auto-saved: {sid}{Colors.RESET}")
            except Exception as e:
                print(f"  {Colors.DIM}⚠️  Auto-save failed: {e}{Colors.RESET}")
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

    def save_session(self, label: str = "") -> str:
        """Save current session. Returns session ID."""
        return self._session_manager.save_session(self.messages, label=label)

    def load_session(self, session_id: str) -> bool:
        """Load a saved session. Returns True on success."""
        messages = self._session_manager.load_session(session_id)
        if messages is None:
            return False
        self.replace_messages(messages)
        return True

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
