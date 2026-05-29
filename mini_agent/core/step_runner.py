"""Step runner for managing individual agent loop steps.

Extracts the per-step logic from the Agent.run() loop into a
clean, testable class with clear responsibilities.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from ..schema import Message
from ..utils import Colors

if TYPE_CHECKING:
    from .agent_context import AgentContext
    from ..agent import Agent


class StepRunner:
    """Manages a single step in the agent execution loop.

    Responsibilities:
    - Process LLM response (add assistant message, log)
    - Health check (throttled)
    - Thinking content pruning
    - Tool execution delegation
    - Auto-save management
    - Step timing and metrics
    """

    def __init__(self, agent: Agent, run_start_time: float):
        self._agent = agent
        self._context = agent._context
        self._run_start_time = run_start_time

    def process_response(self, response: Any, _step: int) -> Message:
        """Process LLM response: create and append assistant message.

        Args:
            response: LLM response object
            step: Current step number

        Returns:
            The created assistant Message
        """
        self._context.api_call_count += 1
        if response.usage:
            self._context.api_total_tokens = response.usage.total_tokens

        tool_count = len(response.tool_calls) if response.tool_calls else 0
        print(
            f"\n  {Colors.DIM}📊 API Call #{self._context.api_call_count} | Tools: {tool_count} | "
            f"Thinking budget: {self._context.thinking_budget} | "
            f"Total tokens: {self._context.api_total_tokens:,}{Colors.RESET}"
        )

        self._agent.logger.log_response(
            content=response.content,
            thinking=response.thinking,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
        )

        assistant_msg = Message(
            role="assistant",
            content=response.content,
            thinking=response.thinking,
            tool_calls=response.tool_calls,
        )
        self._context.add_message(assistant_msg)
        return assistant_msg

    def check_health(self, step: int) -> list[str]:
        """Run health check if throttling interval allows.

        Args:
            step: Current step number

        Returns:
            List of health issues found (empty if check was skipped)
        """
        if (
            self._context.consecutive_failures > 0
            or step - self._agent._last_health_check_step >= self._agent._health_check_interval
        ):
            issues = self._agent._check_health()
            self._agent._last_health_check_step = step
            return issues
        return []

    def prune_thinking(self) -> int:
        """Prune thinking content if it exceeds threshold.

        Returns:
            Number of tokens freed
        """
        messages = self._context.get_messages()
        if self._agent._thinking_manager and len(messages) > 5:
            tokens_freed = self._agent._thinking_manager.prune_thinking(messages)
            if tokens_freed > 1000:
                print(
                    f"{Colors.DIM}🧠 Pruned {tokens_freed:,} thinking tokens to prevent context overflow{Colors.RESET}"
                )
                self._agent._token_tracker.invalidate_cache()
            return tokens_freed
        return 0

    def is_complete(self, response: Any) -> bool:
        """Check if the task is complete (no tool calls in response)."""
        return not response.tool_calls

    def print_completion_summary(self, step: int, step_start_time: float) -> None:
        """Print step and total timing summary."""
        step_elapsed = perf_counter() - step_start_time
        total_elapsed = perf_counter() - self._run_start_time
        print(
            f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s"
            f" (total: {total_elapsed:.2f}s){Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_GREEN}💰 Total API calls: {self._context.api_call_count} (per-call billing){Colors.RESET}")

    def auto_save(self, step: int, prefix: str = "auto_step") -> None:
        """Auto-save session if enabled and interval reached.

        Args:
            step: Current step number
            prefix: Save prefix for identification
        """
        if not self._context.auto_save:
            return
        if prefix == "auto_step" and step - self._context.last_auto_save_step < 3:
            return
        try:
            sid = self._agent._session_manager.save(self._context.get_messages(), f"{prefix}_{step}")
            print(f"  {Colors.DIM}💾 Session auto-saved: {sid}{Colors.RESET}")
            if prefix == "auto_step":
                self._context.last_auto_save_step = step
        except Exception as e:
            print(f"  {Colors.DIM}⚠️  Auto-save failed: {e}{Colors.RESET}")

    def print_step_timing(self, step: int, step_start_time: float) -> None:
        """Print step timing info."""
        step_elapsed = perf_counter() - step_start_time
        total_elapsed = perf_counter() - self._run_start_time
        print(f"\n  {Colors.DIM}✔  Step {step + 1}  ({step_elapsed:.1f}s | total: {total_elapsed:.1f}s){Colors.RESET}")