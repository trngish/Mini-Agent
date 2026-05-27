"""Core message handling and summarization logic."""

from __future__ import annotations

import logging
from typing import Any

from ..schema import Message
from ..utils import Colors
from ..utils.summary_manager import AdaptiveSummaryManager
from .token_tracker import TokenTracker

_logger = logging.getLogger(__name__)


class MessageManager:
    """Manages message history and summarization.

    Handles message history, token estimation, and automatic summarization
    when token limits are approached.
    """

    def __init__(self, token_limit: int):
        self.token_limit = token_limit
        self.messages: list[Message] = []
        self._token_tracker = TokenTracker()
        self._summary_manager = AdaptiveSummaryManager(token_limit)
        self._skip_next_token_check = False
        self._last_summary_quality = 1.0

    def initialize(self, system_prompt: str) -> None:
        """Initialize with system prompt."""
        self.messages = [Message(role="system", content=system_prompt)]

    def add_message(self, message: Message) -> None:
        """Add a message to history."""
        self.messages.append(message)

    def estimate_tokens(self) -> int:
        """Estimate total tokens for message history."""
        return self._token_tracker.estimate_tokens(self.messages)

    def should_summarize(self, api_total_tokens: int) -> tuple[bool, str]:
        """Check if summarization is needed.

        Returns:
            Tuple of (should_summarize, reason)
        """
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return False, ""

        estimated_tokens = self.estimate_tokens()
        should_summarize, reason = self._summary_manager.should_summarize(
            self.messages, estimated_tokens, api_total_tokens
        )
        return should_summarize, reason

    def mark_skip_next_check(self, quality: float = 1.0) -> None:
        """Mark to skip next token check after summary."""
        self._skip_next_token_check = True
        if self._summary_manager.should_skip_next_check(quality):
            self._skip_next_token_check = True

    def get_messages(self) -> list[Message]:
        """Get current message list."""
        return self.messages

    def replace_messages(self, messages: list[Message]) -> None:
        """Replace entire message list."""
        self.messages = messages

    async def summarize_messages(
        self,
        messages: list[Message],
        api_total_tokens: int,
        logger: Any,
        _max_truncation: int = 2000,
    ) -> list[Message]:
        """Summarize message history when tokens exceed limit.

        Strategy:
        - Keep all user messages (these are user intents)
        - Summarize content between each user-user pair (agent execution process)
        - If last round is still executing, also summarize
        - Structure: system -> user1 -> summary1 -> user2 -> summary2 -> ...

        Args:
            messages: Current message list (will be modified in place)
            api_total_tokens: Token count reported by last API response
            logger: Logger instance for warnings
            max_truncation: Maximum truncation length for summary content

        Returns:
            New message list after summarization (same list if no summarization needed)
        """
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return messages

        estimated_tokens = self._token_tracker.estimate_tokens(messages)

        should_summarize, reason = self._summary_manager.should_summarize(messages, estimated_tokens, api_total_tokens)

        if not should_summarize:
            return messages

        print(
            f"\n{Colors.BRIGHT_YELLOW}📊 Token usage - Local estimate: {estimated_tokens},"
            f" API reported: {api_total_tokens}, Limit: {self.token_limit}{Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_YELLOW}🔄 Triggering message summarization ({reason})...{Colors.RESET}")

        user_indices = [i for i, msg in enumerate(messages) if msg.role == "user" and i > 0]

        if len(user_indices) < 1:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  Insufficient messages, cannot summarize{Colors.RESET}")
            return messages

        new_messages = [messages[0]]
        summary_count = 0

        for i, user_idx in enumerate(user_indices):
            new_messages.append(messages[user_idx])

            user_content = messages[user_idx].content
            if isinstance(user_content, str) and len(user_content) > 5000:
                logger.warning(
                    "User message truncated from %d to 5000 chars during summarization",
                    len(user_content),
                )
                messages[user_idx].content = user_content[:5000] + "...[truncated]..."

            next_user_idx = user_indices[i + 1] if i < len(user_indices) - 1 else len(messages)

            execution_messages = messages[user_idx + 1 : next_user_idx]

            if execution_messages:
                tier = "medium"
                if ":" in reason and "tier=" in reason:
                    tier = reason.split("tier=")[1]
                elif reason.startswith("early_trigger"):
                    tier = "low"

                summary_config = self._summary_manager.get_summary_config(tier)
                summary_text = self._create_local_summary(
                    execution_messages,
                    i + 1,
                    preserve_ratio=summary_config["preserve_ratio"],
                    max_truncation=summary_config["max_truncation"],
                )
                if summary_text:
                    summary_message = Message(
                        role="user",
                        content=f"[Execution Summary {i + 1}]\n\n{summary_text}",
                    )
                    new_messages.append(summary_message)
                    summary_count += 1

        self._skip_next_token_check = True

        self._last_summary_quality = self._summary_manager.estimate_summary_quality(
            new_messages,
            str(len(new_messages) * 100),
        )

        if self._summary_manager.should_skip_next_check(self._last_summary_quality):
            self._skip_next_token_check = True

        new_tokens = self._token_tracker.estimate_tokens(new_messages)

        print(
            f"{Colors.BRIGHT_GREEN}✓ Summary completed, local tokens: {estimated_tokens} → {new_tokens}{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}  Structure: system + {len(user_indices)} user messages"
            f" + {summary_count} summaries{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}  Quality: {self._last_summary_quality:.2f} |"
            f" Note: API token count will update on next LLM call{Colors.RESET}"
        )

        return new_messages

    def _create_local_summary(
        self,
        messages: list[Message],
        round_num: int,
        preserve_ratio: float = 0.6,
        max_truncation: int = 1000,
    ) -> str:
        """Create summary locally without LLM call (saves tokens).

        Preserves more detail to reduce redundant LLM calls.

        Args:
            messages: List of messages to summarize
            round_num: Round number
            preserve_ratio: How much content to preserve (0-1)
            max_truncation: Maximum characters to truncate to

        Returns:
            Summary text
        """
        if not messages:
            return ""

        lines = [f"Round {round_num}:"]
        tool_calls_count = 0
        tool_results_count = 0
        assistant_responses = []

        for msg in messages:
            if msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    if len(content) > max_truncation:
                        content = content[:max_truncation] + "..."
                    assistant_responses.append(content)
                if msg.tool_calls:
                    tool_details = []
                    for tc in msg.tool_calls:
                        args_str = str(tc.function.arguments)
                        args_max = max(150, int(150 * preserve_ratio * 1.5))
                        if len(args_str) > args_max:
                            args_str = args_str[:args_max] + "..."
                        tool_details.append(f"{tc.function.name}({args_str})")
                    lines.append(f"  Tools called: {', '.join(tool_details)}")
                    tool_calls_count += len(msg.tool_calls)
            elif msg.role == "tool":
                tool_results_count += 1
                result = msg.content if isinstance(msg.content, str) else str(msg.content)
                result_max = max(1500, int(1500 * preserve_ratio * 1.5))
                if len(result) > result_max:
                    result = result[:result_max] + "..."
                lines.append(f"  Result: {result}")

        if not tool_calls_count and assistant_responses:
            lines.append(f"  Response: {assistant_responses[0]}")

        if tool_calls_count or tool_results_count:
            lines.append(f"  Stats: {tool_calls_count} tool(s), {tool_results_count} result(s)")

        return "\n".join(lines)
