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

        # D12 FIX: Detect summary generation depth to prevent cascading degeneration.
        # Each time a summary is summarized again, we need higher preservation.
        summary_generation = self._detect_summary_generation(messages)
        if summary_generation > 0:
            print(
                f"{Colors.BRIGHT_YELLOW}🔍 Summary generation {summary_generation} detected"
                f" - applying anti-degeneration boost{Colors.RESET}"
            )

        new_messages = [messages[0]]
        summary_count = 0

        # D12 FIX: Anti-degeneration minimum preservation ratios
        # Generation 0 (fresh): normal ratios
        # Generation 1: +20% boost
        # Generation 2+: +35% boost with golden fact preservation
        gen_boost = min(0.35, summary_generation * 0.15)

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
                is_last_round = (i == len(user_indices) - 1)

                # CRITICAL FIX: For the most recent round, preserve assistant messages
                # verbatim instead of summarizing them. The user's follow-up instruction
                # almost always references the AI's most recent output. Summarizing it
                # away causes the "AI forgets what it just said" problem.
                if is_last_round and self._should_preserve_last_round(execution_messages):
                    for msg in execution_messages:
                        if msg.role in ("assistant", "tool"):
                            new_messages.append(msg)
                    summary_count += 1  # Count as "handled" even though preserved
                    continue

                tier = "medium"
                if ":" in reason and "tier=" in reason:
                    # D12 FIX: Parse tier from reason, handle degen_w suffix
                    tier_part = reason.split("tier=")[1]
                    tier = tier_part.split(":")[0]  # Strip :degen_w{N} suffix
                elif reason.startswith("early_trigger"):
                    tier = "low"

                summary_config = self._summary_manager.get_summary_config(tier)
                # D12 FIX: Apply anti-degeneration boost
                # Higher generation → higher preservation to prevent cascading loss
                boosted_ratio = min(1.0, summary_config["preserve_ratio"] + gen_boost)
                boosted_truncation = summary_config["max_truncation"]
                if summary_generation > 0:
                    # Don't truncate as aggressively when re-summarizing
                    boosted_truncation = min(summary_config["max_truncation"] * 2, 3000)
                summary_text = self._create_local_summary(
                    execution_messages,
                    i + 1,
                    preserve_ratio=boosted_ratio,
                    max_truncation=boosted_truncation,
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

    @staticmethod
    def _detect_summary_generation(messages: list[Message]) -> int:
        """D12 FIX: Detect how many generations of summarization have occurred.

        Scans for [Execution Summary] markers in user messages to determine
        the depth of the summary chain. Each layer of summarization adds
        another generation.

        Returns:
            Summary generation depth (0 = fresh, 1+ = re-summarized).
        """
        max_gen = 0
        for msg in messages:
            if msg.role == "user" and msg.content and isinstance(msg.content, str):
                content = msg.content
                # Count nested summary markers
                if "[Execution Summary" in content:
                    gen = content.count("[Execution Summary")
                    max_gen = max(max_gen, gen)
                if "Stats:" in content and "tool(s)" in content:
                    # Also count re-summarized stats
                    gen = content.count("tool(s)")
                    max_gen = max(max_gen, gen)
        return max_gen

    @staticmethod
    def _should_preserve_last_round(execution_messages: list[Message]) -> bool:
        """Determine if the last round's assistant messages should be preserved verbatim.

        When the AI has just provided analysis/suggestions that the user will
        reference in their next message, summarizing those messages causes
        the AI to "forget" what it just said.

        We preserve the last round if any assistant message has substantial
        text content (more than 100 chars) — this indicates meaningful output
        the user is likely to follow up on.
        """
        for msg in execution_messages:
            if msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content) if msg.content else ""
                if len(content) > 100:
                    return True
        return False

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

        # CRITICAL FIX: Include assistant's analysis responses in summary
        # Previously, assistant_responses were collected but only output when NO tool_calls existed.
        # In analysis scenarios, the AI reads files (tool_calls) AND gives suggestions (assistant content).
        # The suggestions were being silently discarded, causing the AI to "forget" what it just said
        # and re-analyze the same files repeatedly.
        if assistant_responses:
            for resp in assistant_responses:
                # Truncate long assistant responses proportionally
                resp_max = max(2000, int(2000 * preserve_ratio * 1.5))
                if len(resp) > resp_max:
                    resp = resp[:resp_max] + "..."
                lines.append(f"  Assistant: {resp}")

        if tool_calls_count or tool_results_count:
            lines.append(f"  Stats: {tool_calls_count} tool(s), {tool_results_count} result(s)")

        return "\n".join(lines)
