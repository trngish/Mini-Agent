"""Thinking content management to prevent context overflow from truncated thinking.

Problem: M2.7's extended thinking can generate 32K+ tokens per response.
When accumulated over multiple steps, this causes context overflow.

Solution:
1. Monitor thinking token usage per message
2. Prune old thinking content when it exceeds thresholds
3. Keep thinking summaries instead of full content for old messages
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schema import Message


@dataclass
class ThinkingStats:
    """Statistics about thinking content for a message."""

    tokens: int
    chars: int
    created_at: float


class ThinkingManager:
    """Manages thinking content to prevent context overflow.

    Strategies:
    1. Track thinking size per message
    2. When total thinking exceeds threshold, prune oldest thinking
    3. Replace pruned thinking with compact summary
    """

    # Default: keep max 100K tokens of thinking across all messages
    DEFAULT_MAX_THINKING_TOKENS = 100_000
    # After this, start pruning
    PRUNE_THRESHOLD_TOKENS = 80_000
    # When pruning, keep top N most recent thinking blocks
    KEEP_RECENT_BLOCKS = 3
    # Prune to this percentage of original
    PRUNE_TO_PERCENT = 0.3

    def __init__(self, max_thinking_tokens: int = DEFAULT_MAX_THINKING_TOKENS):
        """Initialize thinking manager.

        Args:
            max_thinking_tokens: Maximum thinking tokens before pruning
        """
        self.max_thinking_tokens = max_thinking_tokens
        self.prune_threshold = int(max_thinking_tokens * 0.8)
        self._thinking_stats: dict[int, ThinkingStats] = {}  # msg_idx -> stats
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        """Lazy load tiktoken encoder."""
        if self._encoder is None:
            from ..utils.token_utils import get_encoder

            self._encoder = get_encoder("cl100k_base")
        return self._encoder

    def estimate_thinking_tokens(self, thinking: str | None) -> int:
        """Estimate tokens in thinking content."""
        if not thinking:
            return 0
        try:
            encoder = self._get_encoder()
            return len(encoder.encode(thinking))
        except Exception:
            return len(thinking) // 4  # Fallback: ~4 chars per token

    def analyze_messages(self, messages: list[Message]) -> dict[str, Any]:
        """Analyze thinking content across all messages.

        Returns:
            Dict with analysis:
            - total_thinking_tokens: Total tokens used by thinking
            - thinking_by_msg: Dict of msg_idx -> tokens
            - messages_needing_prune: Indices that should be pruned
        """
        total_tokens = 0
        thinking_by_msg = {}

        for i, msg in enumerate(messages):
            if msg.thinking:
                tokens = self.estimate_thinking_tokens(msg.thinking)
                thinking_by_msg[i] = tokens
                total_tokens += tokens
            else:
                thinking_by_msg[i] = 0

        # Find messages that need pruning
        messages_needing_prune = []
        if total_tokens > self.prune_threshold:
            # Sort by index (oldest first), skip system and last 2 assistant messages
            candidates = [i for i in thinking_by_msg if i > 0 and thinking_by_msg[i] > 0 and i < len(messages) - 2]
            candidates.sort()

            # Mark oldest for pruning until under threshold
            tokens_to_free = total_tokens - self.prune_threshold
            accumulated_free = 0
            for i in candidates:
                if accumulated_free >= tokens_to_free:
                    break
                messages_needing_prune.append(i)
                accumulated_free += thinking_by_msg[i]

        return {
            "total_thinking_tokens": total_tokens,
            "thinking_by_msg": thinking_by_msg,
            "messages_needing_prune": messages_needing_prune,
            "over_threshold": total_tokens > self.prune_threshold,
        }

    def prune_thinking(self, messages: list[Message]) -> int:
        """Prune thinking content from old messages to free context space.

        Returns:
            Number of tokens freed
        """
        analysis = self.analyze_messages(messages)

        if not analysis["messages_needing_prune"]:
            return 0

        tokens_freed = 0
        encoder = self._get_encoder()

        for i in analysis["messages_needing_prune"]:
            msg = messages[i]
            if not msg.thinking:
                continue

            original_tokens = self.estimate_thinking_tokens(msg.thinking)

            # Create compact summary instead of full thinking
            summary = self._create_thinking_summary(msg.thinking, encoder)
            msg.thinking = summary

            tokens_freed += original_tokens - self.estimate_thinking_tokens(summary)

        return tokens_freed

    def _create_thinking_summary(self, thinking: str, encoder: Any) -> str:
        """Create compact summary of thinking content.

        Strategy:
        1. Split into lines/sections
        2. Keep first few lines (initial reasoning)
        3. Keep last few lines (conclusion)
        4. Truncate middle
        """
        lines = thinking.split("\n")

        if len(lines) <= 6:
            # Already short, just truncate if needed
            return self._truncate_thinking(thinking, encoder)

        # Keep: first 2 lines + last 2 lines
        kept_lines = lines[:2] + lines[-2:]

        # Estimate tokens
        kept_text = "\n".join(kept_lines)
        kept_tokens = len(encoder.encode(kept_text))

        # Target: ~30% of original
        target_tokens = int(self.estimate_thinking_tokens(thinking) * self.PRUNE_TO_PERCENT)

        if kept_tokens >= target_tokens:
            return self._truncate_thinking(kept_text, encoder)

        # Can add some middle content
        available_budget = target_tokens - kept_tokens
        middle_lines = []
        middle_tokens = 0

        for line in lines[2:-2]:
            line_tokens = len(encoder.encode(line))
            if middle_tokens + line_tokens > available_budget:
                break
            middle_lines.append(line)
            middle_tokens += line_tokens

        result = "\n".join(kept_lines[:2] + middle_lines + kept_lines[2:])
        return self._truncate_thinking(result, encoder)

    def _truncate_thinking(self, thinking: str, encoder: Any) -> str:
        """Truncate thinking to fit within token budget."""
        tokens = encoder.encode(thinking)

        if len(tokens) <= 500:
            return thinking

        # Keep first 250 and last 250 tokens
        truncated = encoder.decode(tokens[:250] + tokens[-250:])
        return truncated + f"\n... [truncated from {len(tokens)} tokens]"  # type: ignore[no-any-return]

    def get_thinking_report(self, messages: list[Message]) -> str:
        """Generate a report of thinking usage."""
        analysis = self.analyze_messages(messages)

        lines = [
            "🧠 Thinking Usage Report",
            "=" * 40,
            f"Total thinking tokens: {analysis['total_thinking_tokens']:,}",
            f"Max allowed: {self.max_thinking_tokens:,}",
            f"Prune threshold: {self.prune_threshold:,}",
            f"Over threshold: {analysis['over_threshold']}",
        ]

        if analysis["messages_needing_prune"]:
            lines.append(f"Messages to prune: {analysis['messages_needing_prune']}")

        return "\n".join(lines)
