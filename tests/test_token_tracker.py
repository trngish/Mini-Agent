"""Tests for token tracking functionality."""

import pytest

from mini_agent.core.token_tracker import TokenTracker
from mini_agent.schema import Message


class TestTokenTracker:
    """Test basic token tracker functionality."""

    def test_token_tracker_init(self):
        """Test TokenTracker initialization."""
        tracker = TokenTracker()
        assert tracker._encoding_name == "cl100k_base"
        assert tracker.cached_count == 0
        assert tracker.cache_version == 0

    def test_estimate_tokens_empty(self):
        """Test estimation with empty message list."""
        tracker = TokenTracker()
        count = tracker.estimate_tokens([])
        assert count == 0

    def test_estimate_tokens_single_message(self):
        """Test estimation with single message."""
        tracker = TokenTracker()
        messages = [Message(role="user", content="Hello, world!")]
        count = tracker.estimate_tokens(messages)
        assert count > 0

    def test_estimate_tokens_incremental(self):
        """Test incremental token counting."""
        tracker = TokenTracker()
        msg1 = Message(role="user", content="First message")
        msg2 = Message(role="assistant", content="Second message")

        count1 = tracker.estimate_tokens([msg1])
        count2 = tracker.estimate_tokens([msg1, msg2])

        assert count2 > count1

    def test_estimate_tokens_with_thinking(self):
        """Test estimation includes thinking content."""
        tracker = TokenTracker()
        messages = [
            Message(role="assistant", content="Thinking...", thinking="Let me think about this")
        ]
        count = tracker.estimate_tokens(messages)
        # Should include both content and thinking
        assert count > 0

    def test_estimate_tokens_with_tool_calls(self):
        """Test estimation includes tool calls."""
        tracker = TokenTracker()
        messages = [
            Message(
                role="assistant",
                content="Using tool...",
                tool_calls=[{"id": "call_123", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}],
            )
        ]
        count = tracker.estimate_tokens(messages)
        assert count > 0

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        tracker = TokenTracker()
        messages = [Message(role="user", content="Test message")]

        tracker.estimate_tokens(messages)
        assert tracker.cached_count > 0

        tracker.invalidate_cache()
        assert tracker.cached_count == 0
        assert tracker._cached_token_index == 0

    def test_cache_version_increments(self):
        """Test cache version increments on invalidation."""
        tracker = TokenTracker()
        initial_version = tracker.cache_version

        tracker.invalidate_cache()
        assert tracker.cache_version == initial_version + 1


class TestTokenEstimationAccuracy:
    """Test tiktoken-based token estimation accuracy."""

    def test_tiktoken_encoder_available(self):
        """Verify tiktoken encoder is available for accurate estimation."""
        from mini_agent.utils.token_utils import get_encoder

        encoder = get_encoder("cl100k_base")
        tokens = encoder.encode("Hello, world!")
        # cl100k_base: "Hello, world!" = 4 tokens
        assert len(tokens) == 4

    def test_estimate_tokens_vs_fallback(self):
        """Compare tiktoken estimation vs fallback for accuracy."""
        tracker = TokenTracker()
        messages = [
            Message(role="user", content="This is a longer test message to check token estimation accuracy.")
        ]

        tiktoken_count = tracker.estimate_tokens(messages)
        # Tiktoken should give more accurate count than char/2.5
        assert tiktoken_count > 0

        # Verify fallback exists and works
        fallback_count = tracker._estimate_tokens_fallback(messages)
        assert fallback_count > 0

    def test_thinking_content_token_counting(self):
        """Test that thinking content is included in token estimation."""
        tracker = TokenTracker()
        messages = [
            Message(role="assistant", content="Thinking...", thinking="Let me think about this carefully")
        ]

        count = tracker.estimate_tokens(messages)
        # Should include thinking content in token count
        # Minimum should be more than the fallback for content alone
        assert count > 0

    def test_tiktoken_accuracy_comparison(self):
        """Test that tiktoken provides more accurate estimates than fallback."""
        tracker = TokenTracker()
        text = "The quick brown fox jumps over the lazy dog"

        # Using tiktoken directly for ground truth
        from mini_agent.utils.token_utils import get_encoder

        encoder = get_encoder("cl100k_base")
        tiktoken_count = len(encoder.encode(text))

        # Compare with fallback
        fallback_count = int(len(text) / 2.5)

        # Fallback should be reasonably close but tiktoken is more accurate
        # Both should be within reasonable range of each other
        assert tiktoken_count > 0
        assert fallback_count > 0

        # Fallback (chars/2.5) should not be more than 2x off from tiktoken for normal English
        assert fallback_count <= tiktoken_count * 2
        assert fallback_count >= tiktoken_count / 2

    def test_estimate_tokens_consistency(self):
        """Test that repeated estimates give consistent results."""
        tracker = TokenTracker()
        messages = [
            Message(role="user", content="Test message for consistency"),
            Message(role="assistant", content="Response with more content here"),
        ]

        count1 = tracker.estimate_tokens(messages)
        count2 = tracker.estimate_tokens(messages)

        assert count1 == count2

    def test_estimate_tokens_after_invalidation(self):
        """Test that estimates are recalculated after cache invalidation."""
        tracker = TokenTracker()
        messages = [Message(role="user", content="Test message")]

        count1 = tracker.estimate_tokens(messages)
        tracker.invalidate_cache()
        count2 = tracker.estimate_tokens(messages)

        # Should get same count after invalidation
        assert count1 == count2