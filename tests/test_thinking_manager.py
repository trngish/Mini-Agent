from mini_agent.schema import Message
from mini_agent.utils.thinking_manager import ThinkingManager, ThinkingStats


class TestThinkingStats:
    def test_dataclass(self):
        stats = ThinkingStats(tokens=100, chars=400, created_at=1.0)
        assert stats.tokens == 100
        assert stats.chars == 400
        assert stats.created_at == 1.0


class TestThinkingManager:
    def test_init(self):
        tm = ThinkingManager(max_thinking_tokens=32768)
        assert tm.max_thinking_tokens == 32768
        assert tm.prune_threshold == int(32768 * 0.8)

    def test_default_max(self):
        tm = ThinkingManager()
        assert tm.max_thinking_tokens == ThinkingManager.DEFAULT_MAX_THINKING_TOKENS
        assert tm.max_thinking_tokens == 150_000

    def test_estimate_thinking_tokens_none(self):
        tm = ThinkingManager()
        assert tm.estimate_thinking_tokens(None) == 0

    def test_estimate_thinking_tokens_empty(self):
        tm = ThinkingManager()
        assert tm.estimate_thinking_tokens("") == 0

    def test_estimate_thinking_tokens_with_content(self):
        tm = ThinkingManager()
        tokens = tm.estimate_thinking_tokens("This is some thinking content")
        assert tokens > 0

    def test_analyze_messages_no_thinking(self):
        tm = ThinkingManager()
        messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
        ]
        result = tm.analyze_messages(messages)
        assert result["total_thinking_tokens"] == 0
        assert result["over_threshold"] is False
        assert result["messages_needing_prune"] == []

    def test_analyze_messages_with_thinking(self):
        tm = ThinkingManager()
        messages = [
            Message(role="system", content="system"),
            Message(role="assistant", content="hi", thinking="I need to think about this carefully"),
        ]
        result = tm.analyze_messages(messages)
        assert result["total_thinking_tokens"] > 0
        assert isinstance(result["thinking_by_msg"], dict)

    def test_prune_thinking_no_prune_needed(self):
        tm = ThinkingManager()
        messages = [
            Message(role="system", content="system"),
            Message(role="assistant", content="hi", thinking="short thought"),
        ]
        freed = tm.prune_thinking(messages)
        assert freed == 0

    def test_get_thinking_report(self):
        tm = ThinkingManager()
        messages = [
            Message(role="system", content="system"),
            Message(role="assistant", content="hi", thinking="thinking"),
        ]
        report = tm.get_thinking_report(messages)
        assert "Thinking Usage Report" in report
        assert "Total thinking tokens" in report

    def test_analyze_messages_over_threshold(self):
        tm = ThinkingManager(max_thinking_tokens=100)
        long_thinking = "word " * 500
        messages = [
            Message(role="system", content="system"),
            Message(role="assistant", content="hi", thinking=long_thinking),
            Message(role="user", content="more"),
            Message(role="assistant", content="hi", thinking=long_thinking),
        ]
        result = tm.analyze_messages(messages)
        assert result["over_threshold"] is True
