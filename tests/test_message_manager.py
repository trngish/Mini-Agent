import logging
from unittest.mock import patch

import pytest

from mini_agent.core.message_manager import MessageManager
from mini_agent.schema import FunctionCall, Message, ToolCall


def _make_tool_call(name="read_file", args=None, call_id="tc_1"):
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=args or {"path": "/tmp/test.py"}),
    )


class TestInit:
    def test_default_attributes(self):
        mm = MessageManager(token_limit=100000)
        assert mm.token_limit == 100000
        assert mm.messages == []
        assert mm._skip_next_token_check is False
        assert mm._last_summary_quality == 1.0

    def test_token_tracker_initialized(self):
        mm = MessageManager(token_limit=50000)
        from mini_agent.core.token_tracker import TokenTracker

        assert isinstance(mm._token_tracker, TokenTracker)

    def test_summary_manager_initialized(self):
        mm = MessageManager(token_limit=50000)
        from mini_agent.utils.summary_manager import AdaptiveSummaryManager

        assert isinstance(mm._summary_manager, AdaptiveSummaryManager)
        assert mm._summary_manager.token_limit == 50000

    def test_different_token_limits(self):
        for limit in [1000, 50000, 200000]:
            mm = MessageManager(token_limit=limit)
            assert mm.token_limit == limit


class TestInitialize:
    def test_sets_system_prompt(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("You are a helpful assistant")
        assert len(mm.messages) == 1
        assert mm.messages[0].role == "system"
        assert mm.messages[0].content == "You are a helpful assistant"

    def test_resets_previous_messages(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="user", content="Hello"))
        mm.initialize("New system prompt")
        assert len(mm.messages) == 1
        assert mm.messages[0].content == "New system prompt"

    def test_empty_system_prompt(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("")
        assert mm.messages[0].content == ""
        assert mm.messages[0].role == "system"


class TestAddMessage:
    def test_add_user_message(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="user", content="Hello"))
        assert len(mm.messages) == 1
        assert mm.messages[0].role == "user"
        assert mm.messages[0].content == "Hello"

    def test_add_assistant_message(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="assistant", content="Hi there"))
        assert mm.messages[0].role == "assistant"

    def test_add_tool_message(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="tool", content="result", tool_call_id="tc_1"))
        assert mm.messages[0].role == "tool"
        assert mm.messages[0].tool_call_id == "tc_1"

    def test_add_message_with_tool_calls(self):
        mm = MessageManager(token_limit=100000)
        tc = _make_tool_call("read_file", {"path": "/tmp/a.py"})
        mm.add_message(Message(role="assistant", content="", tool_calls=[tc]))
        assert mm.messages[0].tool_calls is not None
        assert len(mm.messages[0].tool_calls) == 1
        assert mm.messages[0].tool_calls[0].function.name == "read_file"

    def test_add_multiple_messages_preserves_order(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="user", content="first"))
        mm.add_message(Message(role="assistant", content="second"))
        mm.add_message(Message(role="user", content="third"))
        assert [m.content for m in mm.messages] == ["first", "second", "third"]

    def test_add_message_with_list_content(self):
        mm = MessageManager(token_limit=100000)
        content = [{"type": "text", "text": "Hello"}]
        mm.add_message(Message(role="user", content=content))
        assert mm.messages[0].content == content


class TestGetMessages:
    def test_returns_empty_list_initially(self):
        mm = MessageManager(token_limit=100000)
        assert mm.get_messages() == []

    def test_returns_all_messages(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("system")
        mm.add_message(Message(role="user", content="Hi"))
        mm.add_message(Message(role="assistant", content="Hello"))
        msgs = mm.get_messages()
        assert len(msgs) == 3

    def test_returns_same_list_object(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="user", content="Hi"))
        assert mm.get_messages() is mm.messages


class TestReplaceMessages:
    def test_replace_with_new_messages(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("old system")
        new_msgs = [
            Message(role="system", content="new system"),
            Message(role="user", content="Hi"),
        ]
        mm.replace_messages(new_msgs)
        assert len(mm.messages) == 2
        assert mm.messages[0].content == "new system"
        assert mm.messages[1].content == "Hi"

    def test_replace_with_empty_list(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("system")
        mm.add_message(Message(role="user", content="Hi"))
        mm.replace_messages([])
        assert mm.messages == []

    def test_replace_preserves_tool_calls(self):
        mm = MessageManager(token_limit=100000)
        tc = _make_tool_call("bash", {"command": "ls"})
        new_msgs = [Message(role="assistant", content="", tool_calls=[tc])]
        mm.replace_messages(new_msgs)
        assert mm.messages[0].tool_calls is not None
        assert mm.messages[0].tool_calls[0].function.name == "bash"


class TestEstimateTokens:
    def test_returns_positive_int(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="user", content="Hello world"))
        count = mm.estimate_tokens()
        assert count > 0
        assert isinstance(count, int)

    def test_more_messages_more_tokens(self):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="user", content="short"))
        short_count = mm.estimate_tokens()
        mm.add_message(Message(role="user", content="a" * 1000))
        long_count = mm.estimate_tokens()
        assert long_count > short_count

    @patch("mini_agent.core.token_tracker.TokenTracker.estimate_tokens", return_value=42)
    def test_delegates_to_token_tracker(self, mock_estimate):
        mm = MessageManager(token_limit=100000)
        mm.add_message(Message(role="user", content="test"))
        result = mm.estimate_tokens()
        assert result == 42
        mock_estimate.assert_called_once_with(mm.messages)


class TestShouldSummarize:
    def test_below_threshold(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("system")
        mm.add_message(Message(role="user", content="short"))
        should, reason = mm.should_summarize(api_total_tokens=100)
        assert should is False

    def test_skip_next_check_returns_false(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("system")
        mm.add_message(Message(role="user", content="test"))
        mm._skip_next_token_check = True
        should, reason = mm.should_summarize(api_total_tokens=999999)
        assert should is False
        assert reason == ""

    def test_skip_next_check_resets_flag(self):
        mm = MessageManager(token_limit=100000)
        mm._skip_next_token_check = True
        mm.should_summarize(api_total_tokens=0)
        assert mm._skip_next_token_check is False

    @patch.object(MessageManager, "estimate_tokens", return_value=150000)
    def test_delegates_to_summary_manager(self, mock_tokens):
        mm = MessageManager(token_limit=100000)
        mm.initialize("system")
        mm.add_message(Message(role="user", content="test"))
        with patch.object(
            mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded")
        ) as mock_should:
            should, reason = mm.should_summarize(api_total_tokens=150000)
            assert should is True
            assert reason == "threshold_exceeded"
            mock_should.assert_called_once_with(mm.messages, 150000, 150000)


class TestMarkSkipNextCheck:
    def test_sets_skip_flag(self):
        mm = MessageManager(token_limit=100000)
        mm.mark_skip_next_check()
        assert mm._skip_next_token_check is True

    def test_high_quality_always_skips(self):
        mm = MessageManager(token_limit=100000)
        mm.mark_skip_next_check(quality=0.9)
        assert mm._skip_next_token_check is True

    def test_low_quality_still_sets_flag(self):
        mm = MessageManager(token_limit=100000)
        mm._skip_next_token_check = False
        mm.mark_skip_next_check(quality=0.1)
        assert mm._skip_next_token_check is True

    def test_high_quality_reinforced_by_summary_manager(self):
        mm = MessageManager(token_limit=100000)
        with patch.object(mm._summary_manager, "should_skip_next_check", return_value=True):
            mm.mark_skip_next_check(quality=0.9)
            assert mm._skip_next_token_check is True


class TestCreateLocalSummary:
    def test_empty_messages(self):
        mm = MessageManager(token_limit=100000)
        result = mm._create_local_summary([], round_num=1)
        assert result == ""

    def test_assistant_content_only(self):
        mm = MessageManager(token_limit=100000)
        messages = [Message(role="assistant", content="Hello from assistant")]
        result = mm._create_local_summary(messages, round_num=1)
        assert "Round 1:" in result
        assert "Response: Hello from assistant" in result

    def test_assistant_with_tool_calls(self):
        mm = MessageManager(token_limit=100000)
        tc = _make_tool_call("read_file", {"path": "/tmp/test.py"})
        messages = [Message(role="assistant", content="", tool_calls=[tc])]
        result = mm._create_local_summary(messages, round_num=2)
        assert "Round 2:" in result
        assert "Tools called:" in result
        assert "read_file" in result

    def test_tool_result_message(self):
        mm = MessageManager(token_limit=100000)
        messages = [Message(role="tool", content="file contents here")]
        result = mm._create_local_summary(messages, round_num=1)
        assert "Result: file contents here" in result
        assert "Stats:" in result

    def test_mixed_messages(self):
        mm = MessageManager(token_limit=100000)
        tc = _make_tool_call("bash", {"command": "ls"})
        messages = [
            Message(role="assistant", content="Running command", tool_calls=[tc]),
            Message(role="tool", content="file1.py\nfile2.py"),
        ]
        result = mm._create_local_summary(messages, round_num=1)
        assert "Round 1:" in result
        assert "Tools called:" in result
        assert "bash" in result
        assert "Result:" in result
        assert "Stats: 1 tool(s), 1 result(s)" in result

    def test_assistant_content_truncation(self):
        mm = MessageManager(token_limit=100000)
        long_content = "x" * 2000
        messages = [Message(role="assistant", content=long_content)]
        result = mm._create_local_summary(messages, round_num=1, max_truncation=500)
        assert "..." in result
        assert len(result) < len(long_content)

    def test_tool_result_truncation(self):
        mm = MessageManager(token_limit=100000)
        long_result = "y" * 5000
        messages = [Message(role="tool", content=long_result)]
        result = mm._create_local_summary(messages, round_num=1, preserve_ratio=0.4)
        assert "..." in result

    def test_tool_call_args_truncation(self):
        mm = MessageManager(token_limit=100000)
        long_args = {"path": "a" * 500}
        tc = _make_tool_call("read_file", long_args)
        messages = [Message(role="assistant", content="", tool_calls=[tc])]
        result = mm._create_local_summary(messages, round_num=1, preserve_ratio=0.4)
        assert "..." in result

    def test_no_tool_calls_shows_response(self):
        mm = MessageManager(token_limit=100000)
        messages = [Message(role="assistant", content="Just a response")]
        result = mm._create_local_summary(messages, round_num=1)
        assert "Response: Just a response" in result
        assert "Stats:" not in result

    def test_with_tool_calls_no_response_line(self):
        mm = MessageManager(token_limit=100000)
        tc = _make_tool_call("bash", {"command": "ls"})
        messages = [Message(role="assistant", content="", tool_calls=[tc])]
        result = mm._create_local_summary(messages, round_num=1)
        assert "Response:" not in result

    def test_multiple_tool_calls(self):
        mm = MessageManager(token_limit=100000)
        tc1 = _make_tool_call("read_file", {"path": "/a"}, "tc_1")
        tc2 = _make_tool_call("bash", {"command": "ls"}, "tc_2")
        messages = [Message(role="assistant", content="", tool_calls=[tc1, tc2])]
        result = mm._create_local_summary(messages, round_num=1)
        assert "read_file" in result
        assert "bash" in result
        assert "Stats: 2 tool(s), 0 result(s)" in result

    def test_list_content_handled(self):
        mm = MessageManager(token_limit=100000)
        content = [{"type": "text", "text": "Hello"}]
        messages = [Message(role="assistant", content=content)]
        result = mm._create_local_summary(messages, round_num=1)
        assert "Response:" in result

    def test_round_number_in_output(self):
        mm = MessageManager(token_limit=100000)
        messages = [Message(role="assistant", content="test")]
        for round_num in [1, 5, 10]:
            result = mm._create_local_summary(messages, round_num=round_num)
            assert f"Round {round_num}:" in result

    def test_preserve_ratio_affects_truncation(self):
        mm = MessageManager(token_limit=100000)
        long_result = "z" * 5000
        messages = [Message(role="tool", content=long_result)]
        result_low = mm._create_local_summary(messages, round_num=1, preserve_ratio=0.2)
        result_high = mm._create_local_summary(messages, round_num=1, preserve_ratio=1.0)
        assert len(result_low) <= len(result_high)


class TestSummarizeMessages:
    @pytest.fixture
    def mm(self):
        return MessageManager(token_limit=100000)

    @pytest.fixture
    def logger(self):
        return logging.getLogger("test")

    def _build_conversation(self, n_rounds=2):
        messages = [Message(role="system", content="You are helpful")]
        for i in range(n_rounds):
            messages.append(Message(role="user", content=f"User message {i + 1}"))
            tc = _make_tool_call("bash", {"command": f"echo {i}"}, f"tc_{i}")
            messages.append(Message(role="assistant", content=f"Assistant reply {i + 1}", tool_calls=[tc]))
            messages.append(Message(role="tool", content=f"output {i}"))
        return messages

    @pytest.mark.asyncio
    async def test_skip_next_check_returns_early(self, mm, logger):
        mm._skip_next_token_check = True
        messages = self._build_conversation()
        result = await mm.summarize_messages(messages, api_total_tokens=0, logger=logger)
        assert result is messages
        assert mm._skip_next_token_check is False

    @pytest.mark.asyncio
    async def test_should_not_summarize_returns_same_list(self, mm, logger):
        messages = self._build_conversation()
        with patch.object(mm._summary_manager, "should_summarize", return_value=(False, "below_threshold")):
            result = await mm.summarize_messages(messages, api_total_tokens=100, logger=logger)
            assert result is messages

    @pytest.mark.asyncio
    async def test_insufficient_user_messages(self, mm, logger):
        messages = [
            Message(role="system", content="system"),
        ]
        with (
            patch.object(mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded")),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=200000),
        ):
            result = await mm.summarize_messages(messages, api_total_tokens=200000, logger=logger)
            assert result is messages

    @pytest.mark.asyncio
    async def test_only_system_no_user_after_index_0(self, mm, logger):
        messages = [
            Message(role="system", content="system"),
            Message(role="assistant", content="hello"),
        ]
        with (
            patch.object(mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded")),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=200000),
        ):
            result = await mm.summarize_messages(messages, api_total_tokens=200000, logger=logger)
            assert result is messages

    @pytest.mark.asyncio
    async def test_normal_summarization(self, mm, logger):
        messages = self._build_conversation(n_rounds=2)
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
        ):
            result = await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            assert result is not messages
            assert result[0].role == "system"
            user_msgs = [m for m in result if m.role == "user" and not m.content.startswith("[Execution Summary")]
            summary_msgs = [m for m in result if m.role == "user" and m.content.startswith("[Execution Summary")]
            assert len(user_msgs) == 2
            assert len(summary_msgs) >= 1

    @pytest.mark.asyncio
    async def test_tier_extraction_from_reason(self, mm, logger):
        messages = self._build_conversation()
        with (
            patch.object(mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=low")),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(
                mm._summary_manager, "get_summary_config", return_value={"preserve_ratio": 0.4, "max_truncation": 1500}
            ) as mock_config,
        ):
            await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            mock_config.assert_called_with("low")

    @pytest.mark.asyncio
    async def test_early_trigger_tier_is_low(self, mm, logger):
        messages = self._build_conversation()
        with (
            patch.object(mm._summary_manager, "should_summarize", return_value=(True, "early_trigger:high")),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(
                mm._summary_manager, "get_summary_config", return_value={"preserve_ratio": 0.4, "max_truncation": 1500}
            ) as mock_config,
        ):
            await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            mock_config.assert_called_with("low")

    @pytest.mark.asyncio
    async def test_default_tier_is_medium(self, mm, logger):
        messages = self._build_conversation()
        with (
            patch.object(mm._summary_manager, "should_summarize", return_value=(True, "some_other_reason")),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(
                mm._summary_manager, "get_summary_config", return_value={"preserve_ratio": 0.6, "max_truncation": 1000}
            ) as mock_config,
        ):
            await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            mock_config.assert_called_with("medium")

    @pytest.mark.asyncio
    async def test_long_user_content_truncation(self, mm, logger):
        long_content = "A" * 6000
        messages = [
            Message(role="system", content="system"),
            Message(role="user", content=long_content),
            Message(role="assistant", content="done"),
        ]
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
        ):
            with patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.5):
                result = await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
                user_msg = [m for m in result if m.role == "user" and not m.content.startswith("[Execution Summary")][0]
                assert "...[truncated]..." in user_msg.content
                assert len(user_msg.content) < len(long_content)

    @pytest.mark.asyncio
    async def test_short_user_content_not_truncated(self, mm, logger):
        messages = [
            Message(role="system", content="system"),
            Message(role="user", content="short message"),
            Message(role="assistant", content="done"),
        ]
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
        ):
            with patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.5):
                result = await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
                user_msg = [m for m in result if m.role == "user" and not m.content.startswith("[Execution Summary")][0]
                assert "...[truncated]..." not in user_msg.content
                assert user_msg.content == "short message"

    @pytest.mark.asyncio
    async def test_sets_skip_next_check_after_summarization(self, mm, logger):
        messages = self._build_conversation()
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.8),
            patch.object(mm._summary_manager, "should_skip_next_check", return_value=True),
        ):
            await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            assert mm._skip_next_token_check is True

    @pytest.mark.asyncio
    async def test_updates_last_summary_quality(self, mm, logger):
        messages = self._build_conversation()
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.75),
            patch.object(mm._summary_manager, "should_skip_next_check", return_value=False),
        ):
            await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            assert mm._last_summary_quality == 0.75

    @pytest.mark.asyncio
    async def test_no_execution_messages_no_summary(self, mm, logger):
        messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
        ]
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.5),
            patch.object(mm._summary_manager, "should_skip_next_check", return_value=False),
        ):
            result = await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            summary_msgs = [m for m in result if m.role == "user" and m.content.startswith("[Execution Summary")]
            assert len(summary_msgs) == 0

    @pytest.mark.asyncio
    async def test_max_truncation_parameter_accepted(self, mm, logger):
        messages = self._build_conversation()
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.5),
            patch.object(mm._summary_manager, "should_skip_next_check", return_value=False),
        ):
            result = await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger, _max_truncation=500)
            assert result is not messages

    @pytest.mark.asyncio
    async def test_summary_structure_preserved(self, mm, logger):
        messages = self._build_conversation(n_rounds=3)
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.5),
            patch.object(mm._summary_manager, "should_skip_next_check", return_value=False),
        ):
            result = await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            assert result[0].role == "system"
            user_msgs = [m for m in result if m.role == "user" and not m.content.startswith("[Execution Summary")]
            assert len(user_msgs) == 3

    @pytest.mark.asyncio
    async def test_single_round_summarization(self, mm, logger):
        messages = [
            Message(role="system", content="system"),
            Message(role="user", content="do something"),
            Message(role="assistant", content="working", tool_calls=[_make_tool_call()]),
            Message(role="tool", content="done"),
        ]
        with (
            patch.object(
                mm._summary_manager, "should_summarize", return_value=(True, "threshold_exceeded:tier=medium")
            ),
            patch.object(mm._token_tracker, "estimate_tokens", return_value=150000),
            patch.object(mm._summary_manager, "estimate_summary_quality", return_value=0.5),
            patch.object(mm._summary_manager, "should_skip_next_check", return_value=False),
        ):
            result = await mm.summarize_messages(messages, api_total_tokens=150000, logger=logger)
            assert len(result) >= 2
            assert result[0].role == "system"
            summary_msgs = [m for m in result if m.content.startswith("[Execution Summary")]
            assert len(summary_msgs) == 1
            assert "[Execution Summary 1]" in summary_msgs[0].content

    @pytest.mark.asyncio
    async def test_skip_check_resets_flag(self, mm, logger):
        mm._skip_next_token_check = True
        messages = self._build_conversation()
        await mm.summarize_messages(messages, api_total_tokens=0, logger=logger)
        assert mm._skip_next_token_check is False


class TestMultipleMessages:
    def test_multiple_messages(self):
        mm = MessageManager(token_limit=100000)
        mm.initialize("system")
        mm.add_message(Message(role="user", content="Hello"))
        mm.add_message(Message(role="assistant", content="Hi there"))
        mm.add_message(Message(role="user", content="How are you?"))
        assert len(mm.get_messages()) == 4
