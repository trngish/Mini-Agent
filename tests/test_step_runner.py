"""Comprehensive unit tests for StepRunner class.

Tests cover all public methods:
- process_response()
- check_health()
- prune_thinking()
- is_complete()
- print_completion_summary()
- auto_save()
- print_step_timing()
"""

from unittest.mock import MagicMock, patch

import pytest

from mini_agent.core.step_runner import StepRunner
from mini_agent.schema import FunctionCall, Message, ToolCall


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.api_call_count = 0
    agent.api_total_tokens = 0
    agent.thinking_budget = 10000
    agent.messages = []
    agent.logger = MagicMock()
    agent._error_recovery = MagicMock()
    agent._error_recovery.consecutive_failures = 0
    agent._last_health_check_step = 0
    agent._health_check_interval = 5
    agent._thinking_manager = None
    agent._token_tracker = MagicMock()
    agent.auto_save = False
    agent._last_auto_save_step = 0
    agent._session_manager = MagicMock()
    return agent


@pytest.fixture
def runner(mock_agent):
    return StepRunner(mock_agent, run_start_time=100.0)


def _make_tool_call(id="tc_1", name="read_file", arguments=None):
    return ToolCall(
        id=id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments or {"path": "/tmp/f"}),
    )


def _make_response(content="Hello", thinking=None, tool_calls=None, finish_reason="stop", usage=None):
    resp = MagicMock()
    resp.content = content
    resp.thinking = thinking
    resp.tool_calls = tool_calls
    resp.finish_reason = finish_reason
    resp.usage = usage
    return resp


class TestInit:
    def test_stores_agent(self, runner, mock_agent):
        assert runner._agent is mock_agent

    def test_stores_run_start_time(self, runner):
        assert runner._run_start_time == 100.0


class TestProcessResponse:
    def test_increments_api_call_count(self, runner, mock_agent):
        mock_response = _make_response()
        runner.process_response(mock_response, 1)
        assert mock_agent.api_call_count == 1

    def test_increments_api_call_count_multiple(self, runner, mock_agent):
        mock_response = _make_response()
        runner.process_response(mock_response, 1)
        runner.process_response(mock_response, 2)
        assert mock_agent.api_call_count == 2

    def test_updates_total_tokens_when_usage_present(self, runner, mock_agent):
        usage = MagicMock()
        usage.total_tokens = 5000
        mock_response = _make_response(usage=usage)
        runner.process_response(mock_response, 1)
        assert mock_agent.api_total_tokens == 5000

    def test_does_not_update_tokens_when_usage_is_none(self, runner, mock_agent):
        mock_response = _make_response(usage=None)
        runner.process_response(mock_response, 1)
        assert mock_agent.api_total_tokens == 0

    def test_creates_assistant_message_with_content(self, runner, mock_agent):
        mock_response = _make_response(content="I will help you.")
        msg = runner.process_response(mock_response, 1)
        assert msg.role == "assistant"
        assert msg.content == "I will help you."

    def test_creates_message_with_thinking(self, runner, mock_agent):
        mock_response = _make_response(thinking="Let me think...")
        msg = runner.process_response(mock_response, 1)
        assert msg.thinking == "Let me think..."

    def test_creates_message_with_tool_calls(self, runner, mock_agent):
        tc = _make_tool_call()
        mock_response = _make_response(tool_calls=[tc])
        msg = runner.process_response(mock_response, 1)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "tc_1"
        assert msg.tool_calls[0].function.name == "read_file"

    def test_creates_message_with_multiple_tool_calls(self, runner, mock_agent):
        tc1 = _make_tool_call(id="tc_1", name="read_file")
        tc2 = _make_tool_call(id="tc_2", name="write_file", arguments={"path": "/tmp/out", "content": "hi"})
        mock_response = _make_response(tool_calls=[tc1, tc2])
        msg = runner.process_response(mock_response, 1)
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].function.name == "read_file"
        assert msg.tool_calls[1].function.name == "write_file"

    def test_creates_message_with_no_tool_calls(self, runner, mock_agent):
        mock_response = _make_response(tool_calls=None)
        msg = runner.process_response(mock_response, 1)
        assert msg.tool_calls is None

    def test_appends_message_to_agent_messages(self, runner, mock_agent):
        mock_response = _make_response(content="response text")
        runner.process_response(mock_response, 1)
        assert len(mock_agent.messages) == 1
        assert mock_agent.messages[0].role == "assistant"
        assert mock_agent.messages[0].content == "response text"

    def test_appends_multiple_messages(self, runner, mock_agent):
        runner.process_response(_make_response(content="first"), 1)
        runner.process_response(_make_response(content="second"), 2)
        assert len(mock_agent.messages) == 2
        assert mock_agent.messages[0].content == "first"
        assert mock_agent.messages[1].content == "second"

    def test_returns_the_created_message(self, runner, mock_agent):
        mock_response = _make_response(content="hello back")
        msg = runner.process_response(mock_response, 1)
        assert isinstance(msg, Message)
        assert msg.content == "hello back"

    def test_logs_response(self, runner, mock_agent):
        tc = _make_tool_call()
        mock_response = _make_response(
            content="text",
            thinking="hmm",
            tool_calls=[tc],
            finish_reason="tool_use",
        )
        runner.process_response(mock_response, 1)
        mock_agent.logger.log_response.assert_called_once_with(
            content="text",
            thinking="hmm",
            tool_calls=[tc],
            finish_reason="tool_use",
        )

    def test_prints_api_call_info(self, runner, mock_agent, capsys):
        mock_response = _make_response(tool_calls=[_make_tool_call()])
        runner.process_response(mock_response, 1)
        printed = capsys.readouterr().out
        assert "API Call #1" in printed
        assert "Tools: 1" in printed
        assert "Total tokens: 0" in printed

    def test_prints_zero_tools_when_no_tool_calls(self, runner, mock_agent, capsys):
        mock_response = _make_response(tool_calls=None)
        runner.process_response(mock_response, 1)
        printed = capsys.readouterr().out
        assert "Tools: 0" in printed

    def test_prints_thinking_budget(self, runner, mock_agent, capsys):
        mock_agent.thinking_budget = 8000
        mock_response = _make_response()
        runner.process_response(mock_response, 1)
        printed = capsys.readouterr().out
        assert "8000" in printed

    def test_prints_total_tokens_with_comma_formatting(self, runner, mock_agent, capsys):
        usage = MagicMock()
        usage.total_tokens = 12345
        mock_response = _make_response(usage=usage)
        runner.process_response(mock_response, 1)
        printed = capsys.readouterr().out
        assert "12,345" in printed


class TestCheckHealth:
    def test_returns_issues_when_consecutive_failures(self, runner, mock_agent):
        mock_agent._error_recovery.consecutive_failures = 3
        mock_agent._check_health.return_value = ["high error rate"]
        issues = runner.check_health(step=5)
        assert issues == ["high error rate"]
        mock_agent._check_health.assert_called_once()

    def test_runs_health_check_when_interval_reached(self, runner, mock_agent):
        mock_agent._last_health_check_step = 0
        mock_agent._health_check_interval = 5
        mock_agent._check_health.return_value = []
        issues = runner.check_health(step=5)
        mock_agent._check_health.assert_called_once()
        assert issues == []

    def test_skips_health_check_when_interval_not_reached(self, runner, mock_agent):
        mock_agent._last_health_check_step = 3
        mock_agent._health_check_interval = 5
        issues = runner.check_health(step=4)
        mock_agent._check_health.assert_not_called()
        assert issues == []

    def test_skips_when_consecutive_failures_zero_and_interval_not_reached(self, runner, mock_agent):
        mock_agent._error_recovery.consecutive_failures = 0
        mock_agent._last_health_check_step = 2
        mock_agent._health_check_interval = 10
        issues = runner.check_health(step=5)
        assert issues == []
        mock_agent._check_health.assert_not_called()

    def test_updates_last_health_check_step(self, runner, mock_agent):
        mock_agent._error_recovery.consecutive_failures = 1
        mock_agent._check_health.return_value = []
        runner.check_health(step=7)
        assert mock_agent._last_health_check_step == 7

    def test_does_not_update_step_when_skipped(self, runner, mock_agent):
        mock_agent._last_health_check_step = 3
        runner.check_health(step=4)
        assert mock_agent._last_health_check_step == 3

    def test_returns_multiple_issues(self, runner, mock_agent):
        mock_agent._error_recovery.consecutive_failures = 1
        mock_agent._check_health.return_value = ["issue1", "issue2", "issue3"]
        issues = runner.check_health(step=1)
        assert len(issues) == 3

    def test_runs_on_first_step(self, runner, mock_agent):
        mock_agent._error_recovery.consecutive_failures = 1
        mock_agent._check_health.return_value = []
        runner.check_health(step=0)
        mock_agent._check_health.assert_called_once()

    def test_interval_boundary_exact(self, runner, mock_agent):
        mock_agent._last_health_check_step = 0
        mock_agent._health_check_interval = 5
        runner.check_health(step=4)
        mock_agent._check_health.assert_not_called()
        runner.check_health(step=5)
        mock_agent._check_health.assert_called_once()


class TestPruneThinking:
    def test_returns_zero_when_no_thinking_manager(self, runner, mock_agent):
        mock_agent._thinking_manager = None
        result = runner.prune_thinking()
        assert result == 0

    def test_returns_zero_when_few_messages(self, runner, mock_agent):
        mock_agent._thinking_manager = MagicMock()
        mock_agent.messages = [MagicMock(), MagicMock()]
        result = runner.prune_thinking()
        assert result == 0
        mock_agent._thinking_manager.prune_thinking.assert_not_called()

    def test_prunes_when_manager_present_and_many_messages(self, runner, mock_agent):
        mock_agent._thinking_manager = MagicMock()
        mock_agent._thinking_manager.prune_thinking.return_value = 500
        mock_agent.messages = [MagicMock() for _ in range(6)]
        result = runner.prune_thinking()
        assert result == 500
        mock_agent._thinking_manager.prune_thinking.assert_called_once_with(mock_agent.messages)

    def test_does_not_invalidate_cache_when_tokens_below_threshold(self, runner, mock_agent):
        mock_agent._thinking_manager = MagicMock()
        mock_agent._thinking_manager.prune_thinking.return_value = 500
        mock_agent.messages = [MagicMock() for _ in range(6)]
        runner.prune_thinking()
        mock_agent._token_tracker.invalidate_cache.assert_not_called()

    def test_invalidates_cache_when_tokens_above_1000(self, runner, mock_agent):
        mock_agent._thinking_manager = MagicMock()
        mock_agent._thinking_manager.prune_thinking.return_value = 5000
        mock_agent.messages = [MagicMock() for _ in range(6)]
        result = runner.prune_thinking()
        assert result == 5000
        mock_agent._token_tracker.invalidate_cache.assert_called_once()

    def test_prints_prune_message_when_tokens_above_1000(self, runner, mock_agent, capsys):
        mock_agent._thinking_manager = MagicMock()
        mock_agent._thinking_manager.prune_thinking.return_value = 2500
        mock_agent.messages = [MagicMock() for _ in range(6)]
        runner.prune_thinking()
        printed = capsys.readouterr().out
        assert "Pruned" in printed
        assert "2,500" in printed

    def test_does_not_print_when_tokens_below_1000(self, runner, mock_agent, capsys):
        mock_agent._thinking_manager = MagicMock()
        mock_agent._thinking_manager.prune_thinking.return_value = 800
        mock_agent.messages = [MagicMock() for _ in range(6)]
        runner.prune_thinking()
        printed = capsys.readouterr().out
        assert "Pruned" not in printed

    def test_exactly_5_messages_does_not_prune(self, runner, mock_agent):
        mock_agent._thinking_manager = MagicMock()
        mock_agent.messages = [MagicMock() for _ in range(5)]
        result = runner.prune_thinking()
        assert result == 0
        mock_agent._thinking_manager.prune_thinking.assert_not_called()

    def test_exactly_6_messages_does_prune(self, runner, mock_agent):
        mock_agent._thinking_manager = MagicMock()
        mock_agent._thinking_manager.prune_thinking.return_value = 100
        mock_agent.messages = [MagicMock() for _ in range(6)]
        result = runner.prune_thinking()
        assert result == 100
        mock_agent._thinking_manager.prune_thinking.assert_called_once()


class TestIsComplete:
    def test_returns_true_when_no_tool_calls(self, runner):
        response = _make_response(tool_calls=None)
        assert runner.is_complete(response) is True

    def test_returns_true_when_empty_tool_calls_list(self, runner):
        response = _make_response(tool_calls=[])
        assert runner.is_complete(response) is True

    def test_returns_false_when_tool_calls_present(self, runner):
        tc = _make_tool_call()
        response = _make_response(tool_calls=[tc])
        assert runner.is_complete(response) is False

    def test_returns_false_with_multiple_tool_calls(self, runner):
        tc1 = _make_tool_call(id="tc_1", name="read_file")
        tc2 = _make_tool_call(id="tc_2", name="write_file")
        response = _make_response(tool_calls=[tc1, tc2])
        assert runner.is_complete(response) is False


class TestPrintCompletionSummary:
    @patch("mini_agent.core.step_runner.perf_counter")
    def test_prints_step_timing(self, mock_perf, runner, mock_agent, capsys):
        mock_perf.side_effect = [110.0, 115.0]
        mock_agent.api_call_count = 3
        runner.print_completion_summary(step=2, step_start_time=105.0)
        printed = capsys.readouterr().out
        assert "Step 3 completed in" in printed
        assert "5.00s" in printed
        assert "total: 15.00s" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_prints_total_api_calls(self, mock_perf, runner, mock_agent, capsys):
        mock_perf.side_effect = [110.0, 110.0]
        mock_agent.api_call_count = 7
        runner.print_completion_summary(step=0, step_start_time=100.0)
        printed = capsys.readouterr().out
        assert "Total API calls: 7" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_uses_step_plus_one_in_output(self, mock_perf, runner, mock_agent, capsys):
        mock_perf.side_effect = [110.0, 110.0]
        runner.print_completion_summary(step=9, step_start_time=100.0)
        printed = capsys.readouterr().out
        assert "Step 10" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_computes_step_elapsed_from_start_time(self, mock_perf, runner, capsys):
        mock_perf.side_effect = [107.5, 110.0]
        runner.print_completion_summary(step=0, step_start_time=100.0)
        printed = capsys.readouterr().out
        assert "7.50s" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_computes_total_elapsed_from_run_start(self, mock_perf, runner, capsys):
        mock_perf.side_effect = [120.0, 120.0]
        runner.print_completion_summary(step=0, step_start_time=110.0)
        printed = capsys.readouterr().out
        assert "total: 20.00s" in printed


class TestAutoSave:
    def test_does_nothing_when_auto_save_disabled(self, runner, mock_agent):
        mock_agent.auto_save = False
        runner.auto_save(step=5)
        mock_agent._session_manager.save.assert_not_called()

    def test_saves_when_auto_save_enabled_and_interval_reached(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 0
        mock_agent._session_manager.save.return_value = "session_123"
        runner.auto_save(step=3)
        mock_agent._session_manager.save.assert_called_once()

    def test_does_not_save_when_interval_not_reached(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 2
        runner.auto_save(step=3)
        mock_agent._session_manager.save.assert_not_called()

    def test_updates_last_auto_save_step_on_success(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 0
        mock_agent._session_manager.save.return_value = "sid_abc"
        runner.auto_save(step=3)
        assert mock_agent._last_auto_save_step == 3

    def test_does_not_update_step_when_prefix_not_auto_step(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._session_manager.save.return_value = "sid_xyz"
        runner.auto_save(step=1, prefix="manual")
        assert mock_agent._last_auto_save_step == 0

    def test_saves_with_correct_prefix(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 0
        mock_agent._session_manager.save.return_value = "sid_1"
        runner.auto_save(step=5)
        mock_agent._session_manager.save.assert_called_once_with(mock_agent.messages, "auto_step_5")

    def test_saves_with_custom_prefix(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._session_manager.save.return_value = "sid_2"
        runner.auto_save(step=2, prefix="checkpoint")
        mock_agent._session_manager.save.assert_called_once_with(mock_agent.messages, "checkpoint_2")

    def test_prints_session_id_on_success(self, runner, mock_agent, capsys):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 0
        mock_agent._session_manager.save.return_value = "sess_999"
        runner.auto_save(step=3)
        printed = capsys.readouterr().out
        assert "sess_999" in printed

    def test_prints_error_on_save_failure(self, runner, mock_agent, capsys):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 0
        mock_agent._session_manager.save.side_effect = OSError("disk full")
        runner.auto_save(step=3)
        printed = capsys.readouterr().out
        assert "Auto-save failed" in printed
        assert "disk full" in printed

    def test_does_not_raise_on_save_exception(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 0
        mock_agent._session_manager.save.side_effect = RuntimeError("boom")
        runner.auto_save(step=3)

    def test_interval_boundary(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 1
        runner.auto_save(step=3)
        mock_agent._session_manager.save.assert_not_called()
        runner.auto_save(step=4)
        mock_agent._session_manager.save.assert_called_once()

    def test_custom_prefix_always_saves_regardless_of_interval(self, runner, mock_agent):
        mock_agent.auto_save = True
        mock_agent._last_auto_save_step = 10
        mock_agent._session_manager.save.return_value = "sid_custom"
        runner.auto_save(step=11, prefix="manual")
        mock_agent._session_manager.save.assert_called_once()


class TestPrintStepTiming:
    @patch("mini_agent.core.step_runner.perf_counter")
    def test_prints_step_number(self, mock_perf, runner, capsys):
        mock_perf.side_effect = [110.0, 110.0]
        runner.print_step_timing(step=4, step_start_time=100.0)
        printed = capsys.readouterr().out
        assert "Step 5" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_prints_step_elapsed(self, mock_perf, runner, capsys):
        mock_perf.side_effect = [107.3, 110.0]
        runner.print_step_timing(step=0, step_start_time=100.0)
        printed = capsys.readouterr().out
        assert "7.3s" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_prints_total_elapsed(self, mock_perf, runner, capsys):
        mock_perf.side_effect = [120.0, 120.0]
        runner.print_step_timing(step=0, step_start_time=110.0)
        printed = capsys.readouterr().out
        assert "total: 20.0s" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_uses_step_plus_one(self, mock_perf, runner, capsys):
        mock_perf.side_effect = [110.0, 110.0]
        runner.print_step_timing(step=0, step_start_time=100.0)
        printed = capsys.readouterr().out
        assert "Step 1" in printed

    @patch("mini_agent.core.step_runner.perf_counter")
    def test_calls_perf_counter_twice(self, mock_perf, runner):
        mock_perf.side_effect = [110.0, 115.0]
        runner.print_step_timing(step=0, step_start_time=100.0)
        assert mock_perf.call_count == 2
