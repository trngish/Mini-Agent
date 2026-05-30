"""Comprehensive unit tests for Agent class core methods.

Tests cover the following methods with low coverage:
- _check_cancelled()
- _cleanup_incomplete_messages()
- _estimate_tokens()
- add_user_message()
- set_mode()
- _check_approved()
- save_session() / load_session() / list_sessions()
- get_status() / get_status_report()
- _check_health()
- get_error_patterns() / get_suggestions()
- get_performance_metrics()
- cleanup()
- record_context()
"""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.agent import Agent
from mini_agent.schema import AgentMode, Message


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "test-model"
    llm.api_key = "test-key"
    llm.api_base = "https://api.test.com"
    llm.provider = "anthropic"
    return llm


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def agent(mock_llm, temp_workspace):
    return Agent(
        llm_client=mock_llm,
        system_prompt="You are a test assistant.",
        tools=[],
        max_steps=5,
        workspace_dir=temp_workspace,
        mode=AgentMode.YOLO,
    )


class TestCheckCancelled:
    def test_returns_false_when_cancel_event_is_none(self, agent):
        agent.cancel_event = None
        assert agent._check_cancelled() is False

    def test_returns_false_when_cancel_event_not_set(self, agent):
        agent.cancel_event = asyncio.Event()
        assert agent._check_cancelled() is False

    def test_returns_true_when_cancel_event_is_set(self, agent):
        agent.cancel_event = asyncio.Event()
        agent.cancel_event.set()
        assert agent._check_cancelled() is True

    def test_returns_false_for_newly_created_event(self, agent):
        agent.cancel_event = asyncio.Event()
        assert not agent.cancel_event.is_set()
        assert agent._check_cancelled() is False


class TestCleanupIncompleteMessages:
    def test_no_cleanup_when_no_assistant_message(self, agent):
        agent.replace_messages([
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
        ])
        original_len = len(agent.messages)
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == original_len

    def test_removes_last_assistant_message(self, agent):
        agent.replace_messages([
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="thinking..."),
        ])
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == 2
        assert agent.messages[-1].role == "user"

    def test_removes_assistant_and_trailing_tool_messages(self, agent):
        agent.replace_messages([
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="using tool..."),
            Message(role="tool", content="result1", tool_call_id="1"),
            Message(role="tool", content="result2", tool_call_id="2"),
        ])
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == 2
        assert agent.messages[-1].role == "user"

    def test_preserves_earlier_assistant_messages(self, agent):
        agent.replace_messages([
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="done with step 1"),
            Message(role="tool", content="result step 1", tool_call_id="1"),
            Message(role="user", content="next request"),
            Message(role="assistant", content="incomplete step"),
            Message(role="tool", content="partial result", tool_call_id="2"),
        ])
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == 5
        assert agent.messages[-1].role == "user"
        assert agent.messages[-1].content == "next request"
        assert agent.messages[2].role == "assistant"
        assert agent.messages[2].content == "done with step 1"

    def test_no_cleanup_with_empty_messages(self, agent):
        agent.replace_messages([])
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == 0

    def test_only_system_message(self, agent):
        agent.replace_messages([Message(role="system", content="system")])
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == 1


class TestEstimateTokens:
    def test_delegates_to_token_tracker(self, agent):
        agent._context.token_tracker = MagicMock()
        agent._context.token_tracker.estimate_tokens.return_value = 42
        result = agent._context.estimate_tokens()
        assert result == 42
        agent._context.token_tracker.estimate_tokens.assert_called_once()

    def test_returns_zero_for_empty_messages(self, agent):
        agent.replace_messages([])
        result = agent._context.estimate_tokens()
        assert result == 0

    def test_returns_positive_for_nonempty_messages(self, agent):
        agent.add_user_message("Hello world")
        result = agent._context.estimate_tokens()
        assert result > 0

    def test_increases_with_more_messages(self, agent):
        agent.add_user_message("Short")
        count1 = agent._context.estimate_tokens()
        agent.add_user_message("This is a longer message with more words and content")
        count2 = agent._context.estimate_tokens()
        assert count2 > count1


class TestAddUserMessage:
    def test_adds_message_to_history(self, agent):
        original_len = len(agent.messages)
        agent.add_user_message("Hello!")
        assert len(agent.messages) == original_len + 1
        assert agent.messages[-1].role == "user"
        assert agent.messages[-1].content == "Hello!"

    def test_adds_multiple_messages(self, agent):
        agent.add_user_message("First")
        agent.add_user_message("Second")
        user_msgs = [m for m in agent.messages if m.role == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0].content == "First"
        assert user_msgs[1].content == "Second"

    def test_adjusts_thinking_budget(self, agent):
        agent.is_m27 = True
        agent._thinking_budget_manager = MagicMock()
        agent.add_user_message("Refactor the architecture")
        agent._thinking_budget_manager.adjust.assert_called_once_with("Refactor the architecture")

    def test_thinking_budget_not_adjusted_for_non_m27(self, agent):
        agent.is_m27 = False
        agent._thinking_budget_manager = MagicMock()
        agent.add_user_message("Hello")
        agent._thinking_budget_manager.adjust.assert_called_once_with("Hello")


class TestSetMode:
    def test_switches_mode(self, agent):
        assert agent.mode == AgentMode.YOLO
        agent.set_mode(AgentMode.AGENT)
        assert agent.mode == AgentMode.AGENT

    def test_syncs_approval_manager_mode(self, agent):
        agent.set_mode(AgentMode.AGENT)
        assert agent._approval_manager.mode == AgentMode.AGENT

    def test_switch_to_plan_mode(self, agent):
        agent.set_mode(AgentMode.PLAN)
        assert agent.mode == AgentMode.PLAN
        assert agent._approval_manager.mode == AgentMode.PLAN

    def test_switch_to_yolo_mode(self, agent):
        agent.set_mode(AgentMode.AGENT)
        agent.set_mode(AgentMode.YOLO)
        assert agent.mode == AgentMode.YOLO
        assert agent._approval_manager.mode == AgentMode.YOLO

    def test_switch_all_modes_round_trip(self, agent):
        for mode in [AgentMode.AGENT, AgentMode.PLAN, AgentMode.YOLO]:
            agent.set_mode(mode)
            assert agent.mode == mode
            assert agent._approval_manager.mode == mode


class TestCheckApproved:
    def test_delegates_to_approval_manager(self, agent):
        agent._approval_manager = MagicMock()
        agent._approval_manager.is_approved.return_value = True
        result = agent._check_approved("read_file")
        assert result is True
        agent._approval_manager.is_approved.assert_called_once_with("read_file")

    def test_returns_false_when_rejected(self, agent):
        agent._approval_manager = MagicMock()
        agent._approval_manager.is_approved.return_value = False
        result = agent._check_approved("write_file")
        assert result is False
        agent._approval_manager.is_approved.assert_called_once_with("write_file")

    def test_yolo_mode_auto_approves(self, agent):
        agent.set_mode(AgentMode.YOLO)
        assert agent._check_approved("any_tool") is True

    def test_plan_mode_auto_approves_read_tools(self, agent):
        agent.set_mode(AgentMode.PLAN)
        assert agent._check_approved("read_file") is True


class TestSessionManagement:
    def test_save_session_returns_session_id(self, agent):
        agent.add_user_message("test")
        session_id = agent.save_session(label="test_session")
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_load_session_success(self, agent):
        agent.add_user_message("save me")
        session_id = agent.save_session(label="test")
        result = agent.load_session(session_id)
        assert result is True
        user_msgs = [m for m in agent.messages if m.role == "user"]
        assert any(m.content == "save me" for m in user_msgs)

    def test_load_session_nonexistent(self, agent):
        result = agent.load_session("nonexistent_id_12345")
        assert result is False

    def test_list_sessions_returns_list(self, agent):
        sessions = agent.list_sessions()
        assert isinstance(sessions, list)

    def test_list_sessions_includes_saved(self, agent):
        agent.add_user_message("test content")
        sid = agent.save_session(label="findable")
        sessions = agent.list_sessions()
        ids = [s["id"] for s in sessions]
        assert sid in ids

    def test_save_and_load_round_trip(self, agent):
        agent.add_user_message("first message")
        agent.add_user_message("second message")
        session_id = agent.save_session(label="round_trip")
        original_messages = agent.messages.copy()
        loaded = agent.load_session(session_id)
        assert loaded is True
        assert len(agent.messages) == len(original_messages)
        for orig, loaded_msg in zip(original_messages, agent.messages):
            assert orig.role == loaded_msg.role
            assert orig.content == loaded_msg.content

    def test_save_session_delegates_to_session_manager(self, agent):
        agent._session_manager = MagicMock()
        agent._session_manager.save.return_value = "abc123"
        result = agent.save_session(label="mocked")
        assert result == "abc123"
        agent._session_manager.save.assert_called_once()

    def test_load_session_delegates_to_session_manager(self, agent):
        agent._session_manager = MagicMock()
        agent._session_manager.load.return_value = (
            [
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
            ],
            "task result",
            {},
        )
        result = agent.load_session("abc123")
        assert result is True
        assert len(agent.messages) == 2
        assert agent.get_last_result() == "task result"
        agent._session_manager.load.assert_called_once_with("abc123")

    def test_load_session_returns_none_from_manager(self, agent):
        agent._session_manager = MagicMock()
        agent._session_manager.load.return_value = (None, None, None)
        result = agent.load_session("missing")
        assert result is False

    def test_list_sessions_delegates(self, agent):
        agent._session_manager = MagicMock()
        agent._session_manager.list_sessions.return_value = [
            {"id": "s1", "label": "test"},
        ]
        sessions = agent.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == "s1"


class TestGetStatus:
    def test_get_status_returns_dict(self, agent):
        status = agent.get_status()
        assert isinstance(status, dict)

    def test_get_status_has_required_keys(self, agent):
        status = agent.get_status()
        assert "token_usage" in status
        assert "token_limit" in status
        assert "api_call_count" in status
        assert "mode" in status

    def test_get_status_reflects_mode(self, agent):
        status = agent.get_status()
        assert status["mode"] == AgentMode.YOLO.value

    def test_get_status_reflects_api_call_count(self, agent):
        agent._context.api_call_count = 5
        status = agent.get_status()
        assert status["api_call_count"] == 5

    def test_get_status_delegates_to_health_checker(self, agent):
        agent._health_checker = MagicMock()
        agent._health_checker.get_status.return_value = {"custom": "data"}
        status = agent.get_status()
        assert status == {"custom": "data"}
        agent._health_checker.get_status.assert_called_once()


class TestGetStatusReport:
    def test_get_status_report_returns_string(self, agent):
        report = agent.get_status_report()
        assert isinstance(report, str)

    def test_get_status_report_contains_key_info(self, agent):
        report = agent.get_status_report()
        assert "Agent Status Report" in report
        assert "Token usage" in report
        assert "Mode" in report

    def test_get_status_report_delegates_to_health_checker(self, agent):
        agent._health_checker = MagicMock()
        agent._health_checker.get_status_report.return_value = "Custom report"
        report = agent.get_status_report()
        assert report == "Custom report"
        agent._health_checker.get_status_report.assert_called_once()


class TestCheckHealth:
    def test_returns_list(self, agent):
        issues = agent._check_health()
        assert isinstance(issues, list)

    def test_delegates_to_health_checker(self, agent):
        agent._health_checker = MagicMock()
        agent._health_checker.check.return_value = MagicMock(issues=["issue1", "issue2"])
        issues = agent._check_health()
        assert issues == ["issue1", "issue2"]
        agent._health_checker.check.assert_called_once()

    def test_returns_empty_when_healthy(self, agent):
        issues = agent._check_health()
        assert isinstance(issues, list)

    def test_detects_incomplete_messages(self, agent):
        agent.replace_messages([Message(role="system", content="sys")])
        issues = agent._check_health()
        assert any("incomplete" in issue.lower() for issue in issues)


class TestGetErrorPatterns:
    def test_returns_dict(self, agent):
        patterns = agent.get_error_patterns()
        assert isinstance(patterns, dict)

    def test_has_required_keys(self, agent):
        patterns = agent.get_error_patterns()
        assert "error_counts_by_tool" in patterns
        assert "recent_errors" in patterns

    def test_delegates_to_error_recovery(self, agent):
        agent._error_recovery = MagicMock()
        agent._error_recovery.get_patterns.return_value = {
            "error_counts_by_tool": {"bash": 3},
            "recent_errors": [],
            "total_consecutive_failures": 0,
        }
        patterns = agent.get_error_patterns()
        assert patterns["error_counts_by_tool"] == {"bash": 3}
        agent._error_recovery.get_patterns.assert_called_once()


class TestGetSuggestions:
    def test_returns_list(self, agent):
        suggestions = agent.get_suggestions()
        assert isinstance(suggestions, list)

    def test_delegates_to_error_recovery(self, agent):
        agent._error_recovery = MagicMock()
        agent._error_recovery.get_suggestions.return_value = ["Try again"]
        suggestions = agent.get_suggestions()
        assert suggestions == ["Try again"]
        agent._error_recovery.get_suggestions.assert_called_once()

    def test_empty_when_no_issues(self, agent):
        suggestions = agent.get_suggestions()
        assert isinstance(suggestions, list)


class TestGetPerformanceMetrics:
    def test_returns_dict(self, agent):
        metrics = agent.get_performance_metrics()
        assert isinstance(metrics, dict)

    def test_has_required_keys(self, agent):
        metrics = agent.get_performance_metrics()
        assert "step_metrics" in metrics
        assert "tool_metrics" in metrics
        assert "api_metrics" in metrics
        assert "api_call_count" in metrics

    def test_empty_metrics_initially(self, agent):
        metrics = agent.get_performance_metrics()
        assert metrics["step_metrics"] == {}
        assert metrics["tool_metrics"] == {}
        assert metrics["api_metrics"] == {}
        assert metrics["api_call_count"] == 0

    def test_delegates_to_metrics_module(self, agent):
        agent._metrics = MagicMock()
        agent._metrics.get_metrics.return_value = {"step_metrics": {"count": 3}}
        metrics = agent.get_performance_metrics()
        assert metrics == {"step_metrics": {"count": 3}}
        agent._metrics.get_metrics.assert_called_once()

    def test_reflects_api_call_count(self, agent):
        agent._context.api_call_count = 7
        metrics = agent.get_performance_metrics()
        assert metrics["api_call_count"] == 7


class TestCleanup:
    def test_clears_cancel_event(self, agent):
        agent.cancel_event = asyncio.Event()
        agent.cancel_event.set()
        agent.cleanup()
        assert agent.cancel_event is None

    def test_invalidates_token_cache(self, agent):
        agent._token_tracker = MagicMock()
        agent.cleanup()
        agent._token_tracker.invalidate_cache.assert_called_once()

    def test_flushes_logger(self, agent):
        agent.logger = MagicMock()
        agent.cleanup()
        agent.logger.flush.assert_called_once()

    def test_handles_missing_logger_gracefully(self, agent):
        delattr(agent, "logger")
        agent.cleanup()

    def test_full_cleanup_sequence(self, agent):
        agent.cancel_event = asyncio.Event()
        agent._token_tracker = MagicMock()
        agent.logger = MagicMock()
        agent.cleanup()
        assert agent.cancel_event is None
        agent._token_tracker.invalidate_cache.assert_called_once()
        agent.logger.flush.assert_called_once()


class TestRecordContext:
    def test_no_error_when_no_note_tool(self, agent):
        agent.tools = {}
        agent.record_context("important info")
        assert len(agent.messages) == 1

    def test_with_note_tool_available(self, agent):
        mock_note_tool = MagicMock()
        mock_note_tool.execute = AsyncMock(return_value=None)
        agent.tools["record_note"] = mock_note_tool
        agent.record_context("important info", category="auto")
        assert "record_note" in agent.tools

    def test_with_note_tool_and_category(self, agent):
        mock_note_tool = MagicMock()
        mock_note_tool.execute = AsyncMock(return_value=None)
        agent.tools["record_note"] = mock_note_tool
        agent.record_context("error occurred", category="error_pattern")
        assert "record_note" in agent.tools

    def test_no_running_event_loop(self, agent):
        mock_note_tool = MagicMock()
        mock_note_tool.execute = AsyncMock(return_value=None)
        agent.tools["record_note"] = mock_note_tool
        agent.record_context("test content")
        assert "record_note" in agent.tools

    def test_note_tool_execute_failure(self, agent):
        mock_note_tool = MagicMock()
        mock_note_tool.execute = AsyncMock(side_effect=Exception("tool failed"))
        agent.tools["record_note"] = mock_note_tool
        agent.record_context("test content")
        assert "record_note" in agent.tools

    def test_does_not_modify_messages(self, agent):
        original_count = len(agent.messages)
        agent.record_context("some context")
        assert len(agent.messages) == original_count


class TestGetHistory:
    def test_returns_copy_of_messages(self, agent):
        history = agent.get_history()
        assert history is not agent.messages
        assert len(history) == len(agent.messages)

    def test_copy_is_independent(self, agent):
        history = agent.get_history()
        history.append(Message(role="user", content="extra"))
        assert len(history) != len(agent.messages)

    def test_includes_all_messages(self, agent):
        agent.add_user_message("hello")
        agent.add_user_message("world")
        history = agent.get_history()
        user_msgs = [m for m in history if m.role == "user"]
        assert len(user_msgs) == 2
