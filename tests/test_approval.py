from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from mini_agent.core.approval import ApprovalManager
from mini_agent.schema import AgentMode


class TestApprovalManagerInit:
    def test_default_mode_is_yolo(self):
        mgr = ApprovalManager()
        assert mgr.mode == AgentMode.YOLO

    def test_default_write_tools_empty(self):
        mgr = ApprovalManager()
        assert mgr.is_write_tool("write_file") is False

    def test_custom_mode(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        assert mgr.mode == AgentMode.AGENT

    def test_custom_write_tools(self):
        mgr = ApprovalManager(write_tools={"write_file", "edit_file"})
        assert mgr.is_write_tool("write_file") is True
        assert mgr.is_write_tool("edit_file") is True

    def test_default_timeout(self):
        mgr = ApprovalManager()
        assert mgr._timeout == ApprovalManager.DEFAULT_TIMEOUT


class TestApprovalManagerModeProperty:
    def test_mode_getter(self):
        mgr = ApprovalManager(mode=AgentMode.PLAN)
        assert mgr.mode == AgentMode.PLAN

    def test_mode_setter(self):
        mgr = ApprovalManager()
        mgr.mode = AgentMode.AGENT
        assert mgr.mode == AgentMode.AGENT


class TestApprovalManagerIsApproved:
    def test_yolo_mode_returns_true(self):
        mgr = ApprovalManager(mode=AgentMode.YOLO)
        assert mgr.is_approved("any_tool") is True

    def test_plan_mode_returns_true(self):
        mgr = ApprovalManager(mode=AgentMode.PLAN)
        assert mgr.is_approved("any_tool") is True

    def test_agent_mode_approve_with_y(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        with patch("builtins.input", return_value="y"):
            assert mgr.is_approved("some_tool") is True

    def test_agent_mode_approve_with_empty(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        with patch("builtins.input", return_value=""):
            assert mgr.is_approved("some_tool") is True

    def test_agent_mode_deny_with_n(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        with patch("builtins.input", return_value="n"):
            assert mgr.is_approved("some_tool") is False

    def test_agent_mode_deny_with_no(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        with patch("builtins.input", return_value="no"):
            assert mgr.is_approved("some_tool") is False

    def test_agent_mode_deny_with_q(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        with patch("builtins.input", return_value="q"):
            assert mgr.is_approved("some_tool") is False

    def test_agent_mode_deny_with_quit(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        with patch("builtins.input", return_value="quit"):
            assert mgr.is_approved("some_tool") is False

    def test_agent_mode_timeout_returns_false(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        mgr._timeout = 0
        with patch("builtins.input", side_effect=lambda *a, **k: __import__("time").sleep(10)):
            result = mgr.is_approved("some_tool")
            assert result is False

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_agent_mode_exception_returns_false(self):
        mgr = ApprovalManager(mode=AgentMode.AGENT)
        with patch("builtins.input", side_effect=RuntimeError("boom")):
            result = mgr.is_approved("some_tool")
            assert result is False


class TestApprovalManagerIsWriteTool:
    def test_write_tool_in_set(self):
        mgr = ApprovalManager(write_tools={"write_file", "edit_file"})
        assert mgr.is_write_tool("write_file") is True

    def test_non_write_tool_not_in_set(self):
        mgr = ApprovalManager(write_tools={"write_file", "edit_file"})
        assert mgr.is_write_tool("read_file") is False

    def test_empty_write_tools(self):
        mgr = ApprovalManager(write_tools=set())
        assert mgr.is_write_tool("write_file") is False


class TestApprovalManagerDefaultTimeout:
    def test_default_timeout_value(self):
        assert ApprovalManager.DEFAULT_TIMEOUT == 10

    def test_timeout_from_env(self):
        with patch.dict(os.environ, {"MINI_AGENT_APPROVAL_TIMEOUT": "30"}):
            mgr = ApprovalManager()
            assert mgr._timeout == 30
