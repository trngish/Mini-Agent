from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mini_agent.logger import AgentLogger
from mini_agent.schema import FunctionCall, Message, ToolCall


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_logs"


@pytest.fixture
def logger(log_dir: Path) -> AgentLogger:
    with patch.object(AgentLogger, "LOG_DIR", log_dir):
        agent_logger = AgentLogger()
    return agent_logger


class TestAgentLoggerInit:
    def test_log_dir_created(self, log_dir: Path) -> None:
        with patch.object(AgentLogger, "LOG_DIR", log_dir):
            AgentLogger()
        assert log_dir.exists()

    def test_log_disabled_defaults_to_false(self, logger: AgentLogger) -> None:
        assert logger._log_disabled is False

    def test_permission_error_on_mkdir_sets_disabled(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "restricted"
        with (
            patch.object(AgentLogger, "LOG_DIR", log_dir),
            patch.object(Path, "mkdir", side_effect=PermissionError("denied")),
        ):
            agent_logger = AgentLogger()
        assert agent_logger._log_disabled is True


class TestAgentLoggerPermissionError:
    def test_graceful_degradation_mkdir_fails(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "no_access"
        with (
            patch.object(AgentLogger, "LOG_DIR", log_dir),
            patch.object(Path, "mkdir", side_effect=PermissionError("no access")),
        ):
            agent_logger = AgentLogger()
        assert agent_logger._log_disabled is True
        assert agent_logger.log_file is None

    def test_graceful_degradation_file_write_fails(self, logger: AgentLogger) -> None:
        logger.start_new_run()
        assert logger.log_file is not None
        with patch("builtins.open", side_effect=PermissionError("write denied")):
            logger._write_log_sync("test entry")
        assert logger._log_disabled is True
        assert logger.log_file is None


class TestStartNewRun:
    def test_creates_log_file(self, logger: AgentLogger) -> None:
        logger.start_new_run()
        assert logger.log_file is not None
        assert logger.log_file.exists()
        content = logger.log_file.read_text(encoding="utf-8")
        assert "Agent Run Log" in content

    def test_resets_log_index(self, logger: AgentLogger) -> None:
        logger.log_index = 5
        logger.start_new_run()
        assert logger.log_index == 0

    def test_resets_current_size(self, logger: AgentLogger) -> None:
        logger._current_size = 9999
        logger.start_new_run()
        assert logger._current_size > 0
        assert logger._current_size < 9999

    def test_handles_permission_error(self, logger: AgentLogger) -> None:
        with patch("builtins.open", side_effect=PermissionError("denied")):
            logger.start_new_run()
        assert logger._log_disabled is True
        assert logger.log_file is None

    def test_skips_when_disabled(self, log_dir: Path) -> None:
        with patch.object(AgentLogger, "LOG_DIR", log_dir):
            agent_logger = AgentLogger()
        agent_logger._log_disabled = True
        agent_logger.start_new_run()
        assert agent_logger.log_file is None


class TestEnsureLogFile:
    def test_skips_when_log_disabled(self, logger: AgentLogger) -> None:
        logger._log_disabled = True
        logger._ensure_log_file()
        assert logger.log_file is None

    def test_skips_when_rotation_check_done_and_file_exists(self, logger: AgentLogger) -> None:
        logger.start_new_run()
        existing_file = logger.log_file
        logger._rotation_check_done = True
        logger._ensure_log_file()
        assert logger.log_file == existing_file

    def test_calls_start_new_run_when_no_file(self, logger: AgentLogger) -> None:
        assert logger.log_file is None
        with patch.object(logger, "_cleanup_old_logs"):
            logger._ensure_log_file()
        assert logger.log_file is not None


class TestWriteLogSync:
    def test_writes_entry(self, logger: AgentLogger) -> None:
        logger.start_new_run()
        entry = "test log entry\n"
        logger._write_log_sync(entry)
        content = logger.log_file.read_text(encoding="utf-8")
        assert "test log entry" in content

    def test_handles_permission_error(self, logger: AgentLogger) -> None:
        logger.start_new_run()
        with patch("builtins.open", side_effect=PermissionError("denied")):
            logger._write_log_sync("entry")
        assert logger._log_disabled is True
        assert logger.log_file is None

    def test_skips_when_disabled(self, logger: AgentLogger) -> None:
        logger._log_disabled = True
        logger._write_log_sync("should not write")

    def test_skips_when_no_log_file(self, logger: AgentLogger) -> None:
        logger.log_file = None
        logger._write_log_sync("should not write")


class TestLogRequest:
    def test_with_messages(self, logger: AgentLogger) -> None:
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_request(messages)
        assert logger.log_file is not None
        content = logger.log_file.read_text(encoding="utf-8")
        assert "REQUEST" in content
        assert "LLM Request" in content
        parsed = json.loads(content.split("LLM Request:\n\n", 1)[1])
        assert len(parsed["messages"]) == 2
        assert parsed["messages"][0]["role"] == "system"
        assert parsed["messages"][1]["role"] == "user"

    def test_with_tools(self, logger: AgentLogger) -> None:
        messages = [Message(role="user", content="test")]
        mock_tool = MagicMock()
        mock_tool.name = "bash"
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_request(messages, tools=[mock_tool])
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("LLM Request:\n\n", 1)[1])
        assert "bash" in parsed["tools"]

    def test_with_tool_calls_in_message(self, logger: AgentLogger) -> None:
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="bash", arguments={"cmd": "ls"}))
        messages = [Message(role="assistant", content="", tool_calls=[tc])]
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_request(messages)
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("LLM Request:\n\n", 1)[1])
        assert parsed["messages"][0]["tool_calls"][0]["id"] == "tc-1"

    def test_with_thinking_in_message(self, logger: AgentLogger) -> None:
        messages = [Message(role="assistant", content="result", thinking="hmm")]
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_request(messages)
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("LLM Request:\n\n", 1)[1])
        assert parsed["messages"][0]["thinking"] == "hmm"


class TestLogResponse:
    def test_with_response(self, logger: AgentLogger) -> None:
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_response(content="Hello world")
        content = logger.log_file.read_text(encoding="utf-8")
        assert "RESPONSE" in content
        assert "LLM Response" in content
        parsed = json.loads(content.split("LLM Response:\n\n", 1)[1])
        assert parsed["content"] == "Hello world"

    def test_with_thinking(self, logger: AgentLogger) -> None:
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_response(content="answer", thinking="deep thought")
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("LLM Response:\n\n", 1)[1])
        assert parsed["thinking"] == "deep thought"

    def test_with_tool_calls(self, logger: AgentLogger) -> None:
        tc = ToolCall(id="tc-2", type="function", function=FunctionCall(name="read", arguments={"path": "a.txt"}))
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_response(content="", tool_calls=[tc])
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("LLM Response:\n\n", 1)[1])
        assert parsed["tool_calls"][0]["id"] == "tc-2"

    def test_with_finish_reason(self, logger: AgentLogger) -> None:
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_response(content="done", finish_reason="stop")
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("LLM Response:\n\n", 1)[1])
        assert parsed["finish_reason"] == "stop"


class TestLogToolResult:
    def test_with_tool_result(self, logger: AgentLogger) -> None:
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_tool_result(
                tool_name="bash",
                arguments={"cmd": "ls"},
                result_success=True,
                result_content="file1.txt\nfile2.txt",
            )
        content = logger.log_file.read_text(encoding="utf-8")
        assert "TOOL_RESULT" in content
        assert "Tool Execution" in content
        parsed = json.loads(content.split("Tool Execution:\n\n", 1)[1])
        assert parsed["tool_name"] == "bash"
        assert parsed["success"] is True
        assert parsed["result"] == "file1.txt\nfile2.txt"

    def test_with_error(self, logger: AgentLogger) -> None:
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_tool_result(
                tool_name="bash",
                arguments={"cmd": "rm -rf /"},
                result_success=False,
                result_error="Permission denied",
            )
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("Tool Execution:\n\n", 1)[1])
        assert parsed["success"] is False
        assert parsed["error"] == "Permission denied"

    def test_success_has_no_error_key(self, logger: AgentLogger) -> None:
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_tool_result(
                tool_name="read",
                arguments={},
                result_success=True,
                result_content="ok",
            )
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("Tool Execution:\n\n", 1)[1])
        assert "error" not in parsed

    def test_failure_has_no_result_key(self, logger: AgentLogger) -> None:
        with patch.object(logger, "_cleanup_old_logs"):
            logger.log_tool_result(
                tool_name="read",
                arguments={},
                result_success=False,
                result_error="fail",
            )
        content = logger.log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.split("Tool Execution:\n\n", 1)[1])
        assert "result" not in parsed


class TestLogDisabledFlag:
    def test_prevents_start_new_run(self, log_dir: Path) -> None:
        with patch.object(AgentLogger, "LOG_DIR", log_dir):
            agent_logger = AgentLogger()
        agent_logger._log_disabled = True
        agent_logger.start_new_run()
        assert agent_logger.log_file is None

    def test_prevents_ensure_log_file(self, logger: AgentLogger) -> None:
        logger._log_disabled = True
        logger._ensure_log_file()
        assert logger.log_file is None

    def test_prevents_write_log_sync(self, logger: AgentLogger) -> None:
        logger.start_new_run()
        file_before = logger.log_file
        logger._log_disabled = True
        logger._write_log_sync("should not appear")
        content = file_before.read_text(encoding="utf-8")
        assert "should not appear" not in content

    def test_cascades_from_init_permission_error(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "blocked"
        with (
            patch.object(AgentLogger, "LOG_DIR", log_dir),
            patch.object(Path, "mkdir", side_effect=PermissionError("no")),
        ):
            agent_logger = AgentLogger()
        assert agent_logger._log_disabled is True
        agent_logger.start_new_run()
        assert agent_logger.log_file is None
        agent_logger._ensure_log_file()
        assert agent_logger.log_file is None

    def test_cascades_from_write_permission_error(self, logger: AgentLogger) -> None:
        logger.start_new_run()
        assert logger.log_file is not None
        with patch("builtins.open", side_effect=PermissionError("write blocked")):
            logger._write_log_sync("entry")
        assert logger._log_disabled is True
        assert logger.log_file is None
        logger.start_new_run()
        assert logger.log_file is None
