from mini_agent.utils.tool_error_handler import (
    ToolExecutionError,
    ToolPermissionError,
    ToolResourceError,
    ToolValidationError,
    handle_tool_error,
)


class TestToolExecutionError:
    def test_str_no_original(self):
        err = ToolExecutionError("bash", {"cmd": "ls"}, "failed")
        assert "bash" in str(err)
        assert "failed" in str(err)

    def test_str_with_original(self):
        orig = RuntimeError("boom")
        err = ToolExecutionError("bash", {"cmd": "ls"}, "failed", original_exception=orig)
        s = str(err)
        assert "bash" in s
        assert "RuntimeError" in s

    def test_recoverable_default(self):
        err = ToolExecutionError("t", {}, "msg")
        assert err.recoverable is True

    def test_recoverable_false(self):
        err = ToolExecutionError("t", {}, "msg", recoverable=False)
        assert err.recoverable is False


class TestToolValidationError:
    def test_not_recoverable(self):
        err = ToolValidationError("tool", {"a": 1}, "bad input")
        assert err.recoverable is False


class TestToolPermissionError:
    def test_not_recoverable(self):
        err = ToolPermissionError("tool", {"a": 1}, "denied")
        assert err.recoverable is False


class TestToolResourceError:
    def test_recoverable(self):
        err = ToolResourceError("tool", {"a": 1}, "not found")
        assert err.recoverable is True

    def test_with_original(self):
        orig = FileNotFoundError("gone")
        err = ToolResourceError("tool", {"a": 1}, "not found", original_exception=orig)
        assert err.original_exception is orig


class TestHandleToolError:
    def test_file_not_found_returns_resource_error(self):
        result = handle_tool_error("read", {"path": "x"}, FileNotFoundError("not found"))
        assert isinstance(result, ToolResourceError)

    def test_os_error_returns_resource_error(self):
        result = handle_tool_error("read", {"path": "x"}, OSError("io error"))
        assert isinstance(result, ToolResourceError)

    def test_permission_error_returns_resource_error(self):
        result = handle_tool_error("write", {"path": "x"}, PermissionError("denied"))
        assert isinstance(result, ToolResourceError)

    def test_value_error_returns_validation_error(self):
        result = handle_tool_error("tool", {"a": 1}, ValueError("bad"))
        assert isinstance(result, ToolValidationError)

    def test_generic_error_returns_execution_error(self):
        result = handle_tool_error("tool", {"a": 1}, RuntimeError("boom"))
        assert isinstance(result, ToolExecutionError)
        assert result.recoverable is True

    def test_passes_through_tool_execution_error(self):
        original = ToolExecutionError("tool", {"a": 1}, "already wrapped")
        result = handle_tool_error("tool", {"a": 1}, original)
        assert result is original

    def test_result_has_tool_name(self):
        result = handle_tool_error("my_tool", {"x": 1}, RuntimeError("err"))
        assert result.tool_name == "my_tool"

    def test_result_has_arguments(self):
        args = {"path": "/tmp/test"}
        result = handle_tool_error("tool", args, RuntimeError("err"))
        assert result.arguments == args
