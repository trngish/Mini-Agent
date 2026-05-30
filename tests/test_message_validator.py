import pytest

from mini_agent.schema import FunctionCall, Message, ToolCall
from mini_agent.utils.message_validator import MessageValidator, ValidationError


class TestValidateMessage:
    def test_valid_system_message(self):
        msg = Message(role="system", content="You are helpful")
        MessageValidator.validate_message(msg)

    def test_valid_user_message(self):
        msg = Message(role="user", content="Hello")
        MessageValidator.validate_message(msg)

    def test_valid_assistant_message(self):
        msg = Message(role="assistant", content="Hi there")
        MessageValidator.validate_message(msg)

    def test_valid_tool_message(self):
        msg = Message(role="tool", content="result", tool_call_id="tc-1", name="bash")
        MessageValidator.validate_message(msg)

    def test_invalid_role(self):
        msg = Message(role="invalid", content="test")
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator.validate_message(msg)
        assert "Invalid role" in str(exc_info.value)

    def test_none_content(self):
        msg = Message(role="user", content="test")
        msg.content = None
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator.validate_message(msg)
        assert "None" in str(exc_info.value)

    def test_tool_call_id_on_non_tool_role(self):
        msg = Message(role="user", content="hi", tool_call_id="tc-1")
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator.validate_message(msg)
        assert "tool_call_id" in str(exc_info.value)

    def test_name_on_non_tool_role(self):
        msg = Message(role="user", content="hi", name="bash")
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator.validate_message(msg)
        assert "name" in str(exc_info.value)

    def test_tool_calls_on_non_assistant_role(self):
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="bash", arguments={"cmd": "ls"}))
        msg = Message(role="user", content="hi", tool_calls=[tc])
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator.validate_message(msg)
        assert "tool_calls" in str(exc_info.value)

    def test_valid_tool_calls_on_assistant(self):
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="bash", arguments={"cmd": "ls"}))
        msg = Message(role="assistant", content="", tool_calls=[tc])
        MessageValidator.validate_message(msg)


class TestValidateToolCalls:
    def test_valid_tool_call(self):
        tc = ToolCall(
            id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={"path": "test.txt"})
        )
        MessageValidator._validate_tool_calls([tc])

    def test_empty_tool_id(self):
        tc = ToolCall(id="", type="function", function=FunctionCall(name="read_file", arguments={}))
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator._validate_tool_calls([tc])
        assert "id is required" in str(exc_info.value)

    def test_empty_function_name(self):
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="", arguments={}))
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator._validate_tool_calls([tc])
        assert "name is required" in str(exc_info.value)

    def test_no_function(self):
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="test", arguments={}))
        tc.function = None
        with pytest.raises(ValidationError):
            MessageValidator._validate_tool_calls([tc])


class TestValidateMessages:
    def test_valid_messages(self):
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]
        MessageValidator.validate_messages(messages)

    def test_empty_messages(self):
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator.validate_messages([])
        assert "empty" in str(exc_info.value).lower() or "空" in str(exc_info.value)

    def test_first_not_system(self):
        messages = [Message(role="user", content="Hello")]
        with pytest.raises(ValidationError) as exc_info:
            MessageValidator.validate_messages(messages)
        assert "system" in str(exc_info.value).lower()


class TestValidateToolResultMatch:
    def test_valid_match(self):
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="bash", arguments={"cmd": "ls"}))
        assistant = Message(role="assistant", content="", tool_calls=[tc])
        tool = Message(role="tool", content="result", tool_call_id="tc-1", name="bash")
        assert MessageValidator.validate_tool_result_match(assistant, tool) is True

    def test_non_assistant_message(self):
        msg = Message(role="user", content="hi")
        tool = Message(role="tool", content="result", tool_call_id="tc-1", name="bash")
        with pytest.raises(ValidationError):
            MessageValidator.validate_tool_result_match(msg, tool)

    def test_non_tool_message(self):
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="bash", arguments={}))
        assistant = Message(role="assistant", content="", tool_calls=[tc])
        tool = Message(role="user", content="hi")
        with pytest.raises(ValidationError):
            MessageValidator.validate_tool_result_match(assistant, tool)

    def test_no_tool_calls_in_assistant(self):
        assistant = Message(role="assistant", content="no tools")
        tool = Message(role="tool", content="result", tool_call_id="tc-1", name="bash")
        with pytest.raises(ValidationError):
            MessageValidator.validate_tool_result_match(assistant, tool)

    def test_mismatched_tool_call_id(self):
        tc = ToolCall(id="tc-1", type="function", function=FunctionCall(name="bash", arguments={}))
        assistant = Message(role="assistant", content="", tool_calls=[tc])
        tool = Message(role="tool", content="result", tool_call_id="tc-999", name="bash")
        with pytest.raises(ValidationError):
            MessageValidator.validate_tool_result_match(assistant, tool)


class TestValidationError:
    def test_has_field_and_value(self):
        err = ValidationError("bad", field="role", value="invalid")
        assert err.field == "role"
        assert err.value == "invalid"
        assert "bad" in str(err)

    def test_no_field(self):
        err = ValidationError("something wrong")
        assert err.field is None
        assert err.value is None
