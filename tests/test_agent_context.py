"""Tests for AgentContext and Message model compatibility."""

import pytest

from mini_agent.schema import FunctionCall, Message, ToolCall


class TestMessageModel:
    """Test Message pydantic model for AgentContext compatibility."""

    def test_message_from_dataclass_conversion(self):
        """Test converting existing Message dataclass to pydantic."""
        # Ensure Message has model_validate and model_dump
        msg = Message(role="user", content="test")
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert restored.role == msg.role
        assert restored.content == msg.content

    def test_message_with_tool_calls_serialization(self):
        """Test Message with tool_calls serializes correctly."""
        tool_call = ToolCall(
            id="test-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"command": "ls"})
        )
        msg = Message(
            role="assistant",
            content="Running command",
            tool_calls=[tool_call]
        )
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].function.name == "bash"

    def test_message_optional_fields(self):
        """Test Message with optional fields."""
        msg = Message(role="user", content="test")
        assert msg.thinking is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_message_with_thinking(self):
        """Test Message with thinking field."""
        msg = Message(
            role="assistant",
            content="I'll help with that",
            thinking="Let me analyze this..."
        )
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert restored.thinking == "Let me analyze this..."


class TestAgentContextWithPydanticMessage:
    """Test AgentContext integration with Pydantic Message model."""

    def test_context_add_message(self):
        """Test adding Message to context."""
        from mini_agent.core.agent_context import AgentContext

        ctx = AgentContext()
        msg = Message(role="user", content="Hello")
        ctx.add_message(msg)
        messages = ctx.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    def test_context_message_serialization_round_trip(self):
        """Test that messages survive serialization round-trip."""
        from mini_agent.core.agent_context import AgentContext

        ctx = AgentContext()
        msg = Message(
            role="assistant",
            content="Response",
            tool_calls=[ToolCall(
                id="call-1",
                type="function",
                function=FunctionCall(name="test", arguments={"arg": "value"})
            )]
        )
        ctx.add_message(msg)

        # Get messages and verify
        messages = ctx.get_messages()
        assert len(messages) == 1
        assert messages[0].tool_calls is not None
        assert len(messages[0].tool_calls) == 1