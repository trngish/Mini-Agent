"""Message validation utilities for agent communication."""

from typing import Any

from ..schema import Message, ToolCall


class ValidationError(Exception):
    """Validation error with details."""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class MessageValidator:
    """Validates message structure before sending to LLM."""

    @staticmethod
    def validate_message(message: Message) -> None:
        """Validate a single message.

        Args:
            message: Message to validate

        Raises:
            ValidationError: If validation fails
        """
        # Validate role
        valid_roles = {"system", "user", "assistant", "tool"}
        if message.role not in valid_roles:
            raise ValidationError(
                f"Invalid role: {message.role}",
                field="role",
                value=message.role,
            )

        # Validate content
        if message.content is None:
            raise ValidationError(
                "Content cannot be None",
                field="content",
                value=message.content,
            )

        # Validate tool_call_id matches role
        if message.tool_call_id and message.role != "tool":
            raise ValidationError(
                f"tool_call_id only valid for tool role, got {message.role}",
                field="tool_call_id",
                value=message.tool_call_id,
            )

        # Validate name field
        if message.name and message.role != "tool":
            raise ValidationError(
                f"name only valid for tool role, got {message.role}",
                field="name",
                value=message.name,
            )

        # Validate tool_calls structure
        if message.tool_calls:
            if message.role not in ("assistant",):
                raise ValidationError(
                    f"tool_calls only valid for assistant role, got {message.role}",
                    field="tool_calls",
                )
            MessageValidator._validate_tool_calls(message.tool_calls)

    @staticmethod
    def _validate_tool_calls(tool_calls: list[ToolCall]) -> None:
        """Validate tool calls list.

        Args:
            tool_calls: List of tool calls to validate

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(tool_calls, list):
            raise ValidationError(
                f"tool_calls must be a list, got {type(tool_calls)}",
                field="tool_calls",
            )

        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, ToolCall):
                raise ValidationError(
                    f"tool_calls[{i}] must be ToolCall, got {type(tc)}",
                    field=f"tool_calls[{i}]",
                )

            if not tc.id:
                raise ValidationError(
                    f"tool_calls[{i}].id is required",
                    field=f"tool_calls[{i}].id",
                )

            if not tc.function:
                raise ValidationError(
                    f"tool_calls[{i}].function is required",
                    field=f"tool_calls[{i}].function",
                )

            if not tc.function.name:
                raise ValidationError(
                    f"tool_calls[{i}].function.name is required",
                    field=f"tool_calls[{i}].function.name",
                )

    @staticmethod
    def validate_messages(messages: list[Message]) -> None:
        """Validate message history.

        Args:
            messages: List of messages to validate

        Raises:
            ValidationError: If validation fails
        """
        if not messages:
            raise ValidationError("Messages list cannot be empty")

        # First message must be system
        if messages[0].role != "system":
            raise ValidationError(
                "First message must be system role",
                field="messages[0].role",
                value=messages[0].role,
            )

        # Validate each message
        for i, msg in enumerate(messages):
            try:
                MessageValidator.validate_message(msg)
            except ValidationError as e:
                raise ValidationError(
                    f"Message {i}: {str(e)}",
                    field=f"messages[{i}].{e.field}" if e.field else None,
                    value=e.value,
                )

    @staticmethod
    def validate_tool_result_match(
        assistant_message: Message,
        tool_message: Message,
    ) -> bool:
        """Validate that tool result matches assistant's tool call.

        Args:
            assistant_message: Assistant message with tool_calls
            tool_message: Tool result message

        Returns:
            True if valid match

        Raises:
            ValidationError: If validation fails
        """
        if assistant_message.role != "assistant":
            raise ValidationError(
                "Assistant message expected",
                field="assistant_message.role",
            )

        if tool_message.role != "tool":
            raise ValidationError(
                "Tool message expected",
                field="tool_message.role",
            )

        if not assistant_message.tool_calls:
            raise ValidationError(
                "Assistant message has no tool_calls",
                field="assistant_message.tool_calls",
            )

        # Check if tool_call_id matches
        for tc in assistant_message.tool_calls:
            if tc.id == tool_message.tool_call_id:
                return True

        raise ValidationError(
            f"tool_call_id '{tool_message.tool_call_id}' not found in assistant tool_calls",
            field="tool_message.tool_call_id",
        )