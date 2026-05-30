"""用于Agent通信的消息验证工具。"""

from typing import Any

from ..schema import Message, ToolCall


class ValidationError(Exception):
    """带详细信息的验证错误。"""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class MessageValidator:
    """在发送到LLM之前验证消息结构。"""

    @staticmethod
    def validate_message(message: Message) -> None:
        """验证单条消息。

        Args:
            message: 要验证的消息

        Raises:
            ValidationError: 如果验证失败
        """
        # 验证role
        valid_roles = {"system", "user", "assistant", "tool"}
        if message.role not in valid_roles:
            raise ValidationError(
                f"Invalid role: {message.role}",
                field="role",
                value=message.role,
            )

        # 验证content
        if message.content is None:
            raise ValidationError(
                "Content cannot be None",
                field="content",
                value=message.content,
            )

        # 验证tool_call_id与role的匹配
        if message.tool_call_id and message.role != "tool":
            raise ValidationError(
                f"tool_call_id only valid for tool role, got {message.role}",
                field="tool_call_id",
                value=message.tool_call_id,
            )

        # 验证name字段
        if message.name and message.role != "tool":
            raise ValidationError(
                f"name only valid for tool role, got {message.role}",
                field="name",
                value=message.name,
            )

        # 验证tool_calls结构
        if message.tool_calls:
            if message.role not in ("assistant",):
                raise ValidationError(
                    f"tool_calls only valid for assistant role, got {message.role}",
                    field="tool_calls",
                )
            MessageValidator._validate_tool_calls(message.tool_calls)

    @staticmethod
    def _validate_tool_calls(tool_calls: list[ToolCall]) -> None:
        """验证工具调用列表。

        Args:
            tool_calls: 要验证的工具调用列表

        Raises:
            ValidationError: 如果验证失败
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
        """验证消息历史。

        Args:
            messages: 要验证的消息列表

        Raises:
            ValidationError: 如果验证失败
        """
        if not messages:
            raise ValidationError("Messages list cannot be empty")

        # 第一条消息必须是system
        if messages[0].role != "system":
            raise ValidationError(
                "First message must be system role",
                field="messages[0].role",
                value=messages[0].role,
            )

        # 验证每条消息
        for i, msg in enumerate(messages):
            try:
                MessageValidator.validate_message(msg)
            except ValidationError as e:
                raise ValidationError(
                    f"Message {i}: {str(e)}",
                    field=f"messages[{i}].{e.field}" if e.field else None,
                    value=e.value,
                ) from None

    @staticmethod
    def validate_tool_result_match(
        assistant_message: Message,
        tool_message: Message,
    ) -> bool:
        """验证工具结果与Assistant的工具调用是否匹配。

        Args:
            assistant_message: 包含tool_calls的Assistant消息
            tool_message: 工具结果消息

        Returns:
            如果匹配则返回True

        Raises:
            ValidationError: 如果验证失败
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

        # 检查tool_call_id是否匹配
        for tc in assistant_message.tool_calls:
            if tc.id == tool_message.tool_call_id:
                return True

        raise ValidationError(
            f"tool_call_id '{tool_message.tool_call_id}' not found in assistant tool_calls",
            field="tool_message.tool_call_id",
        )
