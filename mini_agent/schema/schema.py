from enum import Enum
from typing import Any

from pydantic import BaseModel


class LLMProvider(str, Enum):
    """LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class AgentMode(str, Enum):
    """Agent execution mode."""

    PLAN = "plan"  # Read-only, model proposes before making changes
    AGENT = "agent"  # Interactive with approval gates
    YOLO = "yolo"  # Auto-approve all tools


# Tools considered as write operations (blocked in PLAN mode)
WRITE_TOOLS = frozenset({"write_file", "edit_file", "bash", "git", "multi_edit", "multi_bash"})


class FunctionCall(BaseModel):
    """Function call details."""

    name: str
    arguments: dict[str, Any]  # Function arguments as dict


class ToolCall(BaseModel):
    """Tool call structure."""

    id: str
    type: str  # "function"
    function: FunctionCall


class Message(BaseModel):
    """Chat message."""

    id: str | None = None  # 消息唯一标识，便于跨对话引用
    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # Can be string or list of content blocks
    thinking: str | None = None  # Extended thinking content for assistant messages
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool role
    metadata: dict[str, Any] | None = None  # 额外信息，如标记为"建议#1"


class TokenUsage(BaseModel):
    """Token usage statistics from LLM API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """LLM response."""

    content: str
    thinking: str | None = None  # Extended thinking blocks
    tool_calls: list[ToolCall] | None = None
    finish_reason: str
    usage: TokenUsage | None = None  # Token usage from API response
