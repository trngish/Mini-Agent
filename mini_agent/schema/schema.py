from enum import Enum
from typing import Any

from pydantic import BaseModel


class LLMProvider(str, Enum):
    """LLM 提供商类型"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class AgentMode(str, Enum):
    """Agent 执行模式"""

    PLAN = "plan"  # 只读模式，模型在执行操作前先提出建议
    AGENT = "agent"  # 交互模式，需要Approval门控
    YOLO = "yolo"  # 自动批准所有工具调用


# 被视为写操作的工具（在 PLAN 模式下会被阻止）
WRITE_TOOLS = frozenset({"write_file", "edit_file", "bash", "git", "multi_edit", "multi_bash"})


class FunctionCall(BaseModel):
    """函数调用详情"""

    name: str
    arguments: dict[str, Any]  # 函数参数，字典形式


class ToolCall(BaseModel):
    """工具调用结构"""

    id: str
    type: str  # "function"
    function: FunctionCall


class Message(BaseModel):
    """聊天消息"""

    id: str | None = None  # 消息唯一标识，便于跨对话引用
    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # 可以是字符串或内容块列表
    thinking: str | None = None  # 助手消息的扩展思考内容
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # 用于工具角色
    metadata: dict[str, Any] | None = None  # 额外信息，如标记为"建议#1"


class TokenUsage(BaseModel):
    """LLM API 响应中的 Token 使用统计"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """LLM 响应"""

    content: str
    thinking: str | None = None  # 扩展思考块
    tool_calls: list[ToolCall] | None = None
    finish_reason: str
    usage: TokenUsage | None = None  # API 响应的 Token 使用量
