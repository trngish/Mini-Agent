"""Mini Agent - Minimal single agent with basic tools and MCP support."""

from .agent import Agent
from .config import RetryConfig
from .llm import LLMClient
from .retry import RetryConfig as BaseRetryConfig
from .retry import async_retry
from .schema import AgentMode, FunctionCall, LLMProvider, LLMResponse, Message, ToolCall
from .session import SessionManager
from .subagent import SubAgent, SubAgentResult, run_sub_agents
from .team import AgentRole, AgentTeam, MessageBus, RoleConfig, TeamResult
from .tools import (
    BashTool,
    DeepContextTool,
    EditTool,
    MultiBashTool,
    MultiEditTool,
    MultiGrepTool,
    MultiReadTool,
    ReadTool,
    RecallNoteTool,
    SessionNoteTool,
    Tool,
    ToolResult,
    WorkspaceContextTool,
    WriteTool,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Agent",
    "SubAgent",
    "SubAgentResult",
    "run_sub_agents",
    # Agent Team
    "AgentTeam",
    "AgentRole",
    "RoleConfig",
    "MessageBus",
    "TeamResult",
    # LLM
    "LLMClient",
    "LLMProvider",
    # Schema
    "AgentMode",
    "Message",
    "LLMResponse",
    "ToolCall",
    "FunctionCall",
    # Tools
    "Tool",
    "ToolResult",
    "BashTool",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "MultiReadTool",
    "MultiEditTool",
    "MultiGrepTool",
    "MultiBashTool",
    "WorkspaceContextTool",
    "DeepContextTool",
    "SessionNoteTool",
    "RecallNoteTool",
    # Retry - RetryConfig is from config.py for pydantic compatibility
    "RetryConfig",
    "BaseRetryConfig",  # Alias for backward compatibility
    "async_retry",
    # Version
    "__version__",
    # Session
    "SessionManager",
]
