"""Mini Agent - Minimal single agent with basic tools and MCP support."""

from .agent import Agent
from .llm import LLMClient
from .schema import AgentMode, FunctionCall, LLMProvider, LLMResponse, Message, ToolCall
from .retry import async_retry
from .retry import RetryConfig as BaseRetryConfig
from .config import RetryConfig
from .subagent import SubAgent, SubAgentResult, run_sub_agents

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Agent",
    "SubAgent",
    "SubAgentResult",
    "run_sub_agents",
    # LLM
    "LLMClient",
    "LLMProvider",
    # Schema
    "AgentMode",
    "Message",
    "LLMResponse",
    "ToolCall",
    "FunctionCall",
    # Retry - RetryConfig is from config.py for pydantic compatibility
    "RetryConfig",
    "BaseRetryConfig",  # Alias for backward compatibility
    "async_retry",
    # Version
    "__version__",
]
