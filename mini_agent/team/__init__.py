"""Agent 团队 - 多智能体协作，包含角色边界和对抗性推理。"""

from .agent_team import AgentMember, AgentTeam, TeamResult
from .message_bus import MessageBus, MessagePriority, MessageType, TeamMessage
from .roles import AgentRole, RoleConfig

__all__ = [
    # Roles
    "AgentRole",
    "RoleConfig",
    # Message Bus
    "MessageBus",
    "TeamMessage",
    "MessagePriority",
    "MessageType",
    # Agent Team
    "AgentTeam",
    "AgentMember",
    "TeamResult",
]
