"""Agent Team - Multi-agent collaboration with role boundaries and adversarial reasoning."""

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
