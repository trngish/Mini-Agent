"""Agent Team - Multi-agent collaboration with role boundaries and adversarial reasoning."""

from .roles import AgentRole, RoleConfig
from .message_bus import MessageBus, TeamMessage, MessagePriority, MessageType
from .agent_team import AgentTeam, AgentMember, TeamResult

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