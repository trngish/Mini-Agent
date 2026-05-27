"""Message bus for inter-agent communication in Agent Team.

This module provides a message-passing system for agents to communicate
within a team, enabling role-boundary isolation and structured dialogue.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MessageType(str, Enum):
    """Types of messages in team communication."""

    TASK = "task"  # New task assignment
    RESULT = "result"  # Task completion result
    FEEDBACK = "feedback"  # Feedback on work
    QUESTION = "question"  # Question to another agent
    ANSWER = "answer"  # Answer to a question
    APPROVAL = "approval"  # Approval or rejection
    ESCALATION = "escalation"  # Escalate to higher authority
    BROADCAST = "broadcast"  # Broadcast to all agents


@dataclass
class TeamMessage:
    """A message passed between agents in a team.

    Attributes:
        id: Unique message identifier
        type: Message type
        sender: Name of sending agent (or "coordinator")
        recipient: Name of receiving agent ("coordinator" or specific agent name)
                   Use "*" for broadcast messages
        content: Message content
        priority: Message priority
        metadata: Additional metadata (task_id, dependencies, etc.)
        timestamp: When the message was created
    """

    id: str
    type: MessageType
    sender: str
    recipient: str
    content: str
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class MessageBus:
    """Message bus for inter-agent communication.

    Provides:
    - Point-to-point messaging between agents
    - Broadcast messaging
    - Message queuing with priority handling
    - Subscription-based message delivery

    Example:
        bus = MessageBus()
        bus.send(TeamMessage(...))

        # Agent checks its messages
        messages = bus.receive("my_agent_name")
        bus.receive("my_agent_name", wait=True)
    """

    def __init__(self) -> None:
        # Queue for each recipient: recipient -> list of messages
        self._queues: dict[str, list[TeamMessage]] = defaultdict(list)
        # All messages for broadcast: list
        self._broadcasts: list[TeamMessage] = []
        # Message counter for ID generation
        self._message_counter = 0

    def send(self, message: TeamMessage) -> str:
        """Send a message through the bus.

        Args:
            message: The message to send

        Returns:
            The message ID
        """
        self._message_counter += 1
        if not message.id:
            message.id = f"msg_{self._message_counter}"

        if message.recipient == "*":
            # Broadcast - add to all individual queues
            self._broadcasts.append(message)
        else:
            # Point-to-point - add to recipient's queue
            self._queues[message.recipient].append(message)

        return message.id

    def receive(
        self,
        recipient: str,
        blocking: bool = False,  # noqa: ARG002
        timeout: float = 30.0,  # noqa: ARG002
        max_count: int = 10,
    ) -> list[TeamMessage]:
        """Receive messages for a recipient.

        Args:
            recipient: Name of the receiving agent
            blocking: If True, wait for messages (not implemented yet)
            timeout: Maximum time to wait for messages
            max_count: Maximum messages to return

        Returns:
            List of messages for this recipient
        """
        messages = []

        # Get direct messages
        direct = self._queues.get(recipient, [])
        messages.extend(direct)
        self._queues[recipient] = []

        # Get broadcasts
        broadcasts = list(self._broadcasts)
        messages.extend(broadcasts)
        self._broadcasts.clear()

        # Sort by priority (critical first)
        priority_order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3,
        }
        messages.sort(key=lambda m: priority_order.get(m.priority, 2))

        return messages[:max_count]

    def peek(self, recipient: str, max_count: int = 10) -> list[TeamMessage]:
        """Peek at messages without removing them.

        Args:
            recipient: Name of the receiving agent
            max_count: Maximum messages to return

        Returns:
            List of messages (still in queue)
        """
        messages = list(self._queues.get(recipient, []))
        messages.extend(self._broadcasts)
        return messages[:max_count]

    def has_messages(self, recipient: str) -> bool:
        """Check if recipient has pending messages.

        Args:
            recipient: Name of the receiving agent

        Returns:
            True if there are messages waiting
        """
        return bool(self._queues.get(recipient) or self._broadcasts)

    def clear(self, recipient: str = "*") -> int:
        """Clear messages for a recipient.

        Args:
            recipient: Name of recipient to clear, "*" for all

        Returns:
            Number of messages cleared
        """
        if recipient == "*":
            count = sum(len(q) for q in self._queues.values()) + len(self._broadcasts)
            self._queues.clear()
            self._broadcasts.clear()
            return count
        else:
            count = len(self._queues.get(recipient, []))
            self._queues[recipient] = []
            return count

    def get_all_recipients(self) -> list[str]:
        """Get list of all recipients with pending messages."""
        return [r for r in self._queues if self._queues[r]]

    def send_task(
        self,
        sender: str,
        executor: str,
        task: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Convenience method to send a task message.

        Args:
            sender: Name of sending agent
            executor: Name of executor agent
            task: Task description
            priority: Message priority
            metadata: Additional metadata

        Returns:
            Message ID
        """
        message = TeamMessage(
            id="",
            type=MessageType.TASK,
            sender=sender,
            recipient=executor,
            content=task,
            priority=priority,
            metadata=metadata or {},
        )
        return self.send(message)

    def send_result(
        self,
        sender: str,
        recipient: str,
        result: str,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Convenience method to send a result message.

        Args:
            sender: Name of sending agent
            recipient: Name of recipient agent
            result: Result content
            success: Whether the task was successful
            metadata: Additional metadata

        Returns:
            Message ID
        """
        metadata = metadata or {}
        metadata["success"] = success
        message = TeamMessage(
            id="",
            type=MessageType.RESULT,
            sender=sender,
            recipient=recipient,
            content=result,
            metadata=metadata,
        )
        return self.send(message)

    def broadcast(
        self,
        sender: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Convenience method to broadcast a message.

        Args:
            sender: Name of sending agent
            content: Message content
            priority: Message priority
            metadata: Additional metadata

        Returns:
            Message ID
        """
        message = TeamMessage(
            id="",
            type=MessageType.BROADCAST,
            sender=sender,
            recipient="*",
            content=content,
            priority=priority,
            metadata=metadata or {},
        )
        return self.send(message)
