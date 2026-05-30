"""Message Bus - 智能体团队内部通信的消息传递系统。

本模块提供智能体在团队内通信的消息传递系统，支持角色边界隔离和结构化对话。
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessagePriority(str, Enum):
    """消息优先级级别。"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MessageType(str, Enum):
    """团队通信中的消息类型。"""

    TASK = "task"  # 新任务分配
    RESULT = "result"  # 任务完成结果
    FEEDBACK = "feedback"  # 工作反馈
    QUESTION = "question"  # 向另一个智能体提问
    ANSWER = "answer"  # 问题回答
    APPROVAL = "approval"  # 批准或拒绝
    ESCALATION = "escalation"  # 升级到更高层级
    BROADCAST = "broadcast"  # 向所有智能体广播


@dataclass
class TeamMessage:
    """团队中智能体之间传递的消息。

    Attributes:
        id: 唯一消息标识符
        type: 消息类型
        sender: 发送智能体的名称（或 "coordinator"）
        recipient: 接收智能体的名称（"coordinator" 或特定智能体名称）
                   使用 "*" 进行广播消息
        content: 消息内容
        priority: 消息优先级
        metadata: 附加元数据（task_id、依赖等）
        timestamp: 消息创建时间
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
        """转换为字典以便序列化。"""
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
    """智能体间通信的消息总线。

    提供：
    - 智能体之间的点对点消息传递
    - 广播消息
    - 优先级处理的消息队列
    - 基于订阅的消息传递

    Example:
        bus = MessageBus()
        bus.send(TeamMessage(...))

        # 智能体检查其消息
        messages = bus.receive("my_agent_name")
        bus.receive("my_agent_name", wait=True)
    """

    def __init__(self) -> None:
        # 每个接收者的队列: recipient -> 消息列表
        self._queues: dict[str, list[TeamMessage]] = defaultdict(list)
        # 所有广播消息: 列表
        self._broadcasts: list[TeamMessage] = []
        # 消息计数器，用于生成ID
        self._message_counter = 0

    def send(self, message: TeamMessage) -> str:
        """通过总线发送消息。

        Args:
            message: 要发送的消息

        Returns:
            消息ID
        """
        self._message_counter += 1
        if not message.id:
            message.id = f"msg_{self._message_counter}"

        if message.recipient == "*":
            # 广播 - 添加到所有单独队列
            self._broadcasts.append(message)
        else:
            # 点对点 - 添加到接收者的队列
            self._queues[message.recipient].append(message)

        return message.id

    def receive(
        self,
        recipient: str,
        blocking: bool = False,  # noqa: ARG002
        timeout: float = 30.0,  # noqa: ARG002
        max_count: int = 10,
    ) -> list[TeamMessage]:
        """接收发送给某接收者的消息。

        Args:
            recipient: 接收智能体的名称
            blocking: 如果为True，则等待消息（尚未实现）
            timeout: 等待消息的最长时间
            max_count: 返回的最大消息数

        Returns:
            该接收者的消息列表
        """
        messages = []

        # 获取直接消息
        direct = self._queues.get(recipient, [])
        messages.extend(direct)
        self._queues[recipient] = []

        # 获取广播
        broadcasts = list(self._broadcasts)
        messages.extend(broadcasts)
        self._broadcasts.clear()

        # 按优先级排序（关键消息优先）
        priority_order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3,
        }
        messages.sort(key=lambda m: priority_order.get(m.priority, 2))

        return messages[:max_count]

    def peek(self, recipient: str, max_count: int = 10) -> list[TeamMessage]:
        """查看消息但不移除它们。

        Args:
            recipient: 接收智能体的名称
            max_count: 返回的最大消息数

        Returns:
            消息列表（仍在队列中）
        """
        messages = list(self._queues.get(recipient, []))
        messages.extend(self._broadcasts)
        return messages[:max_count]

    def has_messages(self, recipient: str) -> bool:
        """检查接收者是否有待处理消息。

        Args:
            recipient: 接收智能体的名称

        Returns:
            如果有消息等待则返回True
        """
        return bool(self._queues.get(recipient) or self._broadcasts)

    def clear(self, recipient: str = "*") -> int:
        """清除某接收者的消息。

        Args:
            recipient: 要清除的接收者名称，"*" 表示全部

        Returns:
            清除的消息数
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
        """获取所有有待处理消息的接收者列表。"""
        return [r for r in self._queues if self._queues[r]]

    def send_task(
        self,
        sender: str,
        executor: str,
        task: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """发送任务消息的便捷方法。

        Args:
            sender: 发送智能体的名称
            executor: 执行者智能体的名称
            task: 任务描述
            priority: 消息优先级
            metadata: 附加元数据

        Returns:
            消息ID
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
        """发送结果消息的便捷方法。

        Args:
            sender: 发送智能体的名称
            recipient: 接收智能体的名称
            result: 结果内容
            success: 任务是否成功
            metadata: 附加元数据

        Returns:
            消息ID
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
        """广播消息的便捷方法。

        Args:
            sender: 发送智能体的名称
            content: 消息内容
            priority: 消息优先级
            metadata: 附加元数据

        Returns:
            消息ID
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
