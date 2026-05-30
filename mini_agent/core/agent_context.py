"""AgentContext - 中心化状态管理与依赖注入。

该类作为代理状态的单一真实来源，
消除循环依赖并为所有组件提供清晰的接口。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..schema import AgentMode, Message

if TYPE_CHECKING:
    from .token_tracker import TokenTracker


@dataclass
class AgentContext:
    """代理状态和依赖的中心化上下文容器。

    该类提供：
    - 统一状态管理（消息、令牌计数、模式等）
    - 需要访问上下文的组件的依赖注入
    - 通过属性实现线程安全的状态更新（使用 Pydantic Message 模型）
    - 所有状态访问的清晰接口契约

    所有组件接收 AgentContext 的引用而不是 Agent 的引用，
    从而打破循环依赖。

    注意：Message 模型使用 Pydantic BaseModel 进行序列化和验证，
    提供 model_dump() 和 model_validate() 方法。
    """

    # 核心状态
    messages: list[Message] = field(default_factory=list)
    mode: AgentMode = AgentMode.YOLO
    max_steps: int = 50
    workspace_dir: Path = field(default_factory=lambda: Path())
    token_limit: int = 80000

    # LLM 状态
    api_call_count: int = 0
    api_total_tokens: int = 0
    is_m27: bool = False
    thinking_budget: int = 16384

    # 引用（初始化后设置）
    llm: Any = None

    # 状态属性
    auto_save: bool = True
    _last_auto_save_step: int = 0
    _consecutive_failures: int = 0

    # 依赖（注入的接口，不是具体实现）
    token_tracker: "TokenTracker | None" = None
    record_context_fn: Callable[[str, str], None] | None = None

    # 线程安全
    _lock: Lock = field(default_factory=Lock)

    def estimate_tokens(self) -> int:
        """使用注入的跟踪器估算令牌数，或使用备用方案。"""
        if self.token_tracker:
            return self.token_tracker.estimate_tokens(self.messages)
        total_chars = sum(len(str(m.content)) for m in self.messages)
        return int(total_chars / 2.5)

    def add_message(self, message: Message) -> None:
        """线程安全的消息添加。"""
        with self._lock:
            self.messages.append(message)

    def get_messages(self) -> list[Message]:
        """获取消息列表（返回副本以保证线程安全）。"""
        with self._lock:
            return self.messages.copy()

    def set_messages(self, messages: list[Message]) -> None:
        """替换整个消息列表（线程安全）。"""
        with self._lock:
            self.messages = messages

    def replace_last_message(self, message: Message) -> None:
        """替换最后一条消息（线程安全）。"""
        with self._lock:
            if self.messages:
                self.messages[-1] = message

    def record_context(self, content: str, category: str = "auto") -> None:
        """使用注入的函数记录上下文。"""
        if self.record_context_fn:
            self.record_context_fn(content, category)

    @property
    def consecutive_failures(self) -> int:
        """获取当前连续失败次数。"""
        return self._consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        """设置连续失败次数。"""
        self._consecutive_failures = value

    @property
    def last_auto_save_step(self) -> int:
        """获取上次自动保存的步骤。"""
        return self._last_auto_save_step

    @last_auto_save_step.setter
    def last_auto_save_step(self, value: int) -> None:
        """设置上次自动保存的步骤。"""
        self._last_auto_save_step = value