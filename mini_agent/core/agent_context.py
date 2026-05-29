"""AgentContext - Central state management with dependency injection.

This class serves as the single source of truth for agent state,
eliminating circular dependencies and providing clean interfaces
for all components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable

from ..schema import AgentMode, Message

if TYPE_CHECKING:
    from .token_tracker import TokenTracker


@dataclass
class AgentContext:
    """Central context container for agent state and dependencies.

    This class provides:
    - Unified state management (messages, token counts, mode, etc.)
    - Dependency injection for components that need context access
    - Thread-safe state updates via properties (using pydantic Message model)
    - Clear interface contracts for all state access

    All components receive a reference to AgentContext instead of Agent,
    breaking circular dependencies.

    Note: The Message model uses Pydantic BaseModel for serialization
    and validation, providing model_dump() and model_validate() methods.
    """

    # Core state
    messages: list[Message] = field(default_factory=list)
    mode: AgentMode = AgentMode.YOLO
    max_steps: int = 50
    workspace_dir: Path = field(default_factory=lambda: Path("."))
    token_limit: int = 80000

    # LLM state
    api_call_count: int = 0
    api_total_tokens: int = 0
    is_m27: bool = False
    thinking_budget: int = 16384

    # References (set after initialization)
    llm: Any = None

    # State properties
    auto_save: bool = True
    _last_auto_save_step: int = 0
    _consecutive_failures: int = 0

    # Dependencies (injected interfaces, not concrete implementations)
    token_tracker: "TokenTracker | None" = None
    record_context_fn: Callable[[str, str], None] | None = None

    # Thread safety
    _lock: Lock = field(default_factory=Lock)

    def estimate_tokens(self) -> int:
        """Estimate token count using injected tracker or fallback."""
        if self.token_tracker:
            return self.token_tracker.estimate_tokens(self.messages)
        total_chars = sum(len(str(m.content)) for m in self.messages)
        return int(total_chars / 2.5)

    def add_message(self, message: Message) -> None:
        """Thread-safe message addition."""
        with self._lock:
            self.messages.append(message)

    def get_messages(self) -> list[Message]:
        """Get message list (returns copy for thread safety)."""
        with self._lock:
            return self.messages.copy()

    def set_messages(self, messages: list[Message]) -> None:
        """Replace entire message list (thread-safe)."""
        with self._lock:
            self.messages = messages

    def replace_last_message(self, message: Message) -> None:
        """Replace the last message (thread-safe)."""
        with self._lock:
            if self.messages:
                self.messages[-1] = message

    def record_context(self, content: str, category: str = "auto") -> None:
        """Record context using injected function."""
        if self.record_context_fn:
            self.record_context_fn(content, category)

    @property
    def consecutive_failures(self) -> int:
        """Get current consecutive failure count."""
        return self._consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        """Set consecutive failure count."""
        self._consecutive_failures = value

    @property
    def last_auto_save_step(self) -> int:
        """Get last auto-save step."""
        return self._last_auto_save_step

    @last_auto_save_step.setter
    def last_auto_save_step(self, value: int) -> None:
        """Set last auto-save step."""
        self._last_auto_save_step = value