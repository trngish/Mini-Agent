"""StepRunner interface for agent decoupling."""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


class IStepRunnerDelegate(ABC):
    """Interface for StepRunner to interact with Agent without direct coupling.

    This interface defines the contract that StepRunner uses to interact
    with Agent components, allowing for easier testing and reduced coupling.

    Implementations should provide access to Agent's context, logging,
    and other necessary components through this interface.
    """

    @property
    @abstractmethod
    def context(self) -> Any:
        """Get the agent's context."""
        pass

    @property
    @abstractmethod
    def logger(self) -> Any:
        """Get the agent's logger."""
        pass

    @property
    @abstractmethod
    def thinking_manager(self) -> Any:
        """Get the thinking manager (can be None)."""
        pass

    @abstractmethod
    def check_health(self) -> list[str]:
        """Run health check and return issues."""
        pass

    @abstractmethod
    def save_session(self, step: int, prefix: str) -> None:
        """Save session."""
        pass