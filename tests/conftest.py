"""Shared test fixtures for Mini-Agent test suite."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.core.agent_context import AgentContext
from mini_agent.llm import LLMClient
from mini_agent.schema import AgentMode, Message, LLMProvider


@pytest.fixture(autouse=True)
def _reset_task_state():
    """Reset global task state between tests."""
    from mini_agent.utils.task_state import get_task_manager

    get_task_manager().reset()
    yield
    get_task_manager().reset()


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for unit testing.

    Returns a MagicMock that mimics LLMClient interface.
    Configure return values in individual tests as needed.
    """
    client = MagicMock(spec=LLMClient)
    client.api_key = "test-key"
    client.api_base = "https://api.test.com"
    client.model = "test-model"
    client.provider = LLMProvider.ANTHROPIC
    client.retry_config = MagicMock()
    # Make generate an async mock by default
    client.generate = AsyncMock()
    return client


@pytest.fixture
def sample_messages():
    """Create a sample message list for testing."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
    ]


@pytest.fixture
def mock_agent_context():
    """Create a mock AgentContext for unit testing.

    This allows testing core modules independently without
    requiring a full Agent instance.
    """
    return AgentContext(
        messages=[Message(role="system", content="You are a test agent.")],
        mode=AgentMode.YOLO,
        max_steps=10,
        workspace_dir=Path(tempfile.gettempdir()),
        token_limit=100000,
        api_call_count=0,
        api_total_tokens=0,
        is_m27=False,
        thinking_budget=16384,
    )


@pytest.fixture
def mock_token_tracker():
    """Create a mock TokenTracker for testing."""
    from mini_agent.core.token_tracker import TokenTracker
    return TokenTracker()


@pytest.fixture
def agent_context_with_tracker(mock_token_tracker):
    """Create AgentContext with TokenTracker for integration tests."""
    return AgentContext(
        messages=[Message(role="system", content="You are a test agent.")],
        mode=AgentMode.YOLO,
        max_steps=10,
        workspace_dir=Path(tempfile.gettempdir()),
        token_limit=100000,
        api_call_count=0,
        api_total_tokens=0,
        is_m27=False,
        thinking_budget=16384,
        token_tracker=mock_token_tracker,
    )


@pytest.fixture
def health_checker_with_context(mock_agent_context):
    """Create HealthChecker with mock AgentContext."""
    from mini_agent.core.health_check import HealthChecker
    return HealthChecker(context=mock_agent_context)


@pytest.fixture
def error_recovery_with_context(mock_agent_context):
    """Create ErrorRecoveryManager with mock AgentContext."""
    from mini_agent.core.error_recovery import ErrorRecoveryManager
    return ErrorRecoveryManager(context=mock_agent_context)


@pytest.fixture
def metrics_with_context(mock_agent_context):
    """Create PerformanceMetrics with mock AgentContext."""
    from mini_agent.core.metrics import PerformanceMetrics
    return PerformanceMetrics(context=mock_agent_context)


@pytest.fixture
def thinking_budget_manager_with_context(mock_agent_context):
    """Create ThinkingBudgetManager with mock AgentContext."""
    from mini_agent.core.thinking_budget import ThinkingBudgetManager
    manager = ThinkingBudgetManager(context=mock_agent_context)
    manager.configure(16384, is_m27=False)
    return manager


@pytest.fixture
def mock_message_manager():
    """Create a mock MessageManager for testing."""
    from mini_agent.core.message_manager import MessageManager
    mm = MessageManager(token_limit=100000)
    mm.initialize("You are a test agent.")
    return mm


@pytest.fixture
def approval_manager():
    """Create ApprovalManager for testing."""
    from mini_agent.core.approval import ApprovalManager
    return ApprovalManager(mode=AgentMode.YOLO)


@pytest.fixture
def rate_limiter():
    """Create RateLimiter for testing."""
    from mini_agent.core.rate_limiter import RateLimiter
    return RateLimiter()


@pytest.fixture(autouse=True)
def reset_background_shells():
    """Reset BackgroundShellManager state before each test."""
    from mini_agent.tools.bash_background import BackgroundShellManager
    BackgroundShellManager.clear_all()
    yield
    BackgroundShellManager.clear_all()


@pytest.fixture(autouse=True)
def reset_context_cache():
    """Reset global context cache before each test."""
    from mini_agent.utils.context_cache import reset_global_cache
    reset_global_cache()
    yield
    reset_global_cache()