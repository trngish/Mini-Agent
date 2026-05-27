"""Shared test fixtures for Mini-Agent test suite."""

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.llm import LLMClient
from mini_agent.schema import LLMProvider


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
    from mini_agent.schema import Message

    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
    ]
