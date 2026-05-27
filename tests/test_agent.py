"""Test cases for Agent.

Note: These tests require a valid API key and call real LLM endpoints.
They are marked with @pytest.mark.integration so they can be excluded
from fast test runs via: pytest -m "not integration"
"""

import tempfile
from pathlib import Path

import pytest

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.config import Config
from mini_agent.tools import BashTool, EditTool, ReadTool, WriteTool


def _load_config() -> Config:
    """Load config or skip test if unavailable."""
    config_path = Path("mini_agent/config/config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")
    config = Config.from_yaml(config_path)
    if not config.llm.api_key or config.llm.api_key == "YOUR_API_KEY_HERE":
        pytest.skip("API key not configured")
    return config


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_simple_task():
    """Test agent with a simple file creation task."""
    config = _load_config()

    with tempfile.TemporaryDirectory() as workspace_dir:
        system_prompt_path = Path("mini_agent/config/system_prompt.md")
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text(encoding="utf-8")
        else:
            system_prompt = "You are a helpful AI assistant that can use tools."

        llm_client = LLMClient(
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
            model=config.llm.model,
        )

        tools = [
            ReadTool(workspace_dir=workspace_dir),
            WriteTool(workspace_dir=workspace_dir),
            EditTool(workspace_dir=workspace_dir),
            BashTool(),
        ]

        agent = Agent(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            max_steps=10,
            workspace_dir=workspace_dir,
        )

        task = "Create a file named 'test.txt' with the content 'Hello from Agent!'"
        agent.add_user_message(task)

        result = await agent.run()
        assert result is not None, "Agent run should return a result"

        # Verify the file was actually created
        test_file = Path(workspace_dir) / "test.txt"
        assert test_file.exists(), f"Agent should have created {test_file}"
        content = test_file.read_text()
        assert "Hello from Agent!" in content, f"File content should contain expected text, got: {content}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_bash_task():
    """Test agent with a bash command task."""
    config = _load_config()

    with tempfile.TemporaryDirectory() as workspace_dir:
        system_prompt_path = Path("mini_agent/config/system_prompt.md")
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text(encoding="utf-8")
        else:
            system_prompt = "You are a helpful AI assistant that can use tools."

        llm_client = LLMClient(
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
            model=config.llm.model,
        )

        tools = [
            ReadTool(workspace_dir=workspace_dir),
            WriteTool(workspace_dir=workspace_dir),
            BashTool(),
        ]

        agent = Agent(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            max_steps=10,
            workspace_dir=workspace_dir,
        )

        task = "Use bash to list all files in the current directory and tell me what you find."
        agent.add_user_message(task)

        result = await agent.run()
        assert result is not None, "Agent run should return a result"
