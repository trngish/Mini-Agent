"""Test cases for LLM wrapper client."""

import asyncio
from pathlib import Path

import pytest
import yaml

from mini_agent.llm import LLMClient
from mini_agent.schema import LLMProvider, Message


def _skip_if_no_config():
    """Skip test if config.yaml or API key is not available."""
    config_path = Path("mini_agent/config/config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    api_key = config.get("api_key", "")
    if not api_key or "YOUR" in api_key.upper():
        pytest.skip("API key not configured")
    return config


@pytest.mark.asyncio
@pytest.mark.integration
async def test_wrapper_anthropic_provider():
    """Test LLM wrapper with Anthropic provider."""
    config = _skip_if_no_config()

    # Create client with Anthropic provider
    client = LLMClient(
        api_key=config["api_key"],
        provider=LLMProvider.ANTHROPIC,
        api_base=config.get("api_base"),
        model=config.get("model"),
    )

    assert client.provider == LLMProvider.ANTHROPIC

    # Simple messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello, Mini Agent!' and nothing else."),
    ]

    response = await client.generate(messages=messages)

    assert response.content, "Response content is empty"
    assert "Hello" in response.content or "hello" in response.content, (
        f"Response doesn't contain 'Hello': {response.content}"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_wrapper_openai_provider():
    """Test LLM wrapper with OpenAI provider."""
    config = _skip_if_no_config()

    # Create client with OpenAI provider
    client = LLMClient(
        api_key=config["api_key"],
        provider=LLMProvider.OPENAI,
        model=config.get("model"),
    )

    assert client.provider == LLMProvider.OPENAI

    # Simple messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello, Mini Agent!' and nothing else."),
    ]

    response = await client.generate(messages=messages)

    assert response.content, "Response content is empty"
    assert "Hello" in response.content or "hello" in response.content, (
        f"Response doesn't contain 'Hello': {response.content}"
    )


@pytest.mark.asyncio
async def test_wrapper_default_provider():
    """Test LLM wrapper with default provider (Anthropic)."""
    config = _skip_if_no_config()

    # Create client without specifying provider (should default to Anthropic)
    client = LLMClient(
        api_key=config["api_key"],
        model=config.get("model"),
    )

    assert client.provider == LLMProvider.ANTHROPIC


@pytest.mark.asyncio
@pytest.mark.integration
async def test_wrapper_tool_calling():
    """Test LLM wrapper with tool calling."""
    config = _skip_if_no_config()

    # Create client with Anthropic provider
    client = LLMClient(
        api_key=config["api_key"],
        provider=LLMProvider.ANTHROPIC,
        model=config.get("model"),
    )

    # Messages requesting tool use
    messages = [
        Message(role="system", content="You are a helpful assistant with access to tools."),
        Message(role="user", content="Calculate 123 + 456 using the calculator tool."),
    ]

    # Define a simple calculator tool using dict format
    tools = [
        {
            "name": "calculator",
            "description": "Perform arithmetic operations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform",
                    },
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["operation", "a", "b"],
            },
        }
    ]

    response = await client.generate(messages=messages, tools=tools)

    assert response.content or response.tool_calls, "Response should have content or tool calls"
    if response.tool_calls:
        assert len(response.tool_calls) > 0, "Tool calls list should not be empty"
        assert response.tool_calls[0].function.name == "calculator", (
            f"Expected 'calculator' tool, got '{response.tool_calls[0].function.name}'"
        )


async def main():
    """Run all LLM wrapper tests."""
    print("=" * 80)
    print("Running LLM Wrapper Tests")
    print("=" * 80)
    print("\nNote: These tests require a valid MiniMax API key in config.yaml")

    results = []

    # Test default provider
    results.append(await test_wrapper_default_provider())

    # Test Anthropic provider
    results.append(await test_wrapper_anthropic_provider())

    # Test OpenAI provider
    results.append(await test_wrapper_openai_provider())

    # Test tool calling
    results.append(await test_wrapper_tool_calling())

    print("\n" + "=" * 80)
    if all(results):
        print("All LLM wrapper tests passed!")
    else:
        print("Some LLM wrapper tests failed. Check the output above.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
