"""Test cases for individual Anthropic and OpenAI LLM clients.

These tests directly test the AnthropicClient and OpenAIClient implementations
without going through the wrapper layer.
"""

import asyncio
from pathlib import Path

import pytest
import yaml

from mini_agent.llm import AnthropicClient, OpenAIClient
from mini_agent.retry import RetryConfig
from mini_agent.schema import Message


def _load_config_or_skip():
    """Load config from config.yaml, skipping if unavailable."""
    config_path = Path("mini_agent/config/config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config.get("api_key") or config["api_key"] == "YOUR_MINIMAX_API_KEY_HERE":
        pytest.skip("API key not configured")
    return config


@pytest.mark.asyncio
@pytest.mark.integration
async def test_anthropic_simple_completion():
    """Test Anthropic client with simple completion."""
    config = _load_config_or_skip()

    # Create Anthropic client
    client = AnthropicClient(
        api_key=config["api_key"],
        api_base="https://api.minimaxi.com/anthropic",
        model=config.get("model", "MiniMax-M2.5"),
        retry_config=RetryConfig(enabled=True, max_retries=2),
    )

    # Simple messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello from Anthropic!' and nothing else."),
    ]

    response = await client.generate(messages=messages)

    assert response.content, "Response content is empty"
    assert "Hello" in response.content or "hello" in response.content, (
        f"Response doesn't contain 'Hello': {response.content}"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_openai_simple_completion():
    """Test OpenAI client with simple completion."""
    config = _load_config_or_skip()

    # Create OpenAI client
    client = OpenAIClient(
        api_key=config["api_key"],
        api_base="https://api.minimaxi.com/v1",
        model=config.get("model", "MiniMax-M2.5"),
        retry_config=RetryConfig(enabled=True, max_retries=2),
    )

    # Simple messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello from OpenAI!' and nothing else."),
    ]

    response = await client.generate(messages=messages)

    assert response.content, "Response content is empty"
    assert "Hello" in response.content or "hello" in response.content, (
        f"Response doesn't contain 'Hello': {response.content}"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_anthropic_tool_calling():
    """Test Anthropic client with tool calling."""
    config = _load_config_or_skip()

    # Create Anthropic client
    client = AnthropicClient(
        api_key=config["api_key"],
        api_base="https://api.minimaxi.com/anthropic",
        model=config.get("model", "MiniMax-M2.5"),
    )

    # Define tool using dict format
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather of a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, US",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    # Messages requesting tool use
    messages = [
        Message(role="user", content="What's the weather in San Francisco?"),
    ]

    response = await client.generate(messages=messages, tools=tools)

    assert response.content or response.tool_calls, (
        "Response should have content or tool calls"
    )
    if response.tool_calls:
        assert len(response.tool_calls) > 0, "Tool calls list should not be empty"
        assert response.tool_calls[0].function.name == "get_weather", (
            f"Expected 'get_weather' tool, got '{response.tool_calls[0].function.name}'"
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_openai_tool_calling():
    """Test OpenAI client with tool calling."""
    config = _load_config_or_skip()

    # Create OpenAI client
    client = OpenAIClient(
        api_key=config["api_key"],
        api_base="https://api.minimaxi.com/v1",
        model=config.get("model", "MiniMax-M2.5"),
    )

    # Define tool using dict format (will be converted internally for OpenAI)
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather of a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, US",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    # Messages requesting tool use
    messages = [
        Message(role="user", content="What's the weather in New York?"),
    ]

    response = await client.generate(messages=messages, tools=tools)

    assert response.content or response.tool_calls, (
        "Response should have content or tool calls"
    )
    if response.tool_calls:
        assert len(response.tool_calls) > 0, "Tool calls list should not be empty"
        assert response.tool_calls[0].function.name == "get_weather", (
            f"Expected 'get_weather' tool, got '{response.tool_calls[0].function.name}'"
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multi_turn_conversation():
    """Test multi-turn conversation with tool calling."""
    config = _load_config_or_skip()

    # Test with Anthropic client
    client = AnthropicClient(
        api_key=config["api_key"],
        api_base="https://api.minimaxi.com/anthropic",
        model=config.get("model", "MiniMax-M2.5"),
    )

    # Define tool using dict format
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
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        }
    ]

    # First turn - user asks
    messages = [Message(role="user", content="What's 5 + 3?")]
    response = await client.generate(messages=messages, tools=tools)

    if response.tool_calls:
        # Add assistant response
        messages.append(
            Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
        )

        # Add tool result
        messages.append(
            Message(
                role="tool",
                tool_call_id=response.tool_calls[0].id,
                content="8",
            )
        )

        # Second turn - get final answer
        final_response = await client.generate(messages=messages, tools=tools)
        assert final_response.content, "Final response should have content"
    else:
        # If LLM didn't use tools, it should still have responded with content
        assert response.content, "Response should have content when no tool calls"


async def main():
    """Run all LLM client tests."""
    print("=" * 80)
    print("Running LLM Client Tests")
    print("=" * 80)
    print("\nNote: These tests require a valid MiniMax API key in config.yaml")

    results = []

    # Test Anthropic client
    results.append(await test_anthropic_simple_completion())
    results.append(await test_anthropic_tool_calling())

    # Test OpenAI client
    results.append(await test_openai_simple_completion())
    results.append(await test_openai_tool_calling())

    # Test multi-turn conversation
    results.append(await test_multi_turn_conversation())

    print("\n" + "=" * 80)
    if all(results):
        print("All LLM client tests passed!")
    else:
        print("Some LLM client tests failed. Check the output above.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
