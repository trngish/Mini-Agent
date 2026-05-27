from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.llm.anthropic_client import AnthropicClient, StreamedResponse
from mini_agent.llm.llm_wrapper import LLMClient
from mini_agent.llm.openai_client import OpenAIClient
from mini_agent.retry import RetryConfig
from mini_agent.schema import (
    FunctionCall,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
)


def _make_usage_mock(**overrides: int) -> MagicMock:
    defaults = dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_anthropic_client(**overrides: Any) -> AnthropicClient:
    defaults = dict(
        api_key="test-key",
        api_base="https://api.test.com/anthropic",
        model="MiniMax-M2.5",
        retry_config=RetryConfig(enabled=False),
    )
    defaults.update(overrides)
    with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
        return AnthropicClient(**defaults)


def _make_openai_client(**overrides: Any) -> OpenAIClient:
    defaults = dict(
        api_key="test-key",
        api_base="https://api.test.com/v1",
        model="MiniMax-M2.5",
        retry_config=RetryConfig(enabled=False),
    )
    defaults.update(overrides)
    with patch("mini_agent.llm.openai_client.AsyncOpenAI"):
        return OpenAIClient(**defaults)


class TestAnthropicClientInit:
    def test_default_values(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic") as mock_cls:
            client = AnthropicClient(api_key="key123")
            mock_cls.assert_called_once_with(
                base_url="https://api.minimaxi.com/anthropic",
                api_key="key123",
            )
            assert client.api_key == "key123"
            assert client.model == "MiniMax-M2.5"
            assert client._enable_extended_thinking is True
            assert client._thinking_budget_tokens == 8192

    def test_custom_values(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic") as mock_cls:
            client = AnthropicClient(
                api_key="k",
                api_base="https://custom.api.com",
                model="MiniMax-M2.7",
                retry_config=RetryConfig(max_retries=5),
            )
            mock_cls.assert_called_once_with(
                base_url="https://custom.api.com",
                api_key="k",
            )
            assert client.api_base == "https://custom.api.com"
            assert client.model == "MiniMax-M2.7"
            assert client.retry_config.max_retries == 5

    def test_retry_config_default(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            client = AnthropicClient(api_key="k")
            assert client.retry_config.enabled is True
            assert client.retry_config.max_retries == 3


class TestOpenAIClientInit:
    def test_default_values(self):
        with patch("mini_agent.llm.openai_client.AsyncOpenAI") as mock_cls:
            client = OpenAIClient(api_key="key123")
            mock_cls.assert_called_once_with(
                api_key="key123",
                base_url="https://api.minimaxi.com/v1",
            )
            assert client.api_key == "key123"
            assert client.model == "MiniMax-M2.5"
            assert client._enable_extended_thinking is True
            assert client._thinking_budget_tokens == 8192

    def test_custom_values(self):
        with patch("mini_agent.llm.openai_client.AsyncOpenAI") as mock_cls:
            client = OpenAIClient(
                api_key="k",
                api_base="https://custom.api.com/v1",
                model="MiniMax-M2.7",
                retry_config=RetryConfig(max_retries=5),
            )
            mock_cls.assert_called_once_with(
                api_key="k",
                base_url="https://custom.api.com/v1",
            )
            assert client.api_base == "https://custom.api.com/v1"
            assert client.model == "MiniMax-M2.7"

    def test_retry_config_default(self):
        with patch("mini_agent.llm.openai_client.AsyncOpenAI"):
            client = OpenAIClient(api_key="k")
            assert client.retry_config.enabled is True


class TestAnthropicConvertMessages:
    def test_system_message_extracted(self):
        client = _make_anthropic_client()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        system_msg, api_msgs = client._convert_messages(messages)
        assert system_msg == "You are helpful."
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[0]["content"] == "Hi"

    def test_assistant_with_thinking_and_tool_calls(self):
        client = _make_anthropic_client()
        messages = [
            Message(
                role="assistant",
                content="Let me check.",
                thinking="I should use the tool.",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        type="function",
                        function=FunctionCall(name="search", arguments={"q": "test"}),
                    )
                ],
            ),
        ]
        system_msg, api_msgs = client._convert_messages(messages)
        assert system_msg is None
        assert len(api_msgs) == 1
        content = api_msgs[0]["content"]
        types = [b["type"] for b in content]
        assert "thinking" in types
        assert "text" in types
        assert "tool_use" in types

    def test_tool_result_message(self):
        client = _make_anthropic_client()
        messages = [
            Message(role="tool", content="result data", tool_call_id="tc_1"),
        ]
        _, api_msgs = client._convert_messages(messages)
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"
        content = api_msgs[0]["content"]
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "tc_1"
        assert content[0]["content"] == "result data"

    def test_no_system_message(self):
        client = _make_anthropic_client()
        messages = [Message(role="user", content="Hello")]
        system_msg, _ = client._convert_messages(messages)
        assert system_msg is None

    def test_long_user_message_gets_cache_control(self):
        client = _make_anthropic_client()
        long_content = "x" * 2000
        messages = [
            Message(role="user", content=long_content),
        ]
        _, api_msgs = client._convert_messages(messages)
        assert isinstance(api_msgs[0]["content"], list)
        assert api_msgs[0]["content"][0].get("cache_control") is not None

    def test_short_user_message_no_cache_control(self):
        client = _make_anthropic_client()
        messages = [Message(role="user", content="short")]
        _, api_msgs = client._convert_messages(messages)
        assert isinstance(api_msgs[0]["content"], str)


class TestOpenAIConvertMessages:
    def test_system_message_included_in_array(self):
        client = _make_openai_client()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        system_msg, api_msgs = client._convert_messages(messages)
        assert system_msg is None
        assert api_msgs[0]["role"] == "system"
        assert api_msgs[0]["content"] == "You are helpful."
        assert api_msgs[1]["role"] == "user"

    def test_assistant_with_thinking(self):
        client = _make_openai_client()
        messages = [
            Message(
                role="assistant",
                content="Here is the answer.",
                thinking="Let me think about this.",
            ),
        ]
        _, api_msgs = client._convert_messages(messages)
        assert api_msgs[0]["role"] == "assistant"
        assert api_msgs[0]["content"] == "Here is the answer."
        assert api_msgs[0]["reasoning_details"] == [{"text": "Let me think about this."}]

    def test_assistant_with_tool_calls(self):
        client = _make_openai_client()
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="get_weather", arguments={"city": "SF"}),
                    )
                ],
            ),
        ]
        _, api_msgs = client._convert_messages(messages)
        tc = api_msgs[0]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "get_weather"
        parsed_args = json.loads(tc["function"]["arguments"])
        assert parsed_args == {"city": "SF"}

    def test_tool_result_message(self):
        client = _make_openai_client()
        messages = [
            Message(role="tool", content="sunny", tool_call_id="call_1"),
        ]
        _, api_msgs = client._convert_messages(messages)
        assert api_msgs[0]["role"] == "tool"
        assert api_msgs[0]["tool_call_id"] == "call_1"
        assert api_msgs[0]["content"] == "sunny"


class TestAnthropicConvertTools:
    def test_dict_tools_passed_through(self):
        client = _make_anthropic_client()
        tool = {"name": "search", "description": "Search", "input_schema": {"type": "object", "properties": {}}}
        result = client._convert_tools([tool])
        assert result == [tool]

    def test_object_with_to_schema(self):
        client = _make_anthropic_client()
        mock_tool = MagicMock()
        mock_tool.to_schema.return_value = {"name": "calc", "description": "Calculate", "input_schema": {}}
        result = client._convert_tools([mock_tool])
        assert result == [{"name": "calc", "description": "Calculate", "input_schema": {}}]

    def test_unsupported_tool_type_raises(self):
        client = _make_anthropic_client()
        with pytest.raises(TypeError, match="Unsupported tool type"):
            client._convert_tools([42])


class TestOpenAIConvertTools:
    def test_openai_format_dict_passed_through(self):
        client = _make_openai_client()
        tool = {
            "type": "function",
            "function": {"name": "search", "description": "Search", "parameters": {"type": "object"}},
        }
        result = client._convert_tools([tool])
        assert result == [tool]

    def test_anthropic_format_dict_converted(self):
        client = _make_openai_client()
        tool = {
            "name": "search",
            "description": "Search things",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        result = client._convert_tools([tool])
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["parameters"] == tool["input_schema"]

    def test_object_with_to_openai_schema(self):
        client = _make_openai_client()
        mock_tool = MagicMock()
        mock_tool.to_openai_schema.return_value = {
            "type": "function",
            "function": {"name": "calc", "description": "Calculate", "parameters": {}},
        }
        result = client._convert_tools([mock_tool])
        assert len(result) == 1

    def test_unsupported_tool_type_raises(self):
        client = _make_openai_client()
        with pytest.raises(TypeError, match="Unsupported tool type"):
            client._convert_tools([42])


class TestAnthropicParseResponse:
    def test_parse_streamed_response_with_text(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(
            text="Hello world",
            thinking="I am thinking",
            input_tokens=10,
            output_tokens=20,
        )
        result = client._parse_response(streamed)
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world"
        assert result.thinking == "I am thinking"
        assert result.finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    def test_parse_streamed_response_with_tool_calls(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(
            text="",
            tool_calls=[
                {"id": "1", "name": "search", "input": {"q": "test"}},
                {"id": "2", "name": "calc", "input": {"expr": "1+1"}},
            ],
            input_tokens=5,
            output_tokens=15,
        )
        result = client._parse_response(streamed)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "search"
        assert result.tool_calls[0].function.arguments == {"q": "test"}
        assert result.tool_calls[1].function.name == "calc"

    def test_parse_streamed_response_no_usage(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(text="Hi")
        result = client._parse_response(streamed)
        assert result.usage is None

    def test_parse_streamed_response_with_cache_tokens(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(
            text="Hi",
            input_tokens=5,
            output_tokens=10,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=2,
        )
        result = client._parse_response(streamed)
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 20

    def test_parse_streamed_response_empty_thinking(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(text="Hello", thinking="")
        result = client._parse_response(streamed)
        assert result.thinking is None

    def test_parse_non_streamed_message(self):
        client = _make_anthropic_client()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello"
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Deep thought"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tc_1"
        tool_block.name = "search"
        tool_block.input = {"q": "test"}
        mock_response.content = [text_block, thinking_block, tool_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = _make_usage_mock(input_tokens=100, output_tokens=50)
        result = client._parse_response(mock_response)
        assert result.content == "Hello"
        assert result.thinking == "Deep thought"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc_1"
        assert result.finish_reason == "end_turn"
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50


class TestOpenAIParseResponse:
    def _make_openai_response(
        self,
        content: str = "Hello",
        thinking: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        finish_reason: str = "stop",
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
        total_tokens: int = 30,
    ) -> MagicMock:
        message = MagicMock()
        message.content = content
        if thinking:
            detail = MagicMock()
            detail.text = thinking
            message.reasoning_details = [detail]
        else:
            message.reasoning_details = None
        if tool_calls:
            tc_list = []
            for tc in tool_calls:
                mock_tc = MagicMock()
                mock_tc.id = tc["id"]
                mock_tc.function.name = tc["name"]
                mock_tc.function.arguments = json.dumps(tc["arguments"])
                tc_list.append(mock_tc)
            message.tool_calls = tc_list
        else:
            message.tool_calls = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = finish_reason
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock()
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        response.usage.total_tokens = total_tokens
        return response

    def test_parse_simple_response(self):
        client = _make_openai_client()
        response = self._make_openai_response(content="Hello world")
        result = client._parse_response(response)
        assert result.content == "Hello world"
        assert result.thinking is None
        assert result.tool_calls is None
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20

    def test_parse_response_with_thinking(self):
        client = _make_openai_client()
        response = self._make_openai_response(content="Answer", thinking="My reasoning")
        result = client._parse_response(response)
        assert result.thinking == "My reasoning"

    def test_parse_response_with_tool_calls(self):
        client = _make_openai_client()
        response = self._make_openai_response(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "search", "arguments": {"q": "test"}},
            ],
        )
        result = client._parse_response(response)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"
        assert result.tool_calls[0].function.arguments == {"q": "test"}

    def test_parse_response_no_usage(self):
        client = _make_openai_client()
        message = MagicMock()
        message.content = "Hi"
        message.reasoning_details = None
        message.tool_calls = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        response = MagicMock()
        response.choices = [choice]
        response.usage = None
        result = client._parse_response(response)
        assert result.usage is None

    def test_parse_response_finish_reason_fallback(self):
        client = _make_openai_client()
        message = MagicMock()
        message.content = "Hi"
        message.reasoning_details = None
        message.tool_calls = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = None
        response = MagicMock()
        response.choices = [choice]
        response.usage = None
        result = client._parse_response(response)
        assert result.finish_reason == "stop"


class TestAnthropicThinkingConfig:
    def test_thinking_config_enabled_m27(self):
        client = _make_anthropic_client(model="MiniMax-M2.7")
        config = client._get_thinking_config()
        assert config is not None
        assert config["type"] == "enabled"
        assert config["budget_tokens"] > 0

    def test_thinking_config_disabled_flag(self):
        client = _make_anthropic_client(model="MiniMax-M2.7")
        client._enable_extended_thinking = False
        config = client._get_thinking_config()
        assert config is None

    def test_thinking_config_non_m27_model(self):
        client = _make_anthropic_client(model="some-other-model")
        config = client._get_thinking_config()
        assert config is None

    def test_thinking_config_zero_budget(self):
        client = _make_anthropic_client(model="MiniMax-M2.7")
        client._thinking_budget_tokens = 0
        config = client._get_thinking_config()
        assert config is None

    def test_configure_thinking_budget(self):
        client = _make_anthropic_client()
        client.configure_thinking_budget(16000)
        assert client._thinking_budget_tokens == 16000

    def test_configure_thinking_budget_clamped_max(self):
        client = _make_anthropic_client()
        client.configure_thinking_budget(99999)
        assert client._thinking_budget_tokens == 32768

    def test_configure_thinking_budget_clamped_min(self):
        client = _make_anthropic_client()
        client.configure_thinking_budget(-100)
        assert client._thinking_budget_tokens == 0

    def test_configure_m27(self):
        client = _make_anthropic_client()
        client.configure_m27({"enable_extended_thinking": False, "thinking_budget_tokens": 4096})
        assert client._enable_extended_thinking is False
        assert client._thinking_budget_tokens == 4096

    def test_configure_m27_defaults(self):
        client = _make_anthropic_client()
        client.configure_m27({})
        assert client._enable_extended_thinking is True
        assert client._thinking_budget_tokens == 16384

    def test_configure_m27_budget_capped(self):
        client = _make_anthropic_client()
        client.configure_m27({"thinking_budget_tokens": 99999})
        assert client._thinking_budget_tokens == 32768


class TestOpenAIThinkingConfig:
    def test_configure_thinking_budget(self):
        client = _make_openai_client()
        client.configure_thinking_budget(12000)
        assert client._thinking_budget_tokens == 12000

    def test_configure_thinking_budget_clamped_max(self):
        client = _make_openai_client()
        client.configure_thinking_budget(99999)
        assert client._thinking_budget_tokens == 32768

    def test_configure_thinking_budget_clamped_min(self):
        client = _make_openai_client()
        client.configure_thinking_budget(-5)
        assert client._thinking_budget_tokens == 0

    def test_configure_m27(self):
        client = _make_openai_client()
        client.configure_m27({"enable_extended_thinking": True, "thinking_budget_tokens": 8192})
        assert client._enable_extended_thinking is True
        assert client._thinking_budget_tokens == 8192

    def test_configure_m27_defaults(self):
        client = _make_openai_client()
        client.configure_m27({})
        assert client._enable_extended_thinking is True
        assert client._thinking_budget_tokens == 16384


class TestAnthropicGenerate:
    @pytest.mark.asyncio
    async def test_generate_simple_text(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(text="Hello!", input_tokens=10, output_tokens=5)
        client._make_api_request = AsyncMock(return_value=streamed)
        messages = [Message(role="user", content="Hi")]
        result = await client.generate(messages=messages)
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_tools(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(
            tool_calls=[{"id": "1", "name": "search", "input": {"q": "test"}}],
            input_tokens=15,
            output_tokens=10,
        )
        client._make_api_request = AsyncMock(return_value=streamed)
        messages = [Message(role="user", content="Search for test")]
        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
        result = await client.generate(messages=messages, tools=tools)
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == "search"

    @pytest.mark.asyncio
    async def test_generate_passes_callbacks(self):
        client = _make_anthropic_client()
        streamed = StreamedResponse(text="Hi")
        client._make_api_request = AsyncMock(return_value=streamed)
        on_text = MagicMock()
        on_thinking = MagicMock()
        messages = [Message(role="user", content="Hi")]
        await client.generate(messages=messages, on_text=on_text, on_thinking=on_thinking)
        client._make_api_request.assert_called_once()
        call_args = client._make_api_request.call_args
        assert call_args[0][3] is on_text
        assert call_args[0][4] is on_thinking

    @pytest.mark.asyncio
    async def test_generate_with_retry_enabled(self):
        client = _make_anthropic_client(retry_config=RetryConfig(enabled=True, max_retries=2, initial_delay=0.01))
        streamed = StreamedResponse(text="Retried result")
        client._make_api_request = AsyncMock(return_value=streamed)
        messages = [Message(role="user", content="Hi")]
        result = await client.generate(messages=messages)
        assert result.content == "Retried result"


class TestOpenAIGenerate:
    @pytest.mark.asyncio
    async def test_generate_simple_text(self):
        client = _make_openai_client()
        mock_response = TestOpenAIParseResponse()._make_openai_response(content="Hello from OpenAI!")
        client._make_api_request = AsyncMock(return_value=mock_response)
        messages = [Message(role="user", content="Hi")]
        result = await client.generate(messages=messages)
        assert result.content == "Hello from OpenAI!"

    @pytest.mark.asyncio
    async def test_generate_with_tools(self):
        client = _make_openai_client()
        mock_response = TestOpenAIParseResponse()._make_openai_response(
            content="",
            tool_calls=[{"id": "call_1", "name": "search", "arguments": {"q": "test"}}],
        )
        client._make_api_request = AsyncMock(return_value=mock_response)
        messages = [Message(role="user", content="Search")]
        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
        result = await client.generate(messages=messages, tools=tools)
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == "search"

    @pytest.mark.asyncio
    async def test_generate_with_retry_enabled(self):
        client = _make_openai_client(retry_config=RetryConfig(enabled=True, max_retries=2, initial_delay=0.01))
        mock_response = TestOpenAIParseResponse()._make_openai_response(content="Result")
        client._make_api_request = AsyncMock(return_value=mock_response)
        messages = [Message(role="user", content="Hi")]
        result = await client.generate(messages=messages)
        assert result.content == "Result"


class TestAnthropicMakeApiRequest:
    @pytest.mark.asyncio
    async def test_streaming_text_deltas(self):
        client = _make_anthropic_client()
        events = []

        text_delta1 = MagicMock()
        text_delta1.type = "content_block_delta"
        text_delta1.delta = MagicMock(type="text_delta", text="Hello ")
        events.append(text_delta1)

        text_delta2 = MagicMock()
        text_delta2.type = "content_block_delta"
        text_delta2.delta = MagicMock(type="text_delta", text="world")
        events.append(text_delta2)

        usage_event = MagicMock()
        usage_event.type = "message_delta"
        usage_event.usage = _make_usage_mock(input_tokens=10, output_tokens=5)
        usage_event.delta = MagicMock(stop_reason="end_turn")
        events.append(usage_event)

        stop_event = MagicMock()
        stop_event.type = "message_stop"
        events.append(stop_event)

        async def mock_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            for event in events:
                yield event

        client.client.messages.create = AsyncMock(return_value=mock_stream())
        result = await client._make_api_request(None, [{"role": "user", "content": "Hi"}])
        assert result.text == "Hello world"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_streaming_thinking_deltas(self):
        client = _make_anthropic_client()
        events = []

        thinking_delta = MagicMock()
        thinking_delta.type = "content_block_delta"
        thinking_delta.delta = MagicMock(type="thinking_delta", thinking="Let me think")
        events.append(thinking_delta)

        text_delta = MagicMock()
        text_delta.type = "content_block_delta"
        text_delta.delta = MagicMock(type="text_delta", text="Answer")
        events.append(text_delta)

        usage_event = MagicMock()
        usage_event.type = "message_delta"
        usage_event.usage = _make_usage_mock(input_tokens=5, output_tokens=10)
        usage_event.delta = MagicMock(stop_reason="stop")
        events.append(usage_event)

        stop_event = MagicMock()
        stop_event.type = "message_stop"
        events.append(stop_event)

        async def mock_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            for event in events:
                yield event

        client.client.messages.create = AsyncMock(return_value=mock_stream())
        result = await client._make_api_request(None, [{"role": "user", "content": "Hi"}])
        assert result.thinking == "Let me think"
        assert result.text == "Answer"

    @pytest.mark.asyncio
    async def test_streaming_tool_use(self):
        client = _make_anthropic_client()
        events = []

        cb = MagicMock()
        cb.type = "tool_use"
        cb.name = "search"
        block_start = MagicMock()
        block_start.type = "content_block_start"
        block_start.content_block = cb
        events.append(block_start)

        input_delta = MagicMock()
        input_delta.type = "content_block_delta"
        input_delta.delta = MagicMock(type="input_json_delta", partial_json='{"q": "test"}')
        events.append(input_delta)

        block_stop = MagicMock()
        block_stop.type = "content_block_stop"
        events.append(block_stop)

        usage_event = MagicMock()
        usage_event.type = "message_delta"
        usage_event.usage = _make_usage_mock(input_tokens=10, output_tokens=20)
        usage_event.delta = MagicMock(stop_reason="tool_use")
        events.append(usage_event)

        stop_event = MagicMock()
        stop_event.type = "message_stop"
        events.append(stop_event)

        async def mock_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            for event in events:
                yield event

        client.client.messages.create = AsyncMock(return_value=mock_stream())
        result = await client._make_api_request(None, [{"role": "user", "content": "Search"}])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["input"] == {"q": "test"}
        assert result.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_on_text_callback_called(self):
        client = _make_anthropic_client()
        collected_text: list[str] = []

        def on_text(t: str) -> None:
            collected_text.append(t)

        events = []
        for i in range(10):
            delta = MagicMock()
            delta.type = "content_block_delta"
            delta.delta = MagicMock(type="text_delta", text=f"word{i} ")
            events.append(delta)

        usage_event = MagicMock()
        usage_event.type = "message_delta"
        usage_event.usage = _make_usage_mock(input_tokens=1, output_tokens=1)
        usage_event.delta = MagicMock(stop_reason="stop")
        events.append(usage_event)

        stop_event = MagicMock()
        stop_event.type = "message_stop"
        events.append(stop_event)

        async def mock_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            for event in events:
                yield event

        client.client.messages.create = AsyncMock(return_value=mock_stream())
        result = await client._make_api_request(None, [{"role": "user", "content": "Hi"}], on_text=on_text)
        assert result.text == "".join(f"word{i} " for i in range(10))
        assert len(collected_text) > 0

    @pytest.mark.asyncio
    async def test_on_thinking_callback_called(self):
        client = _make_anthropic_client()
        collected_thinking: list[str] = []

        def on_thinking(t: str) -> None:
            collected_thinking.append(t)

        events = []
        for i in range(10):
            delta = MagicMock()
            delta.type = "content_block_delta"
            delta.delta = MagicMock(type="thinking_delta", thinking=f"think{i} ")
            events.append(delta)

        usage_event = MagicMock()
        usage_event.type = "message_delta"
        usage_event.usage = _make_usage_mock(input_tokens=1, output_tokens=1)
        usage_event.delta = MagicMock(stop_reason="stop")
        events.append(usage_event)

        stop_event = MagicMock()
        stop_event.type = "message_stop"
        events.append(stop_event)

        async def mock_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            for event in events:
                yield event

        client.client.messages.create = AsyncMock(return_value=mock_stream())
        result = await client._make_api_request(None, [{"role": "user", "content": "Hi"}], on_thinking=on_thinking)
        assert result.thinking == "".join(f"think{i} " for i in range(10))
        assert len(collected_thinking) > 0

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        client = _make_anthropic_client(retry_config=RetryConfig(enabled=False, max_delay=0))

        async def mock_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            yield MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="partial"))
            await asyncio.sleep(10)

        client.client.messages.create = AsyncMock(return_value=mock_stream())
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await client._make_api_request(None, [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client = _make_anthropic_client()
        client.client.messages.create = AsyncMock(side_effect=Exception("API error"))
        with pytest.raises(Exception, match="API error"):
            await client._make_api_request(None, [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_system_message_included_in_params(self):
        client = _make_anthropic_client()
        events = []

        usage_event = MagicMock()
        usage_event.type = "message_delta"
        usage_event.usage = _make_usage_mock(input_tokens=1, output_tokens=1)
        usage_event.delta = MagicMock(stop_reason="stop")
        events.append(usage_event)

        stop_event = MagicMock()
        stop_event.type = "message_stop"
        events.append(stop_event)

        async def mock_stream(**kwargs: Any) -> AsyncIterator[Any]:
            assert "system" in kwargs
            assert kwargs["system"][0]["text"] == "Be helpful"
            for event in events:
                yield event

        client.client.messages.create = AsyncMock(side_effect=mock_stream)
        await client._make_api_request("Be helpful", [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_tool_json_decode_error_fallback(self):
        client = _make_anthropic_client()
        events = []

        cb = MagicMock()
        cb.type = "tool_use"
        cb.name = "broken_tool"
        block_start = MagicMock()
        block_start.type = "content_block_start"
        block_start.content_block = cb
        events.append(block_start)

        input_delta = MagicMock()
        input_delta.type = "content_block_delta"
        input_delta.delta = MagicMock(type="input_json_delta", partial_json="not valid json{")
        events.append(input_delta)

        block_stop = MagicMock()
        block_stop.type = "content_block_stop"
        events.append(block_stop)

        usage_event = MagicMock()
        usage_event.type = "message_delta"
        usage_event.usage = _make_usage_mock(input_tokens=5, output_tokens=5)
        usage_event.delta = MagicMock(stop_reason="tool_use")
        events.append(usage_event)

        stop_event = MagicMock()
        stop_event.type = "message_stop"
        events.append(stop_event)

        async def mock_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            for event in events:
                yield event

        client.client.messages.create = AsyncMock(return_value=mock_stream())
        result = await client._make_api_request(None, [{"role": "user", "content": "Hi"}])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["input"] == "not valid json{"


class TestOpenAIMakeApiRequest:
    @pytest.mark.asyncio
    async def test_request_params(self):
        client = _make_openai_client()
        mock_response = TestOpenAIParseResponse()._make_openai_response()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        await client._make_api_request([{"role": "user", "content": "Hi"}])
        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.5"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        assert call_kwargs["extra_body"] == {"reasoning_split": True}

    @pytest.mark.asyncio
    async def test_request_with_tools(self):
        client = _make_openai_client()
        mock_response = TestOpenAIParseResponse()._make_openai_response()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        tools = [{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {}}}]
        await client._make_api_request([{"role": "user", "content": "Hi"}], tools=tools)
        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        client = _make_openai_client()
        client.client.chat.completions.create = AsyncMock(side_effect=Exception("Connection error"))
        with pytest.raises(Exception, match="Connection error"):
            await client._make_api_request([{"role": "user", "content": "Hi"}])


class TestLLMClientWrapper:
    def test_init_anthropic_provider(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.minimaxi.com",
                model="MiniMax-M2.5",
            )
            assert wrapper.provider == LLMProvider.ANTHROPIC
            assert wrapper.api_base == "https://api.minimaxi.com/anthropic"
            assert isinstance(wrapper._client, AnthropicClient)

    def test_init_openai_provider(self):
        with patch("mini_agent.llm.openai_client.AsyncOpenAI"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.OPENAI,
                api_base="https://api.minimaxi.com",
                model="MiniMax-M2.5",
            )
            assert wrapper.provider == LLMProvider.OPENAI
            assert wrapper.api_base == "https://api.minimaxi.com/v1"
            assert isinstance(wrapper._client, OpenAIClient)

    def test_minimax_api_suffix_anthropic(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.minimaxi.com",
            )
            assert wrapper.api_base == "https://api.minimaxi.com/anthropic"

    def test_minimax_api_suffix_openai(self):
        with patch("mini_agent.llm.openai_client.AsyncOpenAI"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.OPENAI,
                api_base="https://api.minimaxi.com",
            )
            assert wrapper.api_base == "https://api.minimaxi.com/v1"

    def test_minimax_api_strips_existing_suffix(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.minimaxi.com/anthropic",
            )
            assert wrapper.api_base == "https://api.minimaxi.com/anthropic"

    def test_third_party_api_no_suffix(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.siliconflow.cn/v1",
            )
            assert wrapper.api_base == "https://api.siliconflow.cn/v1"

    def test_trailing_slash_stripped(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.siliconflow.cn/v1/",
            )
            assert not wrapper.api_base.endswith("/")

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClient(api_key="key", provider="invalid")  # type: ignore[arg-type]

    def test_configure_m27(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.test.com",
                model="MiniMax-M2.7",
            )
            wrapper.configure_m27({"enable_extended_thinking": False, "thinking_budget_tokens": 4096})
            assert wrapper._client._enable_extended_thinking is False
            assert wrapper._client._thinking_budget_tokens == 4096

    def test_retry_callback_property(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.test.com",
            )

            def callback(_exc: Exception, _attempt: int) -> None:
                pass

            wrapper.retry_callback = callback
            assert wrapper.retry_callback is callback

    @pytest.mark.asyncio
    async def test_generate_delegates(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.test.com",
                retry_config=RetryConfig(enabled=False),
            )
            mock_response = LLMResponse(content="Hello", finish_reason="stop")
            wrapper._client.generate = AsyncMock(return_value=mock_response)
            messages = [Message(role="user", content="Hi")]
            result = await wrapper.generate(messages=messages)
            assert result.content == "Hello"

    def test_clone(self):
        with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
            wrapper = LLMClient(
                api_key="key",
                provider=LLMProvider.ANTHROPIC,
                api_base="https://api.test.com/anthropic",
                model="MiniMax-M2.5",
                retry_config=RetryConfig(max_retries=5),
            )
            cloned = wrapper.clone()
            assert cloned.api_key == wrapper.api_key
            assert cloned.provider == wrapper.provider
            assert cloned.api_base == wrapper.api_base
            assert cloned.model == wrapper.model
            assert cloned.retry_config.max_retries == 5
            assert cloned is not wrapper


class TestAnthropicPrepareRequest:
    def test_prepare_request_with_tools(self):
        client = _make_anthropic_client()
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hi"),
        ]
        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
        result = client._prepare_request(messages, tools)
        assert result["system_message"] == "Be helpful"
        assert len(result["api_messages"]) == 1
        assert result["tools"] == tools

    def test_prepare_request_without_tools(self):
        client = _make_anthropic_client()
        messages = [Message(role="user", content="Hi")]
        result = client._prepare_request(messages)
        assert result["system_message"] is None
        assert result["tools"] is None


class TestOpenAIPrepareRequest:
    def test_prepare_request(self):
        client = _make_openai_client()
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hi"),
        ]
        result = client._prepare_request(messages)
        assert len(result["api_messages"]) == 2
        assert result["api_messages"][0]["role"] == "system"
        assert result["tools"] is None

    def test_prepare_request_with_tools(self):
        client = _make_openai_client()
        messages = [Message(role="user", content="Hi")]
        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
        result = client._prepare_request(messages, tools)
        assert result["tools"] == tools


class TestAnthropicGetMaxTokens:
    def test_m25_model(self):
        client = _make_anthropic_client(model="MiniMax-M2.5")
        assert client._get_max_tokens() == 8192

    def test_m27_model(self):
        client = _make_anthropic_client(model="MiniMax-M2.7")
        assert client._get_max_tokens() == 32768

    def test_unknown_model(self):
        client = _make_anthropic_client(model="unknown-model")
        assert client._get_max_tokens() == 4096


class TestStreamedResponse:
    def test_default_values(self):
        resp = StreamedResponse()
        assert resp.text == ""
        assert resp.thinking == ""
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.cache_read_input_tokens == 0
        assert resp.cache_creation_input_tokens == 0

    def test_custom_values(self):
        resp = StreamedResponse(
            text="Hello",
            thinking="Thinking",
            tool_calls=[{"id": "1", "name": "tool", "input": {}}],
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
        )
        assert resp.text == "Hello"
        assert resp.thinking == "Thinking"
        assert len(resp.tool_calls) == 1
        assert resp.stop_reason == "end_turn"
