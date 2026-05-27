"""Anthropic LLM client implementation."""

import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic

from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from ..utils.model_utils import (
    get_max_output_tokens,
    get_thinking_budget,
    is_m27_model,
)
from .base import LLMClientBase

# Constants - avoid magic numbers
STREAM_BUFFER_SIZE = int(os.environ.get("MINI_AGENT_STREAM_BUFFER_SIZE", "8"))
DEFAULT_TIMEOUT_SECONDS = 300

logger = logging.getLogger(__name__)


@dataclass
class StreamedResponse:
    """Accumulated data from streaming response."""

    text: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class AnthropicClient(LLMClientBase):
    """LLM client using Anthropic's protocol.

    This client uses the official Anthropic SDK and supports:
    - Extended thinking content
    - Tool calling
    - Retry logic
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/anthropic",
        model: str = "MiniMax-M2.5",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax Anthropic endpoint)
            model: Model name to use (default: MiniMax-M2.5)
            retry_config: Optional retry configuration
        """
        super().__init__(api_key, api_base, model, retry_config)

        # Initialize Anthropic async client
        self.client = anthropic.AsyncAnthropic(
            base_url=api_base,
            api_key=api_key,
        )

        # M2.7 configuration attributes
        self._enable_extended_thinking = True
        self._thinking_budget_tokens = 8192  # Default, can be updated via configure_m27 or configure_thinking_budget

    def configure_thinking_budget(self, budget: int) -> None:
        """Configure thinking budget dynamically.

        This is the official API for adjusting thinking budget at runtime
        based on task complexity analysis.

        Args:
            budget: Thinking budget in tokens (0 to disable)
        """
        self._thinking_budget_tokens = max(0, min(budget, 32768))

    def configure_m27(self, config: dict[str, Any]) -> None:
        """Configure M2.7 specific settings.

        Args:
            config: M2.7 configuration dict from Config.m27
        """
        self._enable_extended_thinking = config.get("enable_extended_thinking", True)
        # Use configured budget or default to 16K
        configured_budget = config.get("thinking_budget_tokens", 16384)
        self._thinking_budget_tokens = min(configured_budget, 32768)

    async def _make_api_request(
        self,
        system_message: str | None,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> StreamedResponse:
        """Execute API request with streaming (core method that can be retried).

        Args:
            system_message: Optional system message
            api_messages: List of messages in Anthropic format
            tools: Optional list of tools
            on_text: Optional callback for incremental text content
            on_thinking: Optional callback for incremental thinking content

        Returns:
            StreamedResponse containing accumulated response data

        Raises:
            Exception: API call failed
        """
        params = {
            "model": self.model,
            "max_tokens": self._get_max_tokens(),
            "messages": api_messages,
            "stream": True,
        }

        if system_message:
            # Per-call billing optimization: use prompt caching to reduce repeated processing
            # cache_control marks let the API cache the system prompt for reuse in subsequent calls
            params["system"] = [
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if tools:
            params["tools"] = self._convert_tools(tools)

        if is_m27_model(self.model):
            thinking_config = self._get_thinking_config()
            if thinking_config:
                params["thinking"] = thinking_config

        result = StreamedResponse()
        current_tool_name = None
        current_tool_input = ""
        current_tool_id = None
        tool_call_index = 0

        # Use deque for efficient buffer management with maxlen
        buffer_size = STREAM_BUFFER_SIZE  # Configurable via MINI_AGENT_STREAM_BUFFER_SIZE
        text_buffer: deque[str] = deque(maxlen=buffer_size)
        thinking_buffer: deque[str] = deque(maxlen=buffer_size)

        try:
            stream = await self.client.messages.create(**params)  # type: ignore[call-overload]
            # Use configurable timeout from retry_config, default to 300s for backward compatibility
            timeout = self.retry_config.max_delay if self.retry_config else 300
            async with asyncio.timeout(timeout):  # type: ignore[attr-defined]
                async for event in stream:
                    try:
                        if event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                result.text += delta.text
                                text_buffer.append(delta.text)
                                if on_text and len(text_buffer) >= buffer_size:
                                    on_text("".join(text_buffer))
                                    text_buffer.clear()
                            elif delta.type == "thinking_delta":
                                result.thinking += delta.thinking
                                thinking_buffer.append(delta.thinking)
                                if on_thinking and len(thinking_buffer) >= buffer_size:
                                    on_thinking("".join(thinking_buffer))
                                    thinking_buffer.clear()
                            elif delta.type == "input_json_delta":
                                current_tool_input += getattr(delta, "partial_json", "")
                        elif event.type == "message_delta":
                            if event.usage:
                                result.input_tokens = event.usage.input_tokens or 0
                                result.output_tokens = event.usage.output_tokens or 0
                                result.cache_read_input_tokens = event.usage.cache_read_input_tokens or 0
                                result.cache_creation_input_tokens = event.usage.cache_creation_input_tokens or 0
                            if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                                result.stop_reason = event.delta.stop_reason or "stop"
                        elif event.type == "content_block_start":
                            if hasattr(event, "content_block"):
                                cb = event.content_block
                                if cb.type == "tool_use":
                                    tool_call_index += 1
                                    current_tool_id = str(tool_call_index)
                                    current_tool_name = cb.name
                                    current_tool_input = ""
                        elif event.type == "content_block_stop" or event.type == "message_stop":
                            if current_tool_name is not None:
                                try:
                                    tool_input = json.loads(current_tool_input) if current_tool_input else {}
                                except json.JSONDecodeError:
                                    tool_input = current_tool_input
                                result.tool_calls.append(
                                    {"id": current_tool_id, "name": current_tool_name, "input": tool_input}
                                )
                                current_tool_name = None
                                current_tool_id = None
                                current_tool_input = ""
                    except (AttributeError, KeyError, TypeError, ValueError) as e:
                        logger.warning("Error processing stream event: %s", e)
                        continue

            # Integrity check: ensure we received meaningful content
            # If both text and thinking are empty but we didn't stop properly, log warning
            if not result.text and not result.thinking and not result.tool_calls and result.stop_reason == "stop":
                logger.warning("Stream completed but received no content - possible truncation")

            # Flush remaining buffers
            if text_buffer and on_text:
                on_text("".join(text_buffer))
            if thinking_buffer and on_thinking:
                on_thinking("".join(thinking_buffer))

        except (TimeoutError, asyncio.TimeoutError):
            logger.error("Stream timed out after 300s")
            # Mark result as incomplete
            result.stop_reason = "timeout"
            raise
        except Exception as e:
            logger.error("Stream iteration error: %s", e)
            raise

        return result

    def _get_max_tokens(self) -> int:
        """Get max tokens based on model type.

        Uses unified model utilities for consistent configuration.
        """
        return get_max_output_tokens(self.model)

    def _get_thinking_config(self) -> dict[str, Any] | None:
        """Get extended thinking configuration for M2.7.

        M2.7 supports extended thinking with budget up to 32K tokens.
        Per-call billing optimization: deeper thinking → higher accuracy → fewer total calls

        Reference: https://www.minimaxi.com/models/text/m27

        Returns:
            Thinking configuration dict or None if disabled
        """
        if not is_m27_model(self.model):
            return None

        if not self._enable_extended_thinking:
            return None

        budget = get_thinking_budget(self.model, self._thinking_budget_tokens)
        if budget <= 0:
            return None

        return {
            "type": "enabled",
            "budget_tokens": budget,
        }

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format.

        Anthropic tool format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in Anthropic dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
            elif hasattr(tool, "to_schema"):
                result.append(tool.to_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic format.

        Per-call billing optimization:
        - Enable prompt caching to reduce repeated token processing (tokens are free but caching speeds up response)
        - Preserve complete tool results (tokens are free, more complete info = higher accuracy)
        - Mark the last user message to improve multi-turn cache hit rate

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
        """
        system_message = None
        api_messages = []

        # Find the index of the last user message for cache_control
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_idx = i
                break

        msg_idx = 0
        for msg in messages:
            if msg.role == "system":
                # Add cache control to system message for prompt caching
                system_message = msg.content
                msg_idx += 1
                continue

            if msg.role in ["user", "assistant"]:
                if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                    content_blocks = []

                    if msg.thinking:
                        content_blocks.append({"type": "thinking", "thinking": msg.thinking})

                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})  # type: ignore[dict-item]

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_id = tool_call.id
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": tool_call.function.name,
                                    "input": tool_call.function.arguments,  # type: ignore[dict-item]
                                }
                            )
                            logger.debug(
                                f"[SubAgent] Added tool_use block with id={tool_id}, name={tool_call.function.name}"
                            )

                    api_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    # Add cache_control to the last user message for multi-turn cache reuse
                    if (
                        msg.role == "user"
                        and msg_idx == last_user_idx
                        and isinstance(msg.content, str)
                        and len(msg.content) > 1024
                    ):
                        api_messages.append(
                            {
                                "role": msg.role,
                                "content": [
                                    {"type": "text", "text": msg.content, "cache_control": {"type": "ephemeral"}}
                                ],
                            }
                        )
                    else:
                        api_messages.append({"role": msg.role, "content": msg.content})
                msg_idx += 1

            elif msg.role == "tool":
                tool_content = msg.content
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": tool_content,
                            }
                        ],
                    }
                )
                msg_idx += 1

        return system_message, api_messages  # type: ignore[return-value]

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for Anthropic API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        system_message, api_messages = self._convert_messages(messages)

        return {
            "system_message": system_message,
            "api_messages": api_messages,
            "tools": tools,
        }

    def _parse_response(self, response: anthropic.types.Message | StreamedResponse) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.

        Args:
            response: StreamedResponse (from streaming) or anthropic.types.Message (legacy)

        Returns:
            LLMResponse object
        """
        if isinstance(response, StreamedResponse):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["input"],
                    ),
                )
                for tc in response.tool_calls
            ]

            total_input = (
                response.input_tokens + response.cache_read_input_tokens + response.cache_creation_input_tokens
            )
            usage = (
                TokenUsage(
                    prompt_tokens=total_input,
                    completion_tokens=response.output_tokens,
                    total_tokens=total_input + response.output_tokens,
                )
                if (response.input_tokens or response.output_tokens)
                else None
            )

            return LLMResponse(
                content=response.text,
                thinking=response.thinking or None,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=response.stop_reason or "stop",
                usage=usage,
            )

        text_content = ""
        thinking_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking_content += block.thinking
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=block.input,
                        ),
                    )
                )

        usage = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens or 0
            output_tokens = response.usage.output_tokens or 0
            cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            total_input_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
            usage = TokenUsage(
                prompt_tokens=total_input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_input_tokens + output_tokens,
            )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        """Generate response from Anthropic LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools
            on_text: Optional callback called incrementally with text content
            on_thinking: Optional callback called incrementally with thinking content

        Returns:
            LLMResponse containing the generated content
        """
        request_params = self._prepare_request(messages, tools)

        if self.retry_config.enabled:
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            response = await api_call(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
                on_text,
                on_thinking,
            )
        else:
            response = await self._make_api_request(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
                on_text,
                on_thinking,
            )

        return self._parse_response(response)
