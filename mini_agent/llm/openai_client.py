"""OpenAI LLM 客户端实现。"""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from ..utils.model_utils import get_max_output_tokens
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClientBase):
    """使用 OpenAI 协议的 LLM 客户端。

    该客户端使用官方 OpenAI SDK，支持：
    - 推理内容（通过 reasoning_split=True）
    - 工具调用
    - 重试逻辑
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/v1",
        model: str = "MiniMax-M2.5",
        retry_config: RetryConfig | None = None,
    ):
        """初始化 OpenAI 客户端。

        Args:
            api_key: 用于认证的 API 密钥
            api_base: API 基础 URL（默认：MiniMax OpenAI 端点）
            model: 使用的模型名称（默认：MiniMax-M2.5）
            retry_config: 可选的重试配置
        """
        super().__init__(api_key, api_base, model, retry_config)

        # 初始化 OpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        # M2.7 配置属性
        self._enable_extended_thinking = True
        self._thinking_budget_tokens = 8192  # 默认值，可通过 configure_m27 或 configure_thinking_budget 更新

    def configure_thinking_budget(self, budget: int) -> None:
        """动态配置思考预算。

        这是用于根据任务复杂度分析在运行时调整思考预算的官方 API。

        Args:
            budget: 以 token 为单位的思考预算（0 表示禁用）
        """
        self._thinking_budget_tokens = max(0, min(budget, 32768))

    def configure_m27(self, config: dict[str, Any]) -> None:
        """配置 M2.7 特定设置。

        Args:
            config: 来自 Config.m27 的 M2.7 配置字典
        """
        self._enable_extended_thinking = config.get("enable_extended_thinking", True)
        # 使用配置的 budget 或默认为 16K
        configured_budget = config.get("thinking_budget_tokens", 16384)
        self._thinking_budget_tokens = min(configured_budget, 32768)

    async def _make_streaming_request(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        on_text: Any = None,
        on_thinking: Any = None,
    ) -> Any:
        """执行带回调支持的流式 API 请求（A1 修复）。

        Args:
            api_messages: OpenAI 格式的消息列表
            tools: 可选的工具列表
            on_text: 文本增量回调
            on_thinking: 思考/推理增量回调

        Returns:
            累积的响应数据字典，包含 text、thinking、tool_calls、usage
        """
        params = {
            "model": self.model,
            "messages": api_messages,
            "extra_body": {"reasoning_split": True},
            "max_tokens": get_max_output_tokens(self.model),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            params["tools"] = self._convert_tools(tools)

        accumulated = {
            "text": "",
            "thinking": "",
            "tool_calls": {},
            "finish_reason": "stop",
            "usage": None,
        }

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 文本内容
                if delta.content and on_text:
                    on_text(delta.content)
                    accumulated["text"] += delta.content
                elif delta.content:
                    accumulated["text"] += delta.content

                # 推理/思考内容
                if hasattr(delta, "reasoning_details") and delta.reasoning_details:
                    for detail in delta.reasoning_details:
                        if hasattr(detail, "text") and detail.text:
                            if on_thinking:
                                on_thinking(detail.text)
                            accumulated["thinking"] += detail.text

                # 工具调用（跨分块累积）
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated["tool_calls"]:
                            accumulated["tool_calls"][idx] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = accumulated["tool_calls"][idx]
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                entry["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                entry["function"]["arguments"] += tc.function.arguments

                # 结束原因
                if chunk.choices[0].finish_reason:
                    accumulated["finish_reason"] = chunk.choices[0].finish_reason

                # 使用量（在末尾通过 stream_options 发送）
                if hasattr(chunk, "usage") and chunk.usage:
                    accumulated["usage"] = chunk.usage

        except Exception as e:
            logger.warning("Streaming request error: %s", e)
            raise

        return accumulated

    async def _make_api_request(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> Any:
        """执行 API 请求（可重试的核心方法）。

        Args:
            api_messages: OpenAI 格式的消息列表
            tools: 可选的工具列表

        Returns:
            OpenAI ChatCompletion 响应（包含 usage 的完整响应）

        Raises:
            Exception: API 调用失败
        """
        params = {
            "model": self.model,
            "messages": api_messages,
            # 启用 reasoning_split 以分离思考内容
            "extra_body": {"reasoning_split": True},
            "max_tokens": get_max_output_tokens(self.model),
        }

        if tools:
            params["tools"] = self._convert_tools(tools)

        # 使用 OpenAI SDK 的 chat.completions.create
        response = await self.client.chat.completions.create(**params)  # type: ignore[call-overload]
        # 返回完整响应以便访问 usage 信息
        return response

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """将工具转换为 OpenAI 格式。

        Args:
            tools: Tool 对象或字典的列表

        Returns:
            OpenAI 字典格式的工具列表
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                # 如果已经是字典，检查是否是 OpenAI 格式
                if "type" in tool and tool["type"] == "function":
                    result.append(tool)
                else:
                    # 假设是 Anthropic 格式，转换为 OpenAI
                    result.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"],
                            },
                        }
                    )
            elif hasattr(tool, "to_openai_schema"):
                # 具有 to_openai_schema 方法的 Tool 对象
                result.append(tool.to_openai_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """将内部消息转换为 OpenAI 格式。

        Args:
            messages: 内部 Message 对象的列表

        Returns:
            (system_message, api_messages) 元组
            注意：OpenAI 在 messages 数组中包含系统消息
        """
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                # OpenAI 在 messages 数组中包含系统消息
                api_messages.append({"role": "system", "content": msg.content})
                continue

            # 用户消息
            if msg.role == "user":
                api_messages.append({"role": "user", "content": msg.content})

            # 助手消息
            elif msg.role == "assistant":
                assistant_msg: dict[str, Any] = {"role": "assistant"}

                # 如果有内容则添加
                if msg.content:
                    assistant_msg["content"] = msg.content

                # 如果有工具调用则添加
                if msg.tool_calls:
                    tool_calls_list = []
                    for tool_call in msg.tool_calls:
                        tool_calls_list.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": json.dumps(tool_call.function.arguments),
                                },
                            }
                        )
                    assistant_msg["tool_calls"] = tool_calls_list

                # 重要：如果存在思考内容，添加 reasoning_details
                # 这对于交错思考（Interleaved Thinking）正常工作至关重要！
                # 完整的 response_message（包括 reasoning_details）必须
                # 保存在消息历史中，并在下一轮传递回模型。
                # 这确保了模型的思维链不会中断。
                if msg.thinking:
                    assistant_msg["reasoning_details"] = [{"text": msg.thinking}]

                api_messages.append(assistant_msg)

            # 工具结果消息
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,  # type: ignore[dict-item]
                        "content": msg.content,
                    }
                )

        return None, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """准备 OpenAI API 请求。

        Args:
            messages: 对话消息列表
            tools: 可用的工具列表

        Returns:
            包含请求参数的字典
        """
        _, api_messages = self._convert_messages(messages)

        return {
            "api_messages": api_messages,
            "tools": tools,
        }

    def _parse_streamed_response(self, accumulated: dict[str, Any]) -> LLMResponse:
        """将累积的流式响应解析为 LLMResponse（A1 修复）。"""
        text = accumulated.get("text", "")
        thinking = accumulated.get("thinking", "") or None

        # 从累积字典解析工具调用
        tool_calls = []
        for idx in sorted(accumulated.get("tool_calls", {}).keys()):
            tc_data = accumulated["tool_calls"][idx]
            try:
                arguments = json.loads(tc_data["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc_data["id"],
                    type="function",
                    function=FunctionCall(
                        name=tc_data["function"]["name"],
                        arguments=arguments,
                    ),
                )
            )

        # 解析使用量
        usage = None
        if accumulated.get("usage"):
            u = accumulated["usage"]
            usage = TokenUsage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        return LLMResponse(
            content=text,
            thinking=thinking,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=accumulated.get("finish_reason", "stop"),
            usage=usage,
        )

    def _parse_response(self, response: Any) -> LLMResponse:
        """将 OpenAI 响应解析为 LLMResponse。

        Args:
            response: OpenAI ChatCompletion 响应（完整响应对象）

        Returns:
            LLMResponse 对象
        """
        # 从响应中获取消息
        message = response.choices[0].message

        # 提取文本内容
        text_content = message.content or ""

        # 从 reasoning_details 提取思考内容
        thinking_content = ""
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            # reasoning_details 是一个推理块列表
            for detail in message.reasoning_details:
                if hasattr(detail, "text"):
                    thinking_content += detail.text

        # 提取工具调用
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # 从 JSON 字符串解析参数
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, KeyError, TypeError):
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        type="function",
                        function=FunctionCall(
                            name=tool_call.function.name,
                            arguments=arguments,
                        ),
                    )
                )

        # 从响应中提取 token 使用量
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        finish_reason = getattr(response.choices[0], "finish_reason", None) or "stop"

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        on_text: Any = None,
        on_thinking: Any = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        """从 OpenAI LLM 生成响应（A1 修复：添加了流式支持）。

        Args:
            messages: 对话消息列表
            tools: 可用的工具列表
            on_text: 用于流式文本内容的可选回调
            on_thinking: 用于流式思考/推理内容的可选回调

        Returns:
            包含生成内容的 LLMResponse
        """
        request_params = self._prepare_request(messages, tools)

        # A1 修复：当提供回调时使用流式处理
        if on_text or on_thinking:
            accumulated = await self._make_streaming_request(
                request_params["api_messages"],
                request_params["tools"],
                on_text=on_text,
                on_thinking=on_thinking,
            )
            return self._parse_streamed_response(accumulated)

        # 后备方案：用于重试或无回调场景的非流式处理
        if self.retry_config.enabled:
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            response = await api_call(
                request_params["api_messages"],
                request_params["tools"],
            )
        else:
            response = await self._make_api_request(
                request_params["api_messages"],
                request_params["tools"],
            )

        return self._parse_response(response)