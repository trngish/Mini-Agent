"""LLM客户端基类。"""

from abc import ABC, abstractmethod
from typing import Any

from ..retry import RetryConfig
from ..schema import LLMResponse, Message


class LLMClientBase(ABC):
    """LLM客户端抽象基类。

    该类定义了所有LLM客户端必须实现的接口，
    无论底层API协议是什么（Anthropic、OpenAI等）。
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        retry_config: RetryConfig | None = None,
    ):
        """初始化LLM客户端。

        Args:
            api_key: API密钥，用于身份验证
            api_base: API基础URL
            model: 使用的模型名称
            retry_config: 可选的 重试配置
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        self.retry_callback = None

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """从LLM生成响应。

        Args:
            messages: 对话消息列表
            tools: 可选的Tool对象或字典列表
            **kwargs: 其他特定于实现的选项
                     （例如，用于流式处理的on_text、on_thinking回调）

        Returns:
            包含生成内容、思考过程和工具调用的LLMResponse
        """
        pass

    @abstractmethod
    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """准备API请求载荷。

        Args:
            messages: 对话消息列表
            tools: 可用的工具列表

        Returns:
            包含请求载荷的字典
        """
        pass

    @abstractmethod
    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """将内部消息格式转换为API特定格式。

        Args:
            messages: 内部Message对象列表

        Returns:
            (system_message, api_messages)元组
        """
        pass
