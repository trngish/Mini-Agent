"""支持多个提供商的 LLM 客户端包装器。

本模块为不同的 LLM 提供商（Anthropic 和 OpenAI）提供统一的接口，
通过单个 LLMClient 类实现。
"""

import logging
from typing import Any

from ..retry import RetryConfig
from ..schema import LLMProvider, LLMResponse, Message
from .anthropic_client import AnthropicClient
from .base import LLMClientBase
from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class LLMClient:
    """支持多个提供商的高可用 LLM 客户端。

    此类为不同的 LLM 提供商提供统一的接口。
    它根据 provider 参数自动实例化正确的底层客户端。

    对于 MiniMax API（api.minimax.io 或 api.minimaxi.com），它会根据提供商
    追加适当的端点后缀：
    - anthropic: /anthropic
    - openai: /v1

    对于第三方 API，直接使用 api_base。
    """

    # 需要自动后缀处理的 MiniMax API 域名
    MINIMAX_DOMAINS = ("api.minimax.io", "api.minimaxi.com")

    def __init__(
        self,
        api_key: str,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        api_base: str = "https://api.minimaxi.com",
        model: str = "MiniMax-M2.5",
        retry_config: RetryConfig | None = None,
    ):
        """使用指定提供商初始化 LLM 客户端。

        Args:
            api_key: 用于认证的 API 密钥
            provider: LLM 提供商（anthropic 或 openai）
            api_base: API 基础 URL（默认：https://api.minimaxi.com）
                     对于 MiniMax API，后缀会根据提供商自动追加。
                     对于第三方 API（如 https://api.siliconflow.cn/v1），直接使用。
            model: 使用的模型名称
            retry_config: 可选的重试配置
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        # 标准化 api_base（去除尾部斜杠）
        api_base = api_base.rstrip("/")

        # 检查这是否是 MiniMax API 端点
        is_minimax = any(domain in api_base for domain in self.MINIMAX_DOMAINS)

        if is_minimax:
            # 对于 MiniMax API，根据提供商确保正确后缀
            # 首先去除任何现有后缀
            api_base = api_base.replace("/anthropic", "").replace("/v1", "")
            if provider == LLMProvider.ANTHROPIC:
                full_api_base = f"{api_base}/anthropic"
            elif provider == LLMProvider.OPENAI:
                full_api_base = f"{api_base}/v1"
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            # 对于第三方 API，直接使用 api_base
            full_api_base = api_base

        self.api_base = full_api_base

        # 实例化适当的客户端
        self._client: LLMClientBase
        if provider == LLMProvider.ANTHROPIC:
            self._client = AnthropicClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
            )
        elif provider == LLMProvider.OPENAI:
            self._client = OpenAIClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info("Initialized LLM client with provider: %s, api_base: %s", provider, full_api_base)

    @property
    def retry_callback(self) -> Any:
        """获取重试回调。"""
        return self._client.retry_callback

    @retry_callback.setter
    def retry_callback(self, value: Any) -> None:
        """设置重试回调。"""
        self._client.retry_callback = value

    def configure_m27(self, config: dict[str, Any]) -> None:
        """在底层 LLM 客户端上配置 M2.7 特定设置。

        Args:
            config: M2.7 配置字典
        """
        if config.get("enable_extended_thinking") is not None:
            self._client._enable_extended_thinking = config["enable_extended_thinking"]  # type: ignore[attr-defined]
        if config.get("thinking_budget_tokens") is not None:
            self._client._thinking_budget_tokens = config["thinking_budget_tokens"]  # type: ignore[attr-defined]

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """从 LLM 生成响应。

        Args:
            messages: 对话消息列表
            tools: 可选的 Tool 对象或字典列表
            **kwargs: 其他选项（如 on_text、on_thinking）

        Returns:
            包含生成内容的 LLMResponse
        """
        # 在发送前验证消息，以及早发现结构错误
        from ..utils.message_validator import MessageValidator, ValidationError

        try:
            for _idx, msg in enumerate(messages):
                MessageValidator.validate_message(msg)
        except ValidationError as e:
            logger.warning("Message validation issue at index %d: %s", _idx, e)

        return await self._client.generate(messages, tools, **kwargs)

    def clone(self) -> "LLMClient":
        """创建此 LLMClient 的深拷贝。

        每个 SubAgent 需要自己的 LLMClient，以避免在并发运行时
        出现 tool_call.id 冲突。API 拒绝带有重复工具 ID 的请求。

        Returns:
            具有相同配置的新 LLMClient 实例
        """
        return LLMClient(
            api_key=self.api_key,
            provider=self.provider,
            api_base=self.api_base,
            model=self.model,
            retry_config=self.retry_config,
        )