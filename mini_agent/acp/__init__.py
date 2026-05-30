"""ACP (Agent Client Protocol) 桥接器，用于 Mini-Agent。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (  # type: ignore[import-untyped]
    PROTOCOL_VERSION,
    AgentSideConnection,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    session_notification,
    start_tool_call,
    stdio_streams,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from acp import (  # type: ignore[import-untyped]
    NewSessionRequest as BaseNewSessionRequest,
)
from acp.schema import AgentCapabilities, Implementation  # type: ignore[import-untyped]
from pydantic import field_validator

from mini_agent.agent import Agent
from mini_agent.bootstrap import add_workspace_tools, initialize_base_tools
from mini_agent.config import Config
from mini_agent.llm import LLMClient
from mini_agent.retry import RetryConfig as RetryConfigBase
from mini_agent.schema import Message


class NewSessionRequest(BaseNewSessionRequest):  # type: ignore[misc]
    """重写以使 cwd 和 mcpServers 成为可选参数。"""

    cwd: str | None = None
    mcpServers: list[Any] = []  # noqa: N815


logger = logging.getLogger(__name__)


try:

    class InitializeRequestPatch(InitializeRequest):  # type: ignore[misc]
        @field_validator("protocolVersion", mode="before")
        @classmethod
        def normalize_protocol_version(cls, value: Any) -> int:
            if isinstance(value, str):
                try:
                    return int(value.split(".")[0])
                except Exception:
                    return 1
            if isinstance(value, (int, float)):
                return int(value)
            return 1

    InitializeRequest = InitializeRequestPatch
    InitializeRequest.model_rebuild(force=True)
except Exception:  # pragma: no cover - defensive
    logger.debug("ACP schema patch skipped")


@dataclass
class SessionState:
    agent: Agent
    cancelled: bool = False


class MiniMaxACPAgent:
    """Minimal ACP 适配器，用于封装现有的 Agent 运行时。"""

    def __init__(
        self,
        conn: AgentSideConnection,
        config: Config,
        llm: LLMClient,
        base_tools: list[Any],
        system_prompt: str,
    ):
        self._conn = conn
        self._config = config
        self._llm = llm
        self._base_tools = base_tools
        self._system_prompt = system_prompt
        self._sessions: dict[str, SessionState] = {}

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:  # noqa: ARG002
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(loadSession=False),
            agentInfo=Implementation(name="mini-agent", title="Mini-Agent", version="0.1.0"),
        )

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:  # noqa: N802
        session_id = f"sess-{len(self._sessions)}-{uuid4().hex[:8]}"
        workspace = Path(params.cwd or self._config.agent.workspace_dir).expanduser()
        if not workspace.is_absolute():
            workspace = workspace.resolve()
        logger.debug(
            f"Creating session with workspace: {workspace} "
            f"(cwd={params.cwd}, config.workspace_dir={self._config.agent.workspace_dir})"
        )
        tools = list(self._base_tools)
        await add_workspace_tools(tools, self._config, workspace)
        agent = Agent(
            llm_client=self._llm,
            system_prompt=self._system_prompt,
            tools=tools,
            max_steps=self._config.agent.max_steps,
            workspace_dir=str(workspace),
        )
        self._sessions[session_id] = SessionState(agent=agent)
        return NewSessionResponse(sessionId=session_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        state = self._sessions.get(params.sessionId)
        if not state:
            # 如果找不到会话则自动创建（兼容跳过 newSession 的客户端）
            logger.warning(f"Session '{params.sessionId}' not found, auto-creating new session")
            try:
                new_session = await self.new_session(
                    NewSessionRequest(cwd=self._config.agent.workspace_dir, mcpServers=[])
                )
                logger.debug(f"Auto-created session: {new_session.sessionId}")
            except Exception as e:
                logger.exception(f"Failed to auto-create session: {e}")
                return PromptResponse(stopReason="refusal")
            state = self._sessions.get(new_session.sessionId)
            if not state:
                logger.error("Failed to create session state")
                return PromptResponse(stopReason="refusal")
        state.cancelled = False
        user_text = "\n".join(
            block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "") for block in params.prompt
        )
        state.agent.add_user_message(user_text)
        stop_reason = await self._run_turn(state, params.sessionId)
        return PromptResponse(stopReason=stop_reason)

    async def cancel(self, params: CancelNotification) -> None:
        state = self._sessions.get(params.sessionId)
        if state:
            state.cancelled = True

    async def _run_turn(self, state: SessionState, session_id: str) -> str:
        agent = state.agent
        for _ in range(agent.max_steps):
            if state.cancelled:
                return "cancelled"
            tools_list = list(agent.tools.values())
            try:
                response = await agent.llm.generate(messages=agent.messages, tools=tools_list)
            except Exception as exc:
                logger.exception("LLM error")
                await self._send(session_id, update_agent_message(text_block(f"Error: {exc}")))
                return "refusal"
            if response.thinking:
                await self._send(session_id, update_agent_thought(text_block(response.thinking)))
            if response.content:
                await self._send(session_id, update_agent_message(text_block(response.content)))
            state.agent.append_message(
                Message(
                    role="assistant",
                    content=response.content,
                    thinking=response.thinking,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                return "end_turn"
            for call in response.tool_calls:
                name, args = call.function.name, call.function.arguments
                # 显示工具名称及关键参数以便更好地查看
                args_preview = (
                    ", ".join(f"{k}={repr(v)[:50]}" for k, v in list(args.items())[:2])
                    if isinstance(args, dict)
                    else ""
                )
                label = f"🔧 {name}({args_preview})" if args_preview else f"🔧 {name}()"
                await self._send(session_id, start_tool_call(call.id, label, kind="execute", raw_input=args))
                tool = agent.tools.get(name)
                if not tool:
                    text, status = f"[ERROR] Unknown tool: {name}", "failed"
                else:
                    try:
                        result = await tool.execute(**args)
                        status = "completed" if result.success else "failed"
                        prefix = "[OK]" if result.success else "[ERROR]"
                        text = (
                            f"{prefix} {result.content if result.success else result.error or 'Tool execution failed'}"
                        )
                    except Exception as exc:
                        status, text = "failed", f"[ERROR] Tool error: {exc}"
                await self._send(
                    session_id,
                    update_tool_call(call.id, status=status, content=[tool_content(text_block(text))], raw_output=text),
                )
                agent.append_message(Message(role="tool", content=text, tool_call_id=call.id, name=name))
        return "max_turn_requests"

    async def _send(self, session_id: str, update: Any) -> None:
        await self._conn.sessionUpdate(session_notification(session_id, update))


async def run_acp_server(config: Config | None = None) -> None:
    """以 ACP 兼容的 stdio 服务器方式运行 Mini-Agent。"""
    config = config or Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    base_tools, skill_loader = await initialize_base_tools(config)
    prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if prompt_path and prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = "You are a helpful AI assistant."
    if skill_loader:
        meta = skill_loader.get_skills_metadata_prompt()
        if meta:
            system_prompt = f"{system_prompt.rstrip()}\n\n{meta}"
    from ..schema import LLMProvider

    rcfg = config.llm.retry
    provider = LLMProvider.ANTHROPIC if config.llm.provider.lower() == "anthropic" else LLMProvider.OPENAI
    llm = LLMClient(
        api_key=config.llm.api_key,
        provider=provider,
        api_base=config.llm.api_base,
        model=config.llm.model,
        retry_config=RetryConfigBase(
            enabled=rcfg.enabled,
            max_retries=rcfg.max_retries,
            initial_delay=rcfg.initial_delay,
            max_delay=rcfg.max_delay,
            exponential_base=rcfg.exponential_base,
        ),
    )
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: MiniMaxACPAgent(conn, config, llm, base_tools, system_prompt), writer, reader)
    logger.info("Mini-Agent ACP server running")
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(run_acp_server())


__all__ = ["MiniMaxACPAgent", "run_acp_server", "main"]
