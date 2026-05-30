"""会话保存/恢复管理器，带索引缓存。"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .schema import Message
from .utils import Colors
from .utils.atomic_io import atomic_read_json, atomic_write_json

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器，支持会话持久化（保存/恢复/列表）并带有索引缓存。

    维护一个索引文件以加快会话列表速度，无需读取所有会话文件。
    使用原子 I/O 操作实现崩溃安全。
    """

    INDEX_FILENAME = ".session_index.json"
    MAX_SESSIONS_IN_INDEX = 1000
    DEFAULT_MAX_SESSIONS = 100

    def __init__(self, workspace_dir: Path | None = None, logger: Any = None, session_dir: Path | None = None):
        self.workspace_dir = workspace_dir
        self.logger = logger

        # D14修复：工作区隔离的会话存储
        # 不同项目的会话存储在基于工作区哈希的单独子目录中，
        # 防止跨项目冲突。
        if session_dir:
            self.session_dir = session_dir
        elif workspace_dir and os.environ.get("MINI_AGENT_SESSION_ISOLATION", "1") != "0":
            ws_hash = hashlib.sha256(str(workspace_dir.absolute()).encode()).hexdigest()[:12]
            self.session_dir = Path.home() / ".mini-agent" / "sessions" / ws_hash
        else:
            self.session_dir = Path.home() / ".mini-agent" / "sessions"

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._index: list[dict[str, Any]] | None = None
        self._index_loaded = False
        self._index_lock = Lock()  # Thread lock for index operations
        self._max_sessions = int(os.environ.get("MINI_AGENT_MAX_SESSIONS", str(self.DEFAULT_MAX_SESSIONS)))

    def _get_index_path(self) -> Path:
        """获取索引文件的路径。"""
        return self.session_dir / self.INDEX_FILENAME

    def _load_index(self) -> list[dict[str, Any]]:
        """从磁盘加载会话索引。"""
        data = atomic_read_json(self._get_index_path())
        if isinstance(data, list):
            return data
        return []

    def _save_index(self, index: list[dict[str, Any]]) -> None:
        """将会话索引原子化地保存到磁盘。"""
        atomic_write_json(index, self._get_index_path())

    def _ensure_index_loaded(self) -> None:
        """确保索引已加载（线程安全）。"""
        if not self._index_loaded:
            with self._index_lock:
                if not self._index_loaded:  # Double-check under lock
                    self._index = self._load_index()
                    self._index_loaded = True

    def _invalidate_index(self) -> None:
        """使缓存的索引失效。"""
        self._index = None
        self._index_loaded = False
        self._max_sessions = int(os.environ.get("MINI_AGENT_MAX_SESSIONS", str(self.DEFAULT_MAX_SESSIONS)))

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        return [msg.model_dump() for msg in messages]

    def _deserialize_messages(self, data: list[dict[str, Any]]) -> list[Message]:
        return [Message(**msg) for msg in data]

    def _save_to_file(
        self,
        messages: list[Message],
        session_id: str,
        label: str,
        result: str | None = None,
        analysis: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        """内部方法，保存消息到会话文件（D3：添加state参数）。

        Args:
            messages: 要保存的消息列表
            session_id: 会话ID
            label: 会话标签
            result: 可选的最新运行结果内容
            analysis: 可选的最新分析结果
        """
        timestamp = datetime.now().isoformat()
        data = {
            "id": session_id,
            "label": label,
            "created": timestamp,
            "messages": self._serialize_messages(messages),
            "result": result,
            "analysis": analysis,
            "state": state or {},
        }
        path = self.session_dir / f"{session_id}.json"
        atomic_write_json(data, path)

    def _update_index(self, session_id: str, label: str, message_count: int, timestamp: str) -> None:
        """内部方法，用于更新会话索引（线程安全）。

        Args:
            session_id: 会话ID
            label: 会话标签
            message_count: 消息数量
            timestamp: 创建时间戳
        """
        with self._index_lock:
            self._ensure_index_loaded()
            if self._index is None:
                self._index = []

            index_entry = {
                "id": session_id,
                "label": label,
                "created": timestamp,
                "message_count": message_count,
            }
            self._index.insert(0, index_entry)

            # 如果索引过大则进行修剪
            if len(self._index) > self.MAX_SESSIONS_IN_INDEX:
                self._index = self._index[: self.MAX_SESSIONS_IN_INDEX]

            self._save_index(self._index)

    def save(
        self,
        messages: list[Message],
        label: str = "",
        result: str | None = None,
        analysis: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> str:
        """保存消息到会话文件。返回会话ID。"""
        # D1修复：保存前检查ID冲突
        max_attempts = 5
        session_id = str(uuid.uuid4())[:8]
        for _ in range(max_attempts):
            path = self.session_dir / f"{session_id}.json"
            if not path.exists():
                break
            session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        self._save_to_file(messages, session_id, label, result, analysis, state)
        self._update_index(session_id, label, len(messages), timestamp)

        # D13修复：当超过最大数量时自动清理旧会话
        self._enforce_session_limit()

        return session_id

    def load(self, session_id: str) -> tuple[list[Message] | None, str | None, dict[str, Any] | None]:
        """从会话文件加载消息、结果和运行时状态。

        Returns:
            (消息列表，结果字符串)的元组。如果未找到则任一可能为None。
        """
        path = self.session_dir / f"{session_id}.json"
        data = atomic_read_json(path)
        if data is None:
            return None, None, None
        messages = self._deserialize_messages(data.get("messages", []))
        result = data.get("result")
        state = data.get("state", {})
        return messages, result, state

    def load_messages(self, session_id: str) -> list[Message] | None:
        """仅从会话文件加载消息（向后兼容）。"""
        messages, _, _ = self.load(session_id)
        return messages

    def load_analysis(self, session_id: str) -> str | None:
        """从会话文件加载分析结果。"""
        path = self.session_dir / f"{session_id}.json"
        data = atomic_read_json(path)
        if data is None:
            return None
        return data.get("analysis")

    def save_analysis(self, session_id: str, analysis: str) -> None:
        """保存分析结果到现有会话文件。"""
        path = self.session_dir / f"{session_id}.json"
        data = atomic_read_json(path)
        if data is None:
            return
        data["analysis"] = analysis
        data["updated"] = datetime.now().isoformat()
        atomic_write_json(data, path)

    def list_sessions(self) -> list[dict[str, Any]]:
        """使用缓存索引列出所有已保存的会话。"""
        self._ensure_index_loaded()

        if self._index is None or not self._index:
            return self._list_sessions_scan()

        # 过滤存在的会话并清理陈旧的索引条目
        sessions = []
        stale_ids = []
        for entry in self._index:
            path = self.session_dir / f"{entry['id']}.json"
            if path.exists():
                sessions.append(
                    {
                        "id": entry.get("id", path.stem),
                        "label": entry.get("label", ""),
                        "created": entry.get("created", ""),
                        "messages": entry.get("message_count", 0),
                        "has_result": False,  # 将在下面更新
                    }
                )
            else:
                stale_ids.append(entry["id"])

        if stale_ids:
            with self._index_lock:
                self._index = [s for s in self._index if s["id"] not in stale_ids]
                self._save_index(self._index)

        # 为每个会话更新 has_result
        for session in sessions:
            path = self.session_dir / f"{session['id']}.json"
            data = atomic_read_json(path)
            if data is not None and data.get("result"):
                session["has_result"] = True

        return sessions

    def _list_sessions_scan(self) -> list[dict[str, Any]]:
        """通过扫描目录列出会话（后备方案）。"""
        sessions = []
        for path in sorted(self.session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if path.stem == self.INDEX_FILENAME:
                continue
            data = atomic_read_json(path)
            if data is not None:
                sessions.append(
                    {
                        "id": data.get("id", path.stem),
                        "label": data.get("label", ""),
                        "created": data.get("created", ""),
                        "messages": len(data.get("messages", [])),
                    }
                )
        return sessions

    def get_latest_session_id(self) -> str | None:
        """获取最近会话的ID。

        Returns:
            最近会话的ID，如果不存在会话则返回None
        """
        sessions = self.list_sessions()
        return sessions[0]["id"] if sessions else None

    def _enforce_session_limit(self) -> int:
        """D13修复：当超过最大数量时自动清理旧会话。"""
        removed = 0
        try:
            sessions = sorted(
                self.session_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            excess = len(sessions) - self._max_sessions
            for path in sessions[:excess]:
                if path.name == self.INDEX_FILENAME:
                    continue
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
            if removed > 0:
                self._invalidate_index()
        except Exception as e:
            logger.warning(f"Failed to enforce session limit: {e}")
        return removed

    def delete(self, session_id: str) -> bool:
        """删除会话文件。"""
        path = self.session_dir / f"{session_id}.json"
        file_existed = path.exists()
        if file_existed:
            path.unlink()

        # 始终更新索引，即使文件已被删除
        with self._index_lock:
            self._ensure_index_loaded()
            if self._index is not None:
                self._index = [s for s in self._index if s.get("id") != session_id]
                self._save_index(self._index)

        return file_existed

    def clear_index(self) -> None:
        """清除会话索引缓存。"""
        self._invalidate_index()
        index_path = self._get_index_path()
        if index_path.exists():
            index_path.unlink()

    def save_session(self, messages: list[Message], session_id: str | None = None, label: str = "", result: str | None = None, analysis: str | None = None, state: dict[str, Any] | None = None) -> str:
        """保存消息到会话并提供用户反馈。返回会话ID。"""
        if session_id is not None:
            timestamp = datetime.now().isoformat()
            self._save_to_file(messages, session_id, label, result, analysis, state)
            self._update_index(session_id, label, len(messages), timestamp)
            print(f"{Colors.GREEN}✅ 会话已保存: {session_id}{Colors.RESET}")
            return session_id

        sid = self.save(messages, label, result, analysis, state)
        print(f"{Colors.GREEN}✅ 会话已保存: {sid}{Colors.RESET}")
        return sid

    def load_session(self, session_id: str) -> list[Message] | None:
        """加载会话并提供用户反馈。返回消息列表或None。"""
        messages, _, _ = self.load(session_id)
        if messages is None:
            print(f"{Colors.RED}❌ 会话未找到: {session_id}{Colors.RESET}")
            return None
        system_count = sum(1 for m in messages if m.role == "system")
        print(
            f"{Colors.GREEN}✅ 会话已恢复: {session_id}"
            f" ({len(messages)} 条消息, {system_count} 条系统提示){Colors.RESET}"
        )
        return messages

