"""Session save/resume manager with index caching."""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import Message
from .utils import Colors
from .utils.atomic_io import atomic_read_json, atomic_write_json


class SessionManager:
    """Manages session persistence (save/resume/list) with index caching.

    Maintains an index file for faster session listing without reading
    all session files. Uses atomic I/O for crash-safe operations.
    """

    INDEX_FILENAME = ".session_index.json"
    MAX_SESSIONS_IN_INDEX = 1000
    DEFAULT_MAX_SESSIONS = 100

    def __init__(self, workspace_dir: Path | None = None, logger: Any = None, session_dir: Path | None = None):
        self.workspace_dir = workspace_dir
        self.logger = logger

        # D14 FIX: Workspace-isolated session storage
        # Sessions from different projects are stored in separate subdirectories
        # based on workspace hash, preventing cross-project collision.
        if session_dir:
            self.session_dir = session_dir
        elif workspace_dir and os.environ.get("MINI_AGENT_SESSION_ISOLATION", "1") != "0":
            ws_hash = hashlib.md5(str(workspace_dir.absolute()).encode()).hexdigest()[:12]
            self.session_dir = Path.home() / ".mini-agent" / "sessions" / ws_hash
        else:
            self.session_dir = Path.home() / ".mini-agent" / "sessions"

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._index: list[dict[str, Any]] | None = None
        self._index_loaded = False
        self._max_sessions = int(os.environ.get("MINI_AGENT_MAX_SESSIONS", str(self.DEFAULT_MAX_SESSIONS)))

    def _get_index_path(self) -> Path:
        """Get path to index file."""
        return self.session_dir / self.INDEX_FILENAME

    def _load_index(self) -> list[dict[str, Any]]:
        """Load session index from disk."""
        data = atomic_read_json(self._get_index_path())
        if isinstance(data, list):
            return data
        return []

    def _save_index(self, index: list[dict[str, Any]]) -> None:
        """Save session index to disk atomically."""
        atomic_write_json(index, self._get_index_path())

    def _ensure_index_loaded(self) -> None:
        """Ensure index is loaded."""
        if not self._index_loaded:
            self._index = self._load_index()
            self._index_loaded = True

    def _invalidate_index(self) -> None:
        """Invalidate cached index."""
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
        """Internal method to save messages to a session file (D3: added state param).

        Args:
            messages: List of messages to save
            session_id: Session ID
            label: Session label
            result: Optional result content from the last run
            analysis: Optional analysis result from the last run
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
        """Internal method to update session index.

        Args:
            session_id: Session ID
            label: Session label
            message_count: Number of messages
            timestamp: Creation timestamp
        """
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

        # Trim index if too large
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
        """Save messages to a session file. Returns session ID."""
        # D1 FIX: Check for ID collisions before saving
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

        # D13 FIX: Auto-cleanup old sessions when exceeding max
        self._enforce_session_limit()

        return session_id

    def load(self, session_id: str) -> tuple[list[Message] | None, str | None, dict[str, Any] | None]:
        """Load messages, result, and runtime state from a session file.

        Returns:
            Tuple of (messages list, result string). Either may be None if not found.
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
        """Load messages only from a session file (backwards compatibility)."""
        messages, _, _ = self.load(session_id)
        return messages

    def load_analysis(self, session_id: str) -> str | None:
        """Load analysis result from a session file."""
        path = self.session_dir / f"{session_id}.json"
        data = atomic_read_json(path)
        if data is None:
            return None
        return data.get("analysis")

    def save_analysis(self, session_id: str, analysis: str) -> None:
        """Save analysis result to an existing session file."""
        path = self.session_dir / f"{session_id}.json"
        data = atomic_read_json(path)
        if data is None:
            return
        data["analysis"] = analysis
        data["updated"] = datetime.now().isoformat()
        atomic_write_json(data, path)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions using cached index."""
        self._ensure_index_loaded()

        if self._index is None or not self._index:
            return self._list_sessions_scan()

        # Filter to existing sessions and clean stale index entries
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
                        "has_result": False,  # Will be updated below
                    }
                )
            else:
                stale_ids.append(entry["id"])

        if stale_ids:
            self._index = [s for s in self._index if s["id"] not in stale_ids]
            self._save_index(self._index)

        # Update has_result for each session
        for session in sessions:
            path = self.session_dir / f"{session['id']}.json"
            data = atomic_read_json(path)
            if data is not None and data.get("result"):
                session["has_result"] = True

        return sessions

    def _list_sessions_scan(self) -> list[dict[str, Any]]:
        """List sessions by scanning directory (fallback)."""
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
        """Get the most recent session ID.

        Returns:
            Session ID of most recent session, or None if no sessions exist
        """
        sessions = self.list_sessions()
        return sessions[0]["id"] if sessions else None

    def _enforce_session_limit(self) -> int:
        """D13 FIX: Auto-cleanup oldest sessions when exceeding max_sessions."""
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
        except Exception:
            pass
        return removed

    def delete(self, session_id: str) -> bool:
        """Delete a session file."""
        path = self.session_dir / f"{session_id}.json"
        file_existed = path.exists()
        if file_existed:
            path.unlink()

        # Always update index, even if file was already deleted
        self._ensure_index_loaded()
        if self._index is not None:
            self._index = [s for s in self._index if s.get("id") != session_id]
            self._save_index(self._index)

        return file_existed

    def clear_index(self) -> None:
        """Clear the session index cache."""
        self._invalidate_index()
        index_path = self._get_index_path()
        if index_path.exists():
            index_path.unlink()

    def save_session(self, messages: list[Message], session_id: str | None = None, label: str = "", state: dict[str, Any] | None = None) -> str:
        """Save messages to a session with user feedback. Returns session ID."""
        if session_id is not None:
            timestamp = datetime.now().isoformat()
            self._save_to_file(messages, session_id, label, state=state)
            self._update_index(session_id, label, len(messages), timestamp)
            print(f"{Colors.GREEN}✅ Session saved: {session_id}{Colors.RESET}")
            return session_id

        sid = self.save(messages, label, state=state)
        print(f"{Colors.GREEN}✅ Session saved: {sid}{Colors.RESET}")
        return sid

    def load_session(self, session_id: str) -> list[Message] | None:
        """Load a session with user feedback. Returns messages or None."""
        messages, _, _ = self.load(session_id)
        if messages is None:
            print(f"{Colors.RED}❌ Session not found: {session_id}{Colors.RESET}")
            return None
        system_count = sum(1 for m in messages if m.role == "system")
        print(
            f"{Colors.GREEN}✅ Session restored: {session_id}"
            f" ({len(messages)} messages, {system_count} system prompts){Colors.RESET}"
        )
        return messages

