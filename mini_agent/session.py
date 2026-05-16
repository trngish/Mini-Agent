"""Session save/resume manager with index caching."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schema import FunctionCall, Message, ToolCall


class SessionManager:
    """Manages session persistence (save/resume/list) with index caching.
    
    Maintains an index file for faster session listing without reading
    all session files.
    """

    INDEX_FILENAME = ".session_index.json"
    MAX_SESSIONS_IN_INDEX = 1000

    def __init__(self, session_dir: Optional[Path] = None):
        self.session_dir = session_dir or Path.home() / ".mini-agent" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._index: list[dict] | None = None
        self._index_loaded = False

    def _get_index_path(self) -> Path:
        """Get path to index file."""
        return self.session_dir / self.INDEX_FILENAME

    def _load_index(self) -> list[dict]:
        """Load session index from disk."""
        index_path = self._get_index_path()
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def _save_index(self, index: list[dict]) -> None:
        """Save session index to disk."""
        index_path = self._get_index_path()
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _ensure_index_loaded(self) -> None:
        """Ensure index is loaded."""
        if not self._index_loaded:
            self._index = self._load_index()
            self._index_loaded = True

    def _invalidate_index(self) -> None:
        """Invalidate cached index."""
        self._index = None
        self._index_loaded = False

    def _serialize_messages(self, messages: list[Message]) -> list[dict]:
        return [msg.model_dump() for msg in messages]

    def _deserialize_messages(self, data: list[dict]) -> list[Message]:
        return [Message(**msg) for msg in data]

    def save(self, messages: list[Message], label: str = "") -> str:
        """Save messages to a session file. Returns session ID."""
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        data = {
            "id": session_id,
            "label": label,
            "created": timestamp,
            "messages": self._serialize_messages(messages),
        }
        path = self.session_dir / f"{session_id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # Update index
        self._ensure_index_loaded()
        if self._index is None:
            self._index = []
        
        # Add to index
        index_entry = {
            "id": session_id,
            "label": label,
            "created": timestamp,
            "message_count": len(messages),
        }
        self._index.insert(0, index_entry)
        
        # Trim index if too large
        if len(self._index) > self.MAX_SESSIONS_IN_INDEX:
            self._index = self._index[:self.MAX_SESSIONS_IN_INDEX]
        
        self._save_index(self._index)
        return session_id

    def load(self, session_id: str) -> Optional[list[Message]]:
        """Load messages from a session file."""
        path = self.session_dir / f"{session_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return self._deserialize_messages(data["messages"])

    def list_sessions(self) -> list[dict]:
        """List all saved sessions using cached index."""
        self._ensure_index_loaded()
        
        if self._index is None or not self._index:
            # Fall back to scanning directory
            return self._list_sessions_scan()
        
        # Filter to existing sessions
        sessions = []
        for entry in self._index:
            path = self.session_dir / f"{entry['id']}.json"
            if path.exists():
                sessions.append({
                    "id": entry.get("id", path.stem),
                    "label": entry.get("label", ""),
                    "created": entry.get("created", ""),
                    "messages": entry.get("message_count", 0),
                })
            else:
                # Session file no longer exists, remove from index
                pass
        
        return sessions

    def _list_sessions_scan(self) -> list[dict]:
        """List sessions by scanning directory (fallback)."""
        sessions = []
        for path in sorted(
            self.session_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        ):
            if path.stem == self.INDEX_FILENAME:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "id": data.get("id", path.stem),
                    "label": data.get("label", ""),
                    "created": data.get("created", ""),
                    "messages": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session file."""
        path = self.session_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            
            # Update index
            self._ensure_index_loaded()
            if self._index is not None:
                self._index = [s for s in self._index if s.get("id") != session_id]
                self._save_index(self._index)
            
            return True
        return False

    def clear_index(self) -> None:
        """Clear the session index cache."""
        self._invalidate_index()
        index_path = self._get_index_path()
        if index_path.exists():
            index_path.unlink()