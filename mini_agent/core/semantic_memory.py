"""Semantic Memory Layer for Mini-Agent (D11 FIX).

Extracts structured, cross-session memories from agent conversations.
Replaces the raw "messages = JSON snapshot" approach with categorized,
searchable semantic memories that persist across sessions.

Categories:
- decision: Architectural choices, technology selections, design decisions
- preference: User habits, preferred approaches, style conventions
- finding: Analysis conclusions, investigation results, discovered facts
- task: Action items, planned work, pending follow-ups
- code_pattern: Repeated coding patterns, API usage conventions
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class MemoryEntry:
    """A single extracted semantic memory."""

    id: str
    category: str
    content: str
    source_session: str = ""
    source_step: int = 0
    timestamp: str = ""
    relevance_score: float = 1.0
    access_count: int = 0
    _last_access: float = 0.0

    def access(self) -> None:
        """Mark as accessed (updates relevance)."""
        import time
        self.access_count += 1
        self._last_access = time.time()

    def decay(self, days_since_creation: float) -> None:
        """Apply time-based relevance decay."""
        # Half-life of 30 days
        self.relevance_score = max(0.1, self.relevance_score * (0.5 ** (days_since_creation / 30)))


class SemanticMemory:
    """Cross-session semantic memory extractor and store.

    Extracts structured memories from assistant messages using
    pattern-based extraction (no LLM overhead). Persists memories
    to disk and provides context injection for new sessions.
    """

    # Extraction patterns by category
    EXTRACTION_PATTERNS: dict[str, list[tuple[str, float]]] = {
        "decision": [
            # Chinese patterns
            (r"(?:决定|选择|采用|最终使用|应该用|推荐使用)\s*[:：]?\s*(.+?)(?:[。\n]|$)", 0.9),
            (r"(?:建议|推荐)\s*(?:使用|采用|用)\s*(.+?)(?:[。\n]|$)", 0.85),
            (r"(?:最佳方案|最优选择|最终方案)[是为：:]\s*(.+?)(?:[。\n]|$)", 0.9),
            # English patterns
            (r"(?:decided to|chose to|opted for|went with)\s+(.+?)(?:[.\n]|$)", 0.85),
            (r"(?:recommend|suggest)\s+(?:using|adopting)?\s*(.+?)(?:[.\n]|$)", 0.8),
        ],
        "preference": [
            (r"(?:偏好|更喜欢|倾向于|习惯(?:了)?)\s*(.+?)(?:[。\n]|$)", 0.85),
            (r"(?:prefer[s]?|likes?|usually)\s+(.+?)(?:[.\n]|$)", 0.8),
            (r"(?:风格|风格偏好)[是为：:]\s*(.+?)(?:[。\n]|$)", 0.85),
        ],
        "finding": [
            # Bullet points starting with markers
            (r"(?:^|\n)\s*[-*•]\s+(.+?)(?:\n|$)", 0.7),
            (r"(?:^|\n)\s*\d+[.)]\s+(.+?)(?:\n|$)", 0.7),
            # Analysis conclusions
            (r"(?:发现|分析显示|结果表明|得出结论)[是为：:]\s*(.+?)(?:[。\n]|$)", 0.85),
            (r"(?:问题根因|根因是|原因是)[是为：:]\s*(.+?)(?:[。\n]|$)", 0.9),
        ],
        "task": [
            (r"(?:接下来|下一步|后续)\s*(?:需要|要|应该)\s*(.+?)(?:[。\n]|$)", 0.8),
            (r"(?:TODO|FIXME|HACK|XXX)[:：]?\s*(.+?)(?:\n|$)", 0.75),
            (r"(?:计划|待办|pending)[:：]?\s*(.+?)(?:[。\n]|$)", 0.8),
        ],
        "code_pattern": [
            # API usage
            (r"(?:使用|调用|通过)\s*(?:`|'|\")?(\w+\.\w+(?:\(\))?)(?:`|'|\")?\s*(?:来|进行|实现)", 0.75),
            (r"(?:use[sd]?|call[sd]?|invoke[sd]?)\s+(?:`|'|\")?(\w+\.\w+(?:\(\))?)(?:`|'|\")?", 0.7),
            # File path patterns
            (r"(?:修改|编辑|创建|删除)\s*(?:了|过)?\s*(?:`|'|\")?([/\w.\\-]+\.\w+)(?:`|'|\")?", 0.7),
        ],
    }

    # Minimum content length to consider for extraction
    MIN_CONTENT_LENGTH = 50

    # Maximum memories per category
    MAX_MEMORIES_PER_CATEGORY = 50

    # Path for persistent storage
    DEFAULT_MEMORY_DIR = Path.home() / ".mini-agent" / "memory"

    def __init__(self, workspace_dir: Path | None = None, memory_dir: Path | None = None):
        """Initialize semantic memory.

        Args:
            workspace_dir: Workspace directory for namespace isolation.
            memory_dir: Custom memory storage directory.
        """
        self._lock = Lock()
        self._entries: dict[str, MemoryEntry] = {}
        self._loaded = False

        if memory_dir:
            self._memory_dir = memory_dir
        elif workspace_dir:
            ws_hash = hashlib.md5(str(workspace_dir.absolute()).encode()).hexdigest()[:12]
            self._memory_dir = self.DEFAULT_MEMORY_DIR / ws_hash
        else:
            self._memory_dir = self.DEFAULT_MEMORY_DIR / "__global__"

        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file = self._memory_dir / "memories.json"

    # --- Extraction ---

    def extract_from_message(self, content: str, session_id: str = "", step: int = 0) -> list[MemoryEntry]:
        """Extract semantic memories from an assistant message.

        Uses regex patterns to identify structured information without
        requiring an LLM call for extraction itself.

        Args:
            content: The assistant's message content.
            session_id: Source session identifier.
            step: Step number within the session.

        Returns:
            List of extracted MemoryEntry objects.
        """
        if not content or len(content) < self.MIN_CONTENT_LENGTH:
            return []

        entries: list[MemoryEntry] = []
        seen_content: set[str] = set()
        timestamp = datetime.now().isoformat()

        for category, patterns in self.EXTRACTION_PATTERNS.items():
            category_entries = 0
            for pattern, base_score in patterns:
                if category_entries >= 10:  # Max 10 per category per message
                    break
                for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                    extracted = match.group(1).strip()
                    # Deduplicate and filter noise
                    if len(extracted) < 8 or len(extracted) > 500:
                        continue
                    normalized = extracted.lower().strip(".,;:!?。，；：！？")  # noqa: RUF001
                    if normalized in seen_content:
                        continue
                    seen_content.add(normalized)

                    entry = MemoryEntry(
                        id=str(uuid.uuid4())[:12],
                        category=category,
                        content=extracted,
                        source_session=session_id,
                        source_step=step,
                        timestamp=timestamp,
                        relevance_score=base_score,
                    )
                    entries.append(entry)
                    category_entries += 1

        return entries

    def extract_from_session(self, messages: list[Any], session_id: str = "") -> list[MemoryEntry]:
        """Extract memories from all assistant messages in a session.

        Args:
            messages: List of Message objects from the session.
            session_id: Session identifier.

        Returns:
            All extracted MemoryEntry objects.
        """
        all_entries: list[MemoryEntry] = []
        step = 0

        for msg in messages:
            if hasattr(msg, "role") and msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content) if msg.content else ""
                if content:
                    entries = self.extract_from_message(content, session_id, step)
                    all_entries.extend(entries)
            step += 1

        return all_entries

    # --- Persistence ---

    def _load(self) -> None:
        """Load memories from disk."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                if self._memory_file.exists():
                    import json
                    data = json.loads(self._memory_file.read_text(encoding="utf-8"))
                    for entry_data in data.get("entries", []):
                        entry = MemoryEntry(**entry_data)
                        self._entries[entry.id] = entry
            except Exception:
                pass
            self._loaded = True

    def _save(self) -> None:
        """Save memories to disk atomically."""
        try:
            import json
            data = {
                "entries": [
                    {
                        "id": e.id,
                        "category": e.category,
                        "content": e.content,
                        "source_session": e.source_session,
                        "source_step": e.source_step,
                        "timestamp": e.timestamp,
                        "relevance_score": e.relevance_score,
                        "access_count": e.access_count,
                        "_last_access": e._last_access,
                    }
                    for e in self._entries.values()
                ],
                "updated": datetime.now().isoformat(),
            }
            tmp_path = self._memory_file.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(self._memory_file)
        except Exception:
            pass

    def add_entries(self, entries: list[MemoryEntry]) -> int:
        """Add extracted memories to the store.

        Args:
            entries: Memory entries to add.

        Returns:
            Number of new entries added (excluding duplicates).
        """
        self._load()
        added = 0
        with self._lock:
            for entry in entries:
                # Deduplicate by normalized content
                normalized = entry.content.lower().strip()
                is_duplicate = any(
                    e.content.lower().strip() == normalized
                    and e.category == entry.category
                    for e in self._entries.values()
                )
                if not is_duplicate:
                    self._entries[entry.id] = entry
                    added += 1
            if added > 0:
                self._prune_by_category()
                self._save()
        return added

    def _prune_by_category(self) -> None:
        """Remove oldest/lowest-relevance memories when per-category limit exceeded."""
        by_category: dict[str, list[MemoryEntry]] = {}
        for entry in self._entries.values():
            by_category.setdefault(entry.category, []).append(entry)

        for category, entries in by_category.items():
            if len(entries) > self.MAX_MEMORIES_PER_CATEGORY:
                # Sort by relevance (ascending) and remove lowest
                entries.sort(key=lambda e: e.relevance_score)
                to_remove = entries[: len(entries) - self.MAX_MEMORIES_PER_CATEGORY]
                for e in to_remove:
                    del self._entries[e.id]

    # --- Query ---

    def get_context_for_injection(self, max_entries: int = 8) -> str:
        """Get top memories formatted for system prompt injection.

        Returns the most relevant, recent memories as a compact
        markdown block suitable for prepending to system prompts.

        Args:
            max_entries: Maximum number of memory entries to include.

        Returns:
            Formatted string for system prompt injection, or empty string.
        """
        self._load()
        with self._lock:
            if not self._entries:
                return ""

            # Sort by relevance * recency
            import time
            now = time.time()
            scored = []
            for entry in self._entries.values():
                # Boost recently accessed + high initial relevance
                recency_boost = min(2.0, 1.0 + (entry._last_access / max(now, 1)) if entry._last_access else 1.0)
                score = entry.relevance_score * recency_boost
                scored.append((score, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:max_entries]

            if not top:
                return ""

            lines = ["\n## Cross-Session Memory (from previous conversations)\n"]
            lines.append("The following insights were extracted from past sessions in this workspace:\n")

            by_cat: dict[str, list[str]] = {}
            for _, entry in top:
                by_cat.setdefault(entry.category, []).append(f"- {entry.content}")

            cat_labels = {
                "decision": "Decisions made",
                "preference": "User preferences",
                "finding": "Previous findings",
                "task": "Pending/planned tasks",
                "code_pattern": "Code patterns used",
            }

            for cat, items in by_cat.items():
                label = cat_labels.get(cat, cat)
                lines.append(f"### {label}")
                lines.extend(items)
                lines.append("")

            return "\n".join(lines)

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[MemoryEntry]:
        """Simple keyword search across memories.

        Args:
            query: Search query.
            category: Optional category filter.
            top_k: Maximum results.

        Returns:
            Matching MemoryEntry objects sorted by relevance.
        """
        self._load()
        query_lower = query.lower()
        results: list[tuple[float, MemoryEntry]] = []

        with self._lock:
            for entry in self._entries.values():
                if category and entry.category != category:
                    continue
                content_lower = entry.content.lower()
                # Simple overlap score
                query_terms = query_lower.split()
                matches = sum(1 for t in query_terms if t in content_lower)
                if matches > 0:
                    score = matches / len(query_terms) * entry.relevance_score
                    results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:top_k]]

    def get_stats(self) -> dict[str, int]:
        """Get memory statistics."""
        self._load()
        with self._lock:
            stats: dict[str, int] = {"total": len(self._entries)}
            for entry in self._entries.values():
                stats[entry.category] = stats.get(entry.category, 0) + 1
            return stats
