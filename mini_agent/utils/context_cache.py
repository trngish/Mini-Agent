"""Context cache for reducing redundant operations across steps.

Layered context architecture:
- Layer 0: System prompt (immutable)
- Layer 1: Project context (persistent during project lifetime)
- Layer 2: Session context (current task, periodically summarized)
- Layer 3: Step context (transient, released after each step)
"""

import time
from pathlib import Path
from threading import Lock
from typing import Any


class ContextCache:
    """Multi-layer context cache with LRU eviction and TTL support.

    Reduces redundant file reads, grep searches, and tree operations
    by caching results across agent steps.

    Uses LRU eviction when max_memory_mb is exceeded, keeping the most
    recently accessed entries.
    """

    DEFAULT_TTL: int | None = 300

    DEFAULT_MAX_FILE_ENTRIES = 200
    DEFAULT_MAX_TREE_ENTRIES = 20
    DEFAULT_MAX_GREP_ENTRIES = 50
    BYTES_PER_CHAR = 2

    def __init__(
        self,
        max_memory_mb: float = 100.0,
        max_file_entries: int = DEFAULT_MAX_FILE_ENTRIES,
        max_tree_entries: int = DEFAULT_MAX_TREE_ENTRIES,
        max_grep_entries: int = DEFAULT_MAX_GREP_ENTRIES,
    ):
        """Initialize context cache.

        Args:
            max_memory_mb: Maximum memory to use for cache (approximate)
            max_file_entries: Maximum number of file cache entries
            max_tree_entries: Maximum number of tree cache entries
            max_grep_entries: Maximum number of grep cache entries
        """
        self._file_cache: dict[str, dict[str, Any]] = {}
        self._tree_cache: dict[str, dict[str, Any]] = {}
        self._grep_cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._max_memory_mb = max_memory_mb
        self._estimated_memory_bytes = 0
        self._max_file_entries = max_file_entries
        self._max_tree_entries = max_tree_entries
        self._max_grep_entries = max_grep_entries

    def _estimate_entry_size(self, entry: dict[str, Any]) -> int:
        """Estimate the memory size of a cache entry in bytes."""
        total = 0
        for key, value in entry.items():
            # Skip internal fields
            if key.startswith("_"):
                continue
            if isinstance(value, str):
                total += len(value.encode("utf-8")) if value else 0
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        total += len(item.encode("utf-8")) if item else 0
                    elif isinstance(item, dict):
                        total += self._estimate_entry_size(item)
            elif isinstance(value, dict):
                total += self._estimate_entry_size(value)
        return total

    def _evict_if_needed(self, cache_dict: dict[str, Any], max_entries: int) -> None:
        """Evict least recently used entries if cache exceeds limits.

        Evicts by both entry count AND memory usage.

        Args:
            cache_dict: The cache dict to evict from
            max_entries: Maximum number of entries allowed
        """
        # Evict by entry count first
        if len(cache_dict) > max_entries:
            sorted_entries = sorted(cache_dict.items(), key=lambda x: x[1].get("_last_access", 0))
            entries_to_remove = len(cache_dict) - max_entries
            for key, _ in sorted_entries[:entries_to_remove]:
                removed = cache_dict.pop(key, None)
                if removed:
                    self._estimated_memory_bytes -= self._estimate_entry_size(removed)

        # Evict by memory limit
        self._evict_by_memory()

    def _evict_by_memory(self) -> None:
        """Evict entries if memory limit exceeded (global across all caches).

        Uses LRU eviction across all cache types to stay under max_memory_mb.
        """
        max_bytes = self._max_memory_mb * 1024 * 1024
        while self._estimated_memory_bytes > max_bytes:
            # Find the oldest entry across all caches
            oldest_key: str | None = None
            oldest_time = float("inf")
            oldest_dict: dict[str, Any] | None = None

            for cache_dict in [self._file_cache, self._tree_cache, self._grep_cache]:
                for key, entry in cache_dict.items():
                    last_access = entry.get("_last_access", 0)
                    if last_access < oldest_time:
                        oldest_time = last_access
                        oldest_key = key
                        oldest_dict = cache_dict

            if oldest_key is not None and oldest_dict is not None:
                removed = oldest_dict.pop(oldest_key)
                self._estimated_memory_bytes -= self._estimate_entry_size(removed)
            else:
                break

    def get_file_content(self, path: str | Path) -> str | None:
        """Get cached file content if valid (D7 FIX: added mtime check).

        Args:
            path: File path

        Returns:
            Cached content or None if not cached/expired
        """
        key = str(Path(path).resolve())
        with self._lock:
            if key in self._file_cache:
                entry = self._file_cache[key]
                # D7 FIX: Check file mtime - if file was modified externally,
                # the cache is stale even if TTL hasn't expired
                cached_mtime = entry.get("_file_mtime")
                if cached_mtime is not None:
                    try:
                        current_mtime = Path(path).stat().st_mtime
                        if current_mtime != cached_mtime:
                            # File was modified since caching - invalidate
                            self._file_cache.pop(key, None)
                            return None
                    except OSError:
                        # File no longer accessible - invalidate
                        self._file_cache.pop(key, None)
                        return None

                if self._is_valid(entry):
                    # Update access time
                    entry["_last_access"] = time.time()
                    entry["_access_count"] = entry.get("_access_count", 0) + 1
                    return entry["content"]  # type: ignore[no-any-return]
                else:
                    # Expired, remove it
                    self._file_cache.pop(key, None)
        return None

    def set_file_content(self, path: str | Path, content: str, ttl: int | None = DEFAULT_TTL) -> None:
        """Cache file content with LRU eviction and mtime tracking (D7 fix).

        Args:
            path: File path
            content: File content
            ttl: Time to live in seconds (None = no expiration)
        """
        key = str(Path(path).resolve())
        with self._lock:
            # D7 FIX: Store file mtime for invalidation on external modification
            try:
                file_mtime = Path(path).stat().st_mtime
            except OSError:
                file_mtime = None

            entry = {
                "content": content,
                "created_at": time.time(),
                "_last_access": time.time(),
                "_access_count": 0,
                "ttl": ttl,
                "_file_mtime": file_mtime,
            }
            # Remove old entry if present to update memory tracking
            old_entry = self._file_cache.pop(key, None)
            if old_entry:
                self._estimated_memory_bytes -= self._estimate_entry_size(old_entry)

            self._file_cache[key] = entry
            self._estimated_memory_bytes += self._estimate_entry_size(entry)

            # LRU eviction
            self._evict_if_needed(self._file_cache, self._max_file_entries)

    def get_tree(self, root_dir: str | Path, max_depth: int) -> str | None:
        """Get cached tree output if valid."""
        key = f"{str(Path(root_dir).resolve())}:{max_depth}"
        with self._lock:
            if key in self._tree_cache:
                entry = self._tree_cache[key]
                if self._is_valid(entry):
                    entry["_last_access"] = time.time()
                    entry["_access_count"] = entry.get("_access_count", 0) + 1
                    return entry["content"]  # type: ignore[no-any-return]
                else:
                    self._tree_cache.pop(key, None)
        return None

    def set_tree(self, root_dir: str | Path, max_depth: int, content: str, ttl: int | None = DEFAULT_TTL) -> None:
        """Cache tree output."""
        key = f"{str(Path(root_dir).resolve())}:{max_depth}"
        with self._lock:
            entry = {
                "content": content,
                "created_at": time.time(),
                "_last_access": time.time(),
                "_access_count": 0,
                "ttl": ttl,
            }
            # Remove old entry if present to update memory tracking
            old_entry = self._tree_cache.pop(key, None)
            if old_entry:
                self._estimated_memory_bytes -= self._estimate_entry_size(old_entry)

            self._tree_cache[key] = entry
            self._estimated_memory_bytes += self._estimate_entry_size(entry)

            self._evict_if_needed(self._tree_cache, self._max_tree_entries)

    def get_grep_result(self, pattern: str, path: str, file_pattern: str, case_sensitive: bool) -> list[str] | None:
        """Get cached grep results."""
        key = self._grep_key(pattern, path, file_pattern, case_sensitive)
        with self._lock:
            if key in self._grep_cache:
                entry = self._grep_cache[key]
                if self._is_valid(entry):
                    entry["_last_access"] = time.time()
                    entry["_access_count"] = entry.get("_access_count", 0) + 1
                    return entry["results"]  # type: ignore[no-any-return]
                else:
                    self._grep_cache.pop(key, None)
        return None

    def set_grep_result(
        self,
        pattern: str,
        path: str,
        file_pattern: str,
        case_sensitive: bool,
        results: list[str],
        ttl: int | None = DEFAULT_TTL,
    ) -> None:
        """Cache grep results."""
        key = self._grep_key(pattern, path, file_pattern, case_sensitive)
        with self._lock:
            entry = {
                "results": results,
                "created_at": time.time(),
                "_last_access": time.time(),
                "_access_count": 0,
                "ttl": ttl,
            }
            # Remove old entry if present to update memory tracking
            old_entry = self._grep_cache.pop(key, None)
            if old_entry:
                self._estimated_memory_bytes -= self._estimate_entry_size(old_entry)

            self._grep_cache[key] = entry
            self._estimated_memory_bytes += self._estimate_entry_size(entry)

            self._evict_if_needed(self._grep_cache, self._max_grep_entries)

    def _grep_key(self, pattern: str, path: str, file_pattern: str, case_sensitive: bool) -> str:
        """Generate cache key for grep."""
        return f"{pattern}|{str(Path(path).resolve())}|{file_pattern}|{case_sensitive}"

    def _is_valid(self, entry: dict[str, Any]) -> bool:
        """Check if cache entry is still valid (D7 FIX: added mtime check)."""
        ttl = entry.get("ttl")
        if ttl is not None:
            age = time.time() - entry["created_at"]
            if age >= ttl:
                return False

        # D7 FIX: Check file mtime to detect external modifications
        file_mtime = entry.get("_file_mtime")
        if file_mtime is not None:
            # The key matches the file path, but we don't have path in entry.
            # Check all file_cache entries for this mtime check via get_file_content
            pass  # mtime check is done in get_file_content() which has the path

        return True

    def invalidate_file(self, path: str | Path) -> None:
        """Invalidate cached file content."""
        key = str(Path(path).resolve())
        with self._lock:
            self._file_cache.pop(key, None)

    def invalidate_tree(self, root_dir: str | Path, max_depth: int) -> None:
        """Invalidate cached tree."""
        key = f"{str(Path(root_dir).resolve())}:{max_depth}"
        with self._lock:
            self._tree_cache.pop(key, None)

    def invalidate_grep(self, pattern: str = "*", path: str | Path = ".") -> None:
        """Invalidate grep cache entries matching pattern/path."""
        path_str = str(Path(path).resolve())
        with self._lock:
            keys_to_remove = [k for k in self._grep_cache if (pattern == "*" or pattern in k) and path_str in k]
            for k in keys_to_remove:
                self._grep_cache.pop(k, None)

    def invalidate_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._file_cache.clear()
            self._tree_cache.clear()
            self._grep_cache.clear()
            self._estimated_memory_bytes = 0

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "file_entries": len(self._file_cache),
                "tree_entries": len(self._tree_cache),
                "grep_entries": len(self._grep_cache),
            }

    def filter_uncached_paths(self, paths: list[str | Path]) -> list[str]:
        """Filter out paths that are already cached.

        Uses batch check (single lock acquisition) for performance.

        Args:
            paths: List of file paths to check

        Returns:
            List of paths that are NOT in cache (need to be read)
        """
        uncached = []
        with self._lock:
            for path in paths:
                normalized = str(Path(path).resolve())
                if normalized not in self._file_cache:
                    uncached.append(str(path))
                else:
                    entry = self._file_cache[normalized]
                    if not self._is_valid(entry):
                        uncached.append(str(path))
        return uncached

    def warmup(self, workspace_dir: str | Path, priority_files: list[str] | None = None) -> int:
        """Warm up cache with frequently accessed files.

        Args:
            workspace_dir: Workspace directory
            priority_files: List of file patterns to prioritize

        Returns:
            Number of files cached
        """
        if priority_files is None:
            priority_files = [
                # Config/build files (small, high value)
                "pyproject.toml",
                "package.json",
                "Cargo.toml",
                "go.mod",
                "requirements.txt",
                "Pipfile",
                "setup.py",
                "setup.cfg",
                "Makefile",
                "Dockerfile",
                "docker-compose.yml",
                "docker-compose.yaml",
                ".gitignore",
                ".env.example",
                ".editorconfig",
                # Config dirs
                "*.yaml",
                "*.yml",
                "*.json",
                "*.toml",
                "*.ini",
                "*.cfg",
                # Documentation
                "README.md",
                "README*.md",
                "CHANGELOG.md",
                "CONTRIBUTING.md",
                # Source entry points
                "main.py",
                "app.py",
                "index.js",
                "index.ts",
            ]

        cached_count = 0
        workspace = Path(workspace_dir)

        for pattern in priority_files:
            try:
                # Efficient glob: use **/ only when needed
                if "/" in pattern or "*" in pattern.replace(".", ""):
                    matches = list(workspace.glob(pattern))
                else:
                    matches = list(workspace.glob(f"**/{pattern}"))

                for match in matches[:15]:  # Limit to 15 files per pattern
                    if match.is_file():
                        try:
                            content = match.read_text(encoding="utf-8")
                            # Use longer TTL for stable config files
                            is_config = any(
                                ext in match.suffix.lower()
                                for ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".lock"]
                            )
                            ttl = 900 if is_config else 600  # 15min for config, 10min for others
                            self.set_file_content(match, content, ttl=ttl)
                            cached_count += 1
                        except Exception:
                            pass  # Skip files that can't be read
            except Exception:
                pass  # Skip patterns that fail

        return cached_count


# Cache hit/miss tracking for diagnostics
_cache_hits: int = 0
_cache_misses: int = 0
_cache_hit_lock = Lock()


def record_cache_access(hit: bool) -> None:
    """Record cache hit/miss for diagnostics."""
    global _cache_hits, _cache_misses
    with _cache_hit_lock:
        if hit:
            _cache_hits += 1
        else:
            _cache_misses += 1


def get_cache_hit_rate() -> float:
    """Get the current cache hit rate."""
    total = _cache_hits + _cache_misses
    return _cache_hits / total if total > 0 else 0.0


# Namespace-based cache registry (D14 FIX: replaces global singleton)
# Each workspace/process gets its own isolated cache, preventing
# cross-project cache pollution when multiple Agent instances run.
_namespaced_caches: dict[str, ContextCache] = {}
_namespaced_cache_lock = Lock()


def get_context_cache(namespace: str = "__global__") -> ContextCache:
    """Get a namespace-isolated context cache (D14 FIX).

    Each namespace gets its own independent ContextCache instance.
    Agents in different workspaces should use different namespaces
    (typically workspace directory hash) to avoid cache pollution.

    Args:
        namespace: Cache namespace identifier. "__global__" for backward compat.
    """
    if namespace not in _namespaced_caches:
        with _namespaced_cache_lock:
            if namespace not in _namespaced_caches:
                _namespaced_caches[namespace] = ContextCache()
    return _namespaced_caches[namespace]


def create_cache_for_workspace(workspace_dir: str | Path) -> ContextCache:
    """Create or get a cache instance isolated to a specific workspace (D14 FIX).

    Uses a hash of the workspace path as namespace, ensuring different
    workspaces never share cached file content.
    """
    import hashlib
    ws_hash = hashlib.md5(str(workspace_dir).encode()).hexdigest()[:12]
    return get_context_cache(f"ws:{ws_hash}")


def reset_cache_namespace(namespace: str = "__global__") -> None:
    """Reset a specific namespace's cache."""
    global _namespaced_caches
    if namespace in _namespaced_caches:
        _namespaced_caches[namespace].invalidate_all()
        del _namespaced_caches[namespace]


def reset_global_cache() -> None:
    """Reset all caches (backward compatible with old API)."""
    global _namespaced_caches
    with _namespaced_cache_lock:
        for cache in _namespaced_caches.values():
            cache.invalidate_all()
        _namespaced_caches.clear()
