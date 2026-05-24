"""Context cache for reducing redundant operations across steps.

Layered context architecture:
- Layer 0: System prompt (immutable)
- Layer 1: Project context (persistent during project lifetime)
- Layer 2: Session context (current task, periodically summarized)
- Layer 3: Step context (transient, released after each step)
"""

import time
from pathlib import Path
from typing import Any, Callable
from threading import Lock


class ContextCache:
    """Multi-layer context cache with TTL support.
    
    Reduces redundant file reads, grep searches, and tree operations
    by caching results across agent steps.
    """

    # TTL in seconds (None = no expiration)
    DEFAULT_TTL = 300  # 5 minutes

    def __init__(self, max_memory_mb: float = 100.0):
        """Initialize context cache.
        
        Args:
            max_memory_mb: Maximum memory to use for cache (approximate)
        """
        self._file_cache: dict[str, dict[str, Any]] = {}
        self._tree_cache: dict[str, dict[str, Any]] = {}
        self._grep_cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._max_memory_mb = max_memory_mb

    def get_file_content(self, path: str | Path) -> str | None:
        """Get cached file content if valid.
        
        Args:
            path: File path
            
        Returns:
            Cached content or None if not cached/expired
        """
        key = str(Path(path).resolve())
        with self._lock:
            if key in self._file_cache:
                entry = self._file_cache[key]
                if self._is_valid(entry):
                    # Update access time
                    entry['_last_access'] = time.time()
                    entry['_access_count'] += 1
                    return entry['content']
        return None

    def set_file_content(self, path: str | Path, content: str, ttl: int | None = DEFAULT_TTL) -> None:
        """Cache file content.
        
        Args:
            path: File path
            content: File content
            ttl: Time to live in seconds (None = no expiration)
        """
        key = str(Path(path).resolve())
        with self._lock:
            self._file_cache[key] = {
                'content': content,
                'created_at': time.time(),
                '_last_access': time.time(),
                '_access_count': 0,
                'ttl': ttl,
            }

    def get_tree(self, root_dir: str | Path, max_depth: int) -> str | None:
        """Get cached tree output if valid.
        
        Args:
            root_dir: Root directory
            max_depth: Tree depth
            
        Returns:
            Cached tree string or None
        """
        key = f"{str(Path(root_dir).resolve())}:{max_depth}"
        with self._lock:
            if key in self._tree_cache:
                entry = self._tree_cache[key]
                if self._is_valid(entry):
                    entry['_last_access'] = time.time()
                    entry['_access_count'] += 1
                    return entry['content']
        return None

    def set_tree(self, root_dir: str | Path, max_depth: int, content: str, ttl: int | None = DEFAULT_TTL) -> None:
        """Cache tree output."""
        key = f"{str(Path(root_dir).resolve())}:{max_depth}"
        with self._lock:
            self._tree_cache[key] = {
                'content': content,
                'created_at': time.time(),
                '_last_access': time.time(),
                '_access_count': 0,
                'ttl': ttl,
            }

    def get_grep_result(self, pattern: str, path: str, file_pattern: str, case_sensitive: bool) -> list[str] | None:
        """Get cached grep results."""
        key = self._grep_key(pattern, path, file_pattern, case_sensitive)
        with self._lock:
            if key in self._grep_cache:
                entry = self._grep_cache[key]
                if self._is_valid(entry):
                    entry['_last_access'] = time.time()
                    entry['_access_count'] += 1
                    return entry['results']
        return None

    def set_grep_result(self, pattern: str, path: str, file_pattern: str, case_sensitive: bool, 
                       results: list[str], ttl: int | None = DEFAULT_TTL) -> None:
        """Cache grep results."""
        key = self._grep_key(pattern, path, file_pattern, case_sensitive)
        with self._lock:
            self._grep_cache[key] = {
                'results': results,
                'created_at': time.time(),
                '_last_access': time.time(),
                '_access_count': 0,
                'ttl': ttl,
            }

    def _grep_key(self, pattern: str, path: str, file_pattern: str, case_sensitive: bool) -> str:
        """Generate cache key for grep."""
        return f"{pattern}|{str(Path(path).resolve())}|{file_pattern}|{case_sensitive}"

    def _is_valid(self, entry: dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        ttl = entry.get('ttl')
        if ttl is None:
            return True  # No TTL = never expires
        age = time.time() - entry['created_at']
        return age < ttl

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
            keys_to_remove = [
                k for k in self._grep_cache.keys()
                if (pattern == "*" or pattern in k) and path_str in k
            ]
            for k in keys_to_remove:
                self._grep_cache.pop(k, None)

    def invalidate_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._file_cache.clear()
            self._tree_cache.clear()
            self._grep_cache.clear()

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                'file_entries': len(self._file_cache),
                'tree_entries': len(self._tree_cache),
                'grep_entries': len(self._grep_cache),
            }


# Global singleton instance
_global_cache: ContextCache | None = None


def get_context_cache() -> ContextCache:
    """Get the global context cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ContextCache()
    return _global_cache