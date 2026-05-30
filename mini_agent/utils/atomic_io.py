"""Atomic file I/O utilities to eliminate duplication.

Provides atomic write/read operations using temp file pattern
to ensure data integrity on crashes during I/O.
"""

import json
from pathlib import Path
from typing import Any


def atomic_write_json(data: Any, path: Path) -> None:
    """Write JSON data atomically using temp file pattern.

    Writes to a temp file first, then renames to target.
    If rename fails (e.g., on Windows), falls back to direct write.

    Args:
        data: JSON-serializable data to write
        path: Target file path
    """
    path = Path(path)
    temp_path = path.with_suffix(".tmp")

    try:
        temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.rename(path)
    except OSError:
        # Fallback for Windows or if rename fails
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_read_json(path: Path) -> Any | None:
    """Read JSON data with error handling.

    Args:
        path: File path to read

    Returns:
        Parsed JSON data or None if file doesn't exist or is invalid
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


def atomic_write_text(text: str, path: Path, encoding: str = "utf-8") -> None:
    """Write text data atomically using temp file pattern.

    Args:
        text: Text content to write
        path: Target file path
        encoding: Text encoding (default: utf-8)
    """
    path = Path(path)
    temp_path = path.with_suffix(".tmp")

    try:
        temp_path.write_text(text, encoding=encoding)
        temp_path.rename(path)
    except OSError:
        # Fallback for Windows or if rename fails
        path.write_text(text, encoding=encoding)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_read_text(path: Path, encoding: str = "utf-8") -> str | None:
    """Read text data with error handling.

    Args:
        path: File path to read
        encoding: Text encoding (default: utf-8)

    Returns:
        Text content or None if file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError):
        return None
