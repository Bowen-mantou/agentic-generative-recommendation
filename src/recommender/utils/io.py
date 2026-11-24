"""Small filesystem helpers used by data and artifact pipelines.

Writes are atomic (temp file + ``os.replace``) so a crashed job cannot leave a
half-written index manifest or catalog behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (including parents) and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_now() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(tz=timezone.utc)


def utc_now_iso() -> str:
    """ISO-8601 timestamp with explicit UTC offset."""
    return utc_now().isoformat()


def read_json(path: Path | str) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Atomically write UTF-8 JSON."""
    target = Path(path)
    ensure_dir(target.parent)
    serialized = json.dumps(payload, ensure_ascii=False, indent=indent, default=str)
    _atomic_write_text(target, serialized + "\n")
    return target


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield record


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    """Atomically write JSONL rows."""
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
    try:
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return target


def file_sha256(path: Path | str) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str, *, length: int = 16) -> str:
    """Deterministic short hash used for query dedup / cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _atomic_write_text(target: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
