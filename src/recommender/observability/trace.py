"""Append-only JSONL run traces.

One trace file per request (``runs/<request_id>/trace.jsonl``). Events include the
request/trace identity, a monotonic sequence number and UTC timestamp so a run can be
replayed and audited. Secrets are redacted before writing.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recommender.observability.logging import redact
from recommender.utils.io import ensure_dir, utc_now_iso


class TraceWriter:
    """Thread-safe append-only JSONL trace writer for a single request."""

    def __init__(self, path: Path | str, *, request_id: str, trace_id: str) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.request_id = request_id
        self.trace_id = trace_id
        self._sequence = 0
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def emit(
        self,
        event: str,
        payload: Mapping[str, Any] | None = None,
        *,
        stage: str | None = None,
    ) -> dict[str, Any]:
        """Append one trace event and return the written record."""
        with self._lock:
            self._sequence += 1
            record: dict[str, Any] = {
                "seq": self._sequence,
                "ts": utc_now_iso(),
                "request_id": self.request_id,
                "trace_id": self.trace_id,
                "event": event,
            }
            if stage is not None:
                record["stage"] = stage
            if payload is not None:
                record["payload"] = redact(dict(payload))
            self._handle.write(json.dumps(record, ensure_ascii=False, default=str))
            self._handle.write("\n")
            self._handle.flush()
            return record

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def trace_path(runs_dir: Path | str, request_id: str) -> Path:
    """Canonical per-request trace location."""
    return Path(runs_dir) / request_id / "trace.jsonl"


def read_trace(path: Path | str) -> list[dict[str, Any]]:
    """Read a trace file back into memory (used by tests and inspection tools)."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records
