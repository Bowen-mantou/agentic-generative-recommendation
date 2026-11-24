"""Structured logging with request/trace context and secret redaction.

Production code logs through :func:`get_logger`; ``print`` is not used (AGENTS.md §4).
Log records are JSON by default so they can be shipped to any collector, and every
record carries the active ``request_id`` / ``trace_id``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from recommender.observability.context import get_request_context

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|auth[_-]?token|access[_-]?token|refresh[_-]?token"
    r"|secret|password|passwd|cookie|session[_-]?id)",
    re.IGNORECASE,
)
_REDACTED = "***REDACTED***"

# Attributes that ``logging`` puts on every record; anything else was added by the caller.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "request_id",
        "trace_id",
    }
)


def redact(value: Any) -> Any:
    """Recursively redact secret-looking keys in mappings."""
    if isinstance(value, Mapping):
        return {
            str(key): (_REDACTED if _SENSITIVE_KEY.search(str(key)) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class ContextFilter(logging.Filter):
    """Attach the active request/trace identity to each record."""

    def __init__(self, *, include_context: bool = True) -> None:
        super().__init__()
        self.include_context = include_context

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.include_context:
            return True
        context = get_request_context()
        if context is not None:
            record.request_id = context.request_id
            record.trace_id = context.trace_id
            if not hasattr(record, "user_id") and context.user_id is not None:
                record.user_id = context.user_id
        return True


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = _REDACTED if _SENSITIVE_KEY.search(str(key)) else redact(value)
        for attr in ("request_id", "trace_id", "user_id"):
            value = getattr(record, attr, None)
            if value:
                payload[attr] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return super().format(record)


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    include_context: bool = True,
    stream: Any | None = None,
) -> None:
    """Configure the root logger once, replacing handlers idempotently."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    handler.addFilter(ContextFilter(include_context=include_context))
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Third-party libraries stay at WARNING unless explicitly raised.
    for noisy in ("urllib3", "httpx", "httpcore", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (never configures handlers itself)."""
    return logging.getLogger(name)
